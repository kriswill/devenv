//! Kitty graphics passthrough.
//!
//! The child's output is fed to a libghostty virtual terminal and the real
//! terminal is repainted from it, so an image the child transmits with the
//! [Kitty graphics protocol](https://sw.kovidgoyal.net/kitty/graphics-protocol/)
//! never reaches the real terminal on its own: the VT parses the APC
//! sequence, stores the image and its placement, and answers the child.
//!
//! This module closes the gap from the VT side. After each rendered frame
//! [`ImageSync::sync`] reads the placements the VT holds for the active
//! screen and mirrors them on the real terminal: images are transmitted once
//! per (id, generation) under a private id range, placements are positioned
//! with explicit cursor moves so they land on the rows the renderer drew,
//! clipped to the content area (never the status line), and removed again
//! when the VT no longer has them. Every command carries `q=2` so the real
//! terminal stays silent — its replies would otherwise arrive on stdin as
//! keystrokes.
//!
//! The VT also needs a PNG decoder and a storage limit before it accepts any
//! image data; see [`install_png_decoder`] and [`enable`].

use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use crossterm::{cursor, queue};
use libghostty_vt::alloc::{Allocator, Bytes};
use libghostty_vt::kitty::graphics::{
    self, Compression, DecodePng, DecodedImage, ImageFormat, Layer, PlacementIterator,
};
use libghostty_vt::terminal::Terminal;
use std::collections::HashMap;
use std::io::{self, Write};

/// Image storage budget for the virtual terminal (Ghostty's own default).
/// A non-zero limit is what enables kitty graphics in libghostty-vt.
pub const IMAGE_STORAGE_LIMIT: u64 = 320 * 1024 * 1024;

/// Image ids used on the real terminal live above this base so they never
/// collide with images placed there before `devenv shell` took over (or by
/// an outer multiplexer). The VT's own id space is the child's; the two are
/// mapped through [`real_image_id`].
const REAL_ID_BASE: u32 = 1 << 30;
const REAL_ID_MASK: u32 = REAL_ID_BASE - 1;

/// Kitty caps a single APC payload at 4096 bytes of base64.
const CHUNK_BYTES: usize = 4096;

/// PNG decoder for libghostty-vt, on the `png` crate. libghostty only stores
/// RGBA8, so palette, grayscale and 16-bit images are expanded on the way in.
#[derive(Default)]
struct PngDecoder;

impl DecodePng for PngDecoder {
    fn decode_png<'alloc>(
        &mut self,
        alloc: &'alloc Allocator<'_>,
        data: &[u8],
    ) -> Option<DecodedImage<'alloc>> {
        use png::{ColorType, Decoder, Transformations};

        let mut decoder = Decoder::new(io::Cursor::new(data));
        decoder.set_transformations(Transformations::ALPHA | Transformations::STRIP_16);
        let mut reader = decoder.read_info().ok()?;
        let mut buf = vec![0u8; reader.output_buffer_size()?];
        let info = reader.next_frame(&mut buf).ok()?;
        let pixels = &buf[..info.buffer_size()];
        let rgba: Vec<u8> = match info.color_type {
            ColorType::Rgba => pixels.to_vec(),
            ColorType::Rgb => pixels
                .chunks_exact(3)
                .flat_map(|p| [p[0], p[1], p[2], 0xff])
                .collect(),
            ColorType::GrayscaleAlpha => pixels
                .chunks_exact(2)
                .flat_map(|p| [p[0], p[0], p[0], p[1]])
                .collect(),
            ColorType::Grayscale => pixels.iter().flat_map(|&g| [g, g, g, 0xff]).collect(),
            ColorType::Indexed => return None,
        };
        let mut bytes = Bytes::new_with_alloc(alloc, rgba.len()).ok()?;
        bytes.copy_from_slice(&rgba);
        Some(DecodedImage {
            width: info.width,
            height: info.height,
            data: bytes,
        })
    }
}

/// Register the PNG decoder with libghostty-vt for the calling thread.
///
/// libghostty-rs keeps the decoder in a thread-local (terminals are `!Send`,
/// so decoding always happens on the thread that owns the terminal), which
/// means it must be installed on the VT thread itself, before the terminal
/// is created. Calling it again on the same thread is a no-op.
pub fn install_png_decoder() {
    thread_local! {
        static INSTALLED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }
    if INSTALLED.replace(true) {
        return;
    }
    if let Err(e) = graphics::set_png_decoder(Some(Box::new(PngDecoder))) {
        tracing::warn!("kitty graphics: failed to install PNG decoder: {e}");
        INSTALLED.set(false);
    }
}

/// Enable kitty graphics on a virtual terminal. Local transfer mediums are
/// allowed because the child runs on this host: `kitten icat`, for one,
/// prefers a temp file over inline base64 when it can. Temp-file transfers
/// are confined to the process temp dir, which is where clients honouring
/// `TMPDIR` write them.
pub fn enable(vt: &mut Terminal<'_, '_>) -> Result<(), libghostty_vt::Error> {
    let temp_dir = std::env::temp_dir();
    vt.set_kitty_image_storage_limit(IMAGE_STORAGE_LIMIT)?
        .set_kitty_image_from_file_allowed(true)?
        .set_kitty_image_temp_file_dir(Some(&temp_dir))?
        .set_kitty_image_from_shared_mem_allowed(true)?;
    Ok(())
}

/// Whether an `on_pty_write` payload is a kitty graphics response
/// (`ESC _ G … ESC \`). These are answers to the child's own image commands
/// and must reach it: `kitten icat` waits for the `a=q` capability reply
/// before it sends anything.
pub fn is_kitty_graphics_reply(data: &[u8]) -> bool {
    data.starts_with(b"\x1b_G") && data.ends_with(b"\x1b\\")
}

/// The id an image is known by on the real terminal.
fn real_image_id(vt_image_id: u32) -> u32 {
    REAL_ID_BASE | (vt_image_id & REAL_ID_MASK)
}

/// One placement as it should appear on the real terminal.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Placement {
    image: u32,
    /// Real-terminal row/column (already offset by the renderer's takeover
    /// offset and clipped to the content area).
    row: u16,
    col: u16,
    cols: u32,
    rows: u32,
    src: SourceRect,
    x_offset: u32,
    y_offset: u32,
    z: i32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SourceRect {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

/// A placement's position and extent inside the viewport, before clipping.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Extent {
    /// Viewport row/col of the top-left cell; negative when scrolled above.
    row: i32,
    col: i32,
    rows: u32,
    cols: u32,
}

/// Clip a placement to the visible content area, trimming the source
/// rectangle proportionally so the visible part keeps its scale. Returns
/// `None` when nothing is left.
fn clip(
    extent: Extent,
    src: SourceRect,
    visible_rows: usize,
    visible_cols: usize,
) -> Option<(Extent, SourceRect)> {
    if extent.rows == 0 || extent.cols == 0 || visible_rows == 0 || visible_cols == 0 {
        return None;
    }
    let cut = |pos: i32, len: u32, limit: usize| -> Option<(u32, u32, i32)> {
        let start_cut = pos.min(0).unsigned_abs();
        let end = pos as i64 + len as i64;
        let end_cut = (end - limit as i64).max(0) as u32;
        let kept = len.checked_sub(start_cut)?.checked_sub(end_cut)?;
        (kept > 0).then_some((start_cut, kept, pos + start_cut as i32))
    };
    let (top_cut, rows, row) = cut(extent.row, extent.rows, visible_rows)?;
    let (left_cut, cols, col) = cut(extent.col, extent.cols, visible_cols)?;
    let scale = |value: u32, part: u32, whole: u32| -> u32 {
        (value as u64 * part as u64 / whole as u64) as u32
    };
    let clipped_src = SourceRect {
        x: src.x + scale(src.width, left_cut, extent.cols),
        y: src.y + scale(src.height, top_cut, extent.rows),
        width: scale(src.width, cols, extent.cols).max(1),
        height: scale(src.height, rows, extent.rows).max(1),
    };
    Some((
        Extent {
            row,
            col,
            rows,
            cols,
        },
        clipped_src,
    ))
}

/// Chunked `a=t` transmission of raw pixel data under a real-terminal id.
fn transmit_commands(
    real_id: u32,
    format: u8,
    width: u32,
    height: u32,
    zlib: bool,
    data: &[u8],
) -> Vec<u8> {
    let encoded = BASE64.encode(data);
    let chunks: Vec<&[u8]> = if encoded.is_empty() {
        vec![&[][..]]
    } else {
        encoded.as_bytes().chunks(CHUNK_BYTES).collect()
    };
    let last = chunks.len() - 1;
    let mut out = Vec::with_capacity(encoded.len() + chunks.len() * 48);
    for (index, chunk) in chunks.iter().enumerate() {
        let more = u8::from(index != last);
        if index == 0 {
            out.extend_from_slice(
                format!(
                    "\x1b_Ga=t,t=d,q=2,i={real_id},f={format},s={width},v={height}{},m={more};",
                    if zlib { ",o=z" } else { "" }
                )
                .as_bytes(),
            );
        } else {
            out.extend_from_slice(format!("\x1b_Gm={more};").as_bytes());
        }
        out.extend_from_slice(chunk);
        out.extend_from_slice(b"\x1b\\");
    }
    out
}

/// `a=p` placement at the cursor. `C=1` keeps the real cursor where the
/// renderer put it.
fn place_command(real_placement: u32, p: &Placement) -> String {
    format!(
        "\x1b_Ga=p,q=2,C=1,i={},p={},c={},r={},x={},y={},w={},h={},X={},Y={},z={}\x1b\\",
        p.image,
        real_placement,
        p.cols,
        p.rows,
        p.src.x,
        p.src.y,
        p.src.width,
        p.src.height,
        p.x_offset,
        p.y_offset,
        p.z
    )
}

/// Delete one placement, keeping the image data for reuse.
fn delete_placement_command(real_image: u32, real_placement: u32) -> String {
    format!("\x1b_Ga=d,d=i,q=2,i={real_image},p={real_placement}\x1b\\")
}

/// Delete an image and every placement of it.
fn delete_image_command(real_image: u32) -> String {
    format!("\x1b_Ga=d,d=I,q=2,i={real_image}\x1b\\")
}

/// Placements are keyed by the VT's (image, placement) ids plus an ordinal,
/// since the protocol allows several id-less placements of one image.
type PlacementKey = (u32, u32, u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Transmitted {
    generation: u64,
}

/// Mirrors the VT's kitty placements onto the real terminal.
#[derive(Default)]
pub struct ImageSync {
    iterator: Option<PlacementIterator<'static>>,
    /// VT image id → what the real terminal currently holds under
    /// `real_image_id(id)`.
    transmitted: HashMap<u32, Transmitted>,
    /// Placements currently shown on the real terminal, with their real
    /// placement ids.
    placed: HashMap<PlacementKey, (u32, Placement)>,
    next_placement_id: u32,
}

fn vt_err(e: libghostty_vt::Error) -> io::Error {
    io::Error::other(format!("kitty graphics: {e}"))
}

impl ImageSync {
    fn allocate_placement_id(&mut self) -> u32 {
        self.next_placement_id = self.next_placement_id.wrapping_add(1).max(1) & REAL_ID_MASK;
        self.next_placement_id.max(1)
    }

    /// Bring the real terminal's placements in line with the VT's active
    /// screen. `row_offset` is the renderer's takeover offset (real row =
    /// VT row + offset) and `visible_rows` how many VT rows are on screen
    /// below it. Returns whether anything was written — the caller must then
    /// reposition the cursor, because placements are made at the cursor.
    pub fn sync(
        &mut self,
        stdout: &mut impl Write,
        vt: &Terminal<'_, '_>,
        row_offset: u16,
        visible_rows: usize,
    ) -> io::Result<bool> {
        // Errors here mean kitty graphics is unavailable in this VT; behave
        // as before this feature existed.
        let Ok(graphics) = vt.kitty_graphics() else {
            return Ok(false);
        };
        if self.iterator.is_none() {
            self.iterator = PlacementIterator::new().ok();
        }
        let Some(iterator) = self.iterator.as_mut() else {
            return Ok(false);
        };

        let visible_cols = vt.cols().unwrap_or(0) as usize;
        let mut desired: Vec<(PlacementKey, Placement, u64)> = Vec::new();
        let mut ordinals: HashMap<(u32, u32), u32> = HashMap::new();
        {
            let mut placements = iterator.update(&graphics).map_err(vt_err)?;
            placements.set_layer(Layer::All).map_err(vt_err)?;
            while let Some(placement) = placements.next() {
                // Unicode-placeholder placements render through text cells
                // the row renderer already copies; nothing to mirror.
                if placement.is_virtual().unwrap_or(false) {
                    continue;
                }
                let (Ok(image_id), Ok(placement_id)) =
                    (placement.image_id(), placement.placement_id())
                else {
                    continue;
                };
                let Some(image) = graphics.image(image_id) else {
                    continue;
                };
                let Ok(info) = placement.placement_render_info(&image, vt) else {
                    continue;
                };
                if !info.viewport_visible {
                    continue;
                }
                let extent = Extent {
                    row: info.viewport_row,
                    col: info.viewport_col,
                    rows: info.grid_rows,
                    cols: info.grid_cols,
                };
                let src = SourceRect {
                    x: info.source_x,
                    y: info.source_y,
                    width: info.source_width,
                    height: info.source_height,
                };
                let Some((extent, src)) = clip(extent, src, visible_rows, visible_cols) else {
                    continue;
                };
                let ordinal = ordinals.entry((image_id, placement_id)).or_insert(0);
                let key = (image_id, placement_id, *ordinal);
                *ordinal += 1;
                let generation = image.generation().unwrap_or(0);
                desired.push((
                    key,
                    Placement {
                        image: real_image_id(image_id),
                        row: (extent.row as usize + row_offset as usize).min(u16::MAX as usize)
                            as u16,
                        col: (extent.col as usize).min(u16::MAX as usize) as u16,
                        cols: extent.cols,
                        rows: extent.rows,
                        src,
                        x_offset: placement.x_offset().unwrap_or(0),
                        y_offset: placement.y_offset().unwrap_or(0),
                        z: placement.z().unwrap_or(0),
                    },
                    generation,
                ));
            }
        }

        let mut wrote = false;

        // Images: transmit new ones and re-transmit changed ones (a changed
        // generation invalidates every placement of that image).
        for (key, _, generation) in &desired {
            let image_id = key.0;
            let current = self.transmitted.get(&image_id).copied();
            if current.map(|t| t.generation) == Some(*generation) {
                continue;
            }
            let Some(image) = graphics.image(image_id) else {
                continue;
            };
            let format = match image.format() {
                Ok(ImageFormat::Rgba) => 32,
                Ok(ImageFormat::Rgb) => 24,
                // PNG is decoded to RGBA on the way in; gray formats have no
                // kitty wire format.
                _ => continue,
            };
            let Ok(Some(data)) = image.data() else {
                continue;
            };
            let zlib = matches!(image.compression(), Ok(Compression::ZlibDeflate));
            let (Ok(width), Ok(height)) = (image.width(), image.height()) else {
                continue;
            };
            let real = real_image_id(image_id);
            if current.is_some() {
                stdout.write_all(delete_image_command(real).as_bytes())?;
                self.placed.retain(|k, _| k.0 != image_id);
            }
            stdout.write_all(&transmit_commands(real, format, width, height, zlib, data))?;
            self.transmitted.insert(
                image_id,
                Transmitted {
                    generation: *generation,
                },
            );
            wrote = true;
        }

        // Placements the VT no longer shows. Placements of an image the VT
        // dropped altogether are skipped here: the image delete below takes
        // them with it.
        let stale: Vec<PlacementKey> = self
            .placed
            .keys()
            .filter(|k| !desired.iter().any(|(dk, _, _)| dk == *k) && graphics.image(k.0).is_some())
            .copied()
            .collect();
        for key in stale {
            if let Some((real_placement, placement)) = self.placed.remove(&key) {
                stdout.write_all(
                    delete_placement_command(placement.image, real_placement).as_bytes(),
                )?;
                wrote = true;
            }
        }

        // New or moved placements.
        for (key, placement, _) in desired {
            if !self.transmitted.contains_key(&key.0) {
                continue;
            }
            let real_placement = match self.placed.get(&key) {
                Some((_, current)) if *current == placement => continue,
                Some((id, current)) => {
                    let id = *id;
                    stdout.write_all(delete_placement_command(current.image, id).as_bytes())?;
                    id
                }
                None => self.allocate_placement_id(),
            };
            queue!(stdout, cursor::MoveTo(placement.col, placement.row))?;
            stdout.write_all(place_command(real_placement, &placement).as_bytes())?;
            self.placed.insert(key, (real_placement, placement));
            wrote = true;
        }

        // Images the VT dropped entirely can go from the real terminal too.
        let dropped: Vec<u32> = self
            .transmitted
            .keys()
            .filter(|id| graphics.image(**id).is_none())
            .copied()
            .collect();
        for image_id in dropped {
            self.transmitted.remove(&image_id);
            self.placed.retain(|k, _| k.0 != image_id);
            stdout.write_all(delete_image_command(real_image_id(image_id)).as_bytes())?;
            wrote = true;
        }

        Ok(wrote)
    }

    /// Remove every placement from the real terminal (images stay cached
    /// there); the next `sync` places them afresh. Used across a resize,
    /// when the renderer repaints from scratch.
    pub fn reset_placements(&mut self, stdout: &mut impl Write) -> io::Result<()> {
        for (_, (real_placement, placement)) in self.placed.drain() {
            stdout
                .write_all(delete_placement_command(placement.image, real_placement).as_bytes())?;
        }
        Ok(())
    }

    /// Remove everything this session put on the real terminal.
    pub fn cleanup(&mut self, stdout: &mut impl Write) -> io::Result<()> {
        self.placed.clear();
        for (image_id, _) in self.transmitted.drain() {
            stdout.write_all(delete_image_command(real_image_id(image_id)).as_bytes())?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn src(x: u32, y: u32, width: u32, height: u32) -> SourceRect {
        SourceRect {
            x,
            y,
            width,
            height,
        }
    }

    #[test]
    fn real_ids_live_above_the_private_base() {
        assert_eq!(real_image_id(1), REAL_ID_BASE | 1);
        assert_eq!(real_image_id(u32::MAX), REAL_ID_BASE | REAL_ID_MASK);
        assert!(real_image_id(31) > 1_000_000);
    }

    #[test]
    fn kitty_graphics_replies_are_recognized_exactly() {
        assert!(is_kitty_graphics_reply(b"\x1b_Gi=31;OK\x1b\\"));
        assert!(is_kitty_graphics_reply(b"\x1b_Gi=1;EINVAL:bad\x1b\\"));
        assert!(!is_kitty_graphics_reply(b"\x1b_Gi=31;OK"));
        assert!(!is_kitty_graphics_reply(b"\x1b[1;1R"));
        assert!(!is_kitty_graphics_reply(b"\x1b]11;rgb:00/00/00\x1b\\"));
    }

    #[test]
    fn fully_visible_placement_is_unchanged_by_clipping() {
        let extent = Extent {
            row: 2,
            col: 3,
            rows: 4,
            cols: 8,
        };
        let (e, s) = clip(extent, src(0, 0, 80, 40), 24, 80).unwrap();
        assert_eq!(e, extent);
        assert_eq!(s, src(0, 0, 80, 40));
    }

    #[test]
    fn placement_scrolled_above_the_viewport_loses_its_top() {
        // 4 rows tall, top 2 rows above the viewport.
        let extent = Extent {
            row: -2,
            col: 0,
            rows: 4,
            cols: 8,
        };
        let (e, s) = clip(extent, src(0, 0, 80, 40), 24, 80).unwrap();
        assert_eq!(
            e,
            Extent {
                row: 0,
                col: 0,
                rows: 2,
                cols: 8
            }
        );
        assert_eq!(s, src(0, 20, 80, 20));
    }

    #[test]
    fn placement_over_the_status_row_is_cut_at_the_content_bottom() {
        // Viewport has 10 visible rows; placement covers rows 8..12.
        let extent = Extent {
            row: 8,
            col: 0,
            rows: 4,
            cols: 8,
        };
        let (e, s) = clip(extent, src(0, 0, 80, 40), 10, 80).unwrap();
        assert_eq!(e.rows, 2);
        assert_eq!(e.row, 8);
        assert_eq!(s, src(0, 0, 80, 20));
    }

    #[test]
    fn placement_entirely_off_screen_is_dropped() {
        let extent = Extent {
            row: 10,
            col: 0,
            rows: 4,
            cols: 8,
        };
        assert!(clip(extent, src(0, 0, 80, 40), 10, 80).is_none());
        let extent = Extent {
            row: -4,
            col: 0,
            rows: 4,
            cols: 8,
        };
        assert!(clip(extent, src(0, 0, 80, 40), 10, 80).is_none());
    }

    #[test]
    fn placement_past_the_right_edge_is_cut_horizontally() {
        let extent = Extent {
            row: 0,
            col: 76,
            rows: 2,
            cols: 8,
        };
        let (e, s) = clip(extent, src(0, 0, 80, 20), 24, 80).unwrap();
        assert_eq!(e.cols, 4);
        assert_eq!(s, src(0, 0, 40, 20));
    }

    #[test]
    fn transmission_is_chunked_with_continuation_flags() {
        let data = vec![0xabu8; 5000]; // 6668 base64 bytes → 2 chunks
        let out = transmit_commands(real_image_id(7), 32, 50, 25, false, &data);
        let text = String::from_utf8(out).unwrap();
        let sequences: Vec<&str> = text.split("\x1b\\").filter(|s| !s.is_empty()).collect();
        assert_eq!(sequences.len(), 2);
        assert!(
            sequences[0].starts_with(&format!(
                "\x1b_Ga=t,t=d,q=2,i={},f=32,s=50,v=25,m=1;",
                real_image_id(7)
            )),
            "{}",
            &sequences[0][..60]
        );
        assert!(sequences[1].starts_with("\x1b_Gm=0;"));
        let payload_len: usize = sequences
            .iter()
            .map(|s| s.split_once(';').unwrap().1.len())
            .sum();
        assert_eq!(payload_len, BASE64.encode(&data).len());
        assert!(sequences[0].split_once(';').unwrap().1.len() == CHUNK_BYTES);
    }

    #[test]
    fn compressed_images_keep_their_compression_flag() {
        let out = transmit_commands(real_image_id(1), 24, 2, 2, true, &[1, 2, 3]);
        assert!(String::from_utf8(out).unwrap().contains(",o=z,m=0;"));
    }

    #[test]
    fn placement_command_keeps_the_cursor_and_stays_quiet() {
        let p = Placement {
            image: real_image_id(3),
            row: 5,
            col: 2,
            cols: 10,
            rows: 4,
            src: src(1, 2, 30, 40),
            x_offset: 0,
            y_offset: 3,
            z: -1,
        };
        assert_eq!(
            place_command(9, &p),
            format!(
                "\x1b_Ga=p,q=2,C=1,i={},p=9,c=10,r=4,x=1,y=2,w=30,h=40,X=0,Y=3,z=-1\x1b\\",
                real_image_id(3)
            )
        );
        assert_eq!(
            delete_placement_command(real_image_id(3), 9),
            format!("\x1b_Ga=d,d=i,q=2,i={},p=9\x1b\\", real_image_id(3))
        );
        assert_eq!(
            delete_image_command(real_image_id(3)),
            format!("\x1b_Ga=d,d=I,q=2,i={}\x1b\\", real_image_id(3))
        );
    }

    #[test]
    fn cleanup_deletes_every_transmitted_image_once() {
        let mut sync = ImageSync::default();
        sync.transmitted.insert(4, Transmitted { generation: 1 });
        sync.transmitted.insert(9, Transmitted { generation: 2 });
        let mut out = Vec::new();
        sync.cleanup(&mut out).unwrap();
        let text = String::from_utf8(out).unwrap();
        assert_eq!(text.matches("a=d,d=I").count(), 2);
        assert!(text.contains(&format!("i={}", real_image_id(4))));
        assert!(text.contains(&format!("i={}", real_image_id(9))));
        assert!(sync.transmitted.is_empty());
    }

    #[test]
    fn placement_ids_are_never_zero() {
        let mut sync = ImageSync {
            next_placement_id: REAL_ID_MASK,
            ..Default::default()
        };
        for _ in 0..3 {
            assert!(sync.allocate_placement_id() >= 1);
        }
    }

    const ONE_PIXEL_PNG_B64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/iZk9HQAAAABJRU5ErkJggg==";

    /// Drive a real libghostty terminal: enable graphics, transmit a PNG,
    /// and mirror the resulting placement.
    #[test]
    fn vt_placement_is_mirrored_with_a_transmit_and_a_place() {
        install_png_decoder();
        let replies = std::cell::RefCell::new(Vec::new());
        let mut vt = Terminal::new(80, 24).expect("terminal");
        vt.resize(80, 24, 9, 20).expect("cell size");
        vt.on_pty_write({
            let replies = &replies;
            move |_t, data| replies.borrow_mut().extend_from_slice(data)
        })
        .expect("hook");
        enable(&mut vt).expect("enable kitty graphics");

        vt.vt_write(
            format!("\x1b_Ga=T,f=100,i=7,c=2,r=1,q=1;{ONE_PIXEL_PNG_B64}\x1b\\").as_bytes(),
        );
        let reply = String::from_utf8_lossy(&replies.borrow()).to_string();
        assert!(
            reply.is_empty() || reply.contains("OK"),
            "VT rejected the image: {reply:?}"
        );

        {
            let graphics = vt.kitty_graphics().expect("graphics handle");
            let image = graphics.image(7).expect("image 7 stored in the VT");
            assert_eq!((image.width().unwrap(), image.height().unwrap()), (1, 1));
            assert_eq!(image.format().unwrap(), ImageFormat::Rgba);
            let mut iterator = PlacementIterator::new().unwrap();
            let mut placements = iterator.update(&graphics).unwrap();
            let placement = placements.next().expect("one placement");
            let info = placement.placement_render_info(&image, &vt).unwrap();
            assert!(info.viewport_visible, "placement not visible: {info:?}");
            assert_eq!((info.grid_cols, info.grid_rows), (2, 1), "{info:?}");
        }

        let mut sync = ImageSync::default();
        let mut out = Vec::new();
        assert!(sync.sync(&mut out, &vt, 0, 24).unwrap());
        let text = String::from_utf8_lossy(&out).to_string();
        let real = real_image_id(7);
        assert!(
            text.contains(&format!("\x1b_Ga=t,t=d,q=2,i={real},f=32,s=1,v=1,m=0;")),
            "{text:?}"
        );
        assert!(
            text.contains(&format!("\x1b_Ga=p,q=2,C=1,i={real},p=1,c=2,r=1,")),
            "{text:?}"
        );

        // Steady state: nothing new to write.
        let mut out = Vec::new();
        assert!(!sync.sync(&mut out, &vt, 0, 24).unwrap());
        assert!(out.is_empty());

        // Deleting the image in the VT deletes it on the real terminal.
        vt.vt_write(b"\x1b_Ga=d,d=I,i=7,q=2\x1b\\");
        let mut out = Vec::new();
        assert!(sync.sync(&mut out, &vt, 0, 24).unwrap());
        assert_eq!(
            String::from_utf8_lossy(&out),
            format!("\x1b_Ga=d,d=I,q=2,i={real}\x1b\\")
        );
    }

    /// Same call order as `ShellSession::run`: scrollback, pty-write hook,
    /// enable, a zero-cell-size resize plus clear, then the probed cell size
    /// arrives as a grid-preserving resize.
    #[test]
    fn session_call_order_still_mirrors_placements() {
        install_png_decoder();
        let replies = std::cell::RefCell::new(Vec::new());
        let mut vt = Terminal::new(80, 24).expect("terminal");
        vt.set_scrollback_max_bytes(Some(10_000_000)).unwrap();
        vt.on_pty_write({
            let replies = &replies;
            move |_t, data| replies.borrow_mut().extend_from_slice(data)
        })
        .expect("hook");
        enable(&mut vt).expect("enable");
        vt.resize(80, 24, 0, 0).unwrap();
        vt.vt_write(b"\x1b[2J\x1b[H");
        vt.resize(80, 24, 9, 20).unwrap();

        vt.vt_write(
            format!("\x1b_Ga=T,f=100,i=7,c=2,r=1,q=0;{ONE_PIXEL_PNG_B64}\x1b\\").as_bytes(),
        );
        let reply = String::from_utf8_lossy(&replies.borrow()).to_string();
        assert!(
            reply.contains("OK"),
            "VT did not accept the image: {reply:?}"
        );

        {
            let graphics = vt.kitty_graphics().expect("graphics");
            let image = graphics.image(7).expect("image stored");
            let mut iterator = PlacementIterator::new().unwrap();
            let mut placements = iterator.update(&graphics).unwrap();
            let placement = placements.next().expect("one placement");
            let info = placement.placement_render_info(&image, &vt);
            assert!(info.is_ok(), "render info: {info:?}");
            let info = info.unwrap();
            assert!(info.viewport_visible, "{info:?}");
            assert_eq!((info.grid_cols, info.grid_rows), (2, 1), "{info:?}");
            eprintln!("render info: {info:?}");
        }

        let mut sync = ImageSync::default();
        let mut out = Vec::new();
        assert!(sync.sync(&mut out, &vt, 0, 24).unwrap(), "nothing mirrored");
        assert!(String::from_utf8_lossy(&out).contains("a=p,q=2,C=1"));
    }
}
