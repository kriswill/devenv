//! Learn the real terminal's cell size in pixels.
//!
//! The child's PTY is created without pixel dimensions and the virtual
//! terminal without a cell size, so anything that needs pixels — `kitten
//! icat`'s `TIOCGWINSZ`, the kitty graphics grid math in libghostty-vt, a
//! `CSI 14 t` query — sees zeros. The session asks the real terminal once at
//! startup and again after every resize (`CSI 16 t`, XTWINOPS cell size) and
//! intercepts the `CSI 6 ; height ; width t` answer on stdin before it can
//! reach the child as keystrokes.

use std::io::{self, Write};
use std::time::{Duration, Instant};

/// How long an unanswered probe stays armed. A terminal that does not
/// implement XTWINOPS never replies; after this the interception stops so a
/// later reply to the child's own forwarded `CSI 16 t` is not swallowed.
pub const PROBE_TIMEOUT: Duration = Duration::from_secs(2);

/// The physical terminal's cell size in pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CellSize {
    pub width: u16,
    pub height: u16,
}

/// Outstanding `CSI 16 t` probes and the stdin interception for their reply.
#[derive(Debug, Default)]
pub struct CellSizeProbe {
    pending: usize,
    since: Option<Instant>,
    /// Tail of the previous stdin chunk that could be the start of a reply
    /// split across reads.
    carry: Vec<u8>,
}

impl CellSizeProbe {
    /// Ask the real terminal for its cell size. The answer arrives on stdin
    /// and is taken by [`Self::intercept`].
    pub fn query(&mut self, stdout: &mut impl Write) -> io::Result<()> {
        stdout.write_all(b"\x1b[16t")?;
        if self.pending == 0 {
            self.since = Some(Instant::now());
        }
        self.pending += 1;
        Ok(())
    }

    fn expire_if_stale(&mut self) {
        if self.pending > 0
            && self
                .since
                .is_some_and(|since| since.elapsed() > PROBE_TIMEOUT)
        {
            self.pending = 0;
            self.since = None;
        }
    }

    /// Whether a probe is outstanding (and not yet timed out).
    pub fn is_pending(&mut self) -> bool {
        self.expire_if_stale();
        self.pending > 0
    }

    /// Remove one cell-size reply from a stdin chunk. Everything else in the
    /// chunk (and any carried-over tail) is appended to `out` in order. When a
    /// reply is found it is returned and the probe count drops by one.
    ///
    /// Only used while a probe is pending; otherwise stdin is untouched.
    pub fn intercept(&mut self, data: &[u8], out: &mut Vec<u8>) -> Option<CellSize> {
        out.clear();
        if !self.is_pending() {
            out.extend_from_slice(&self.carry);
            self.carry.clear();
            out.extend_from_slice(data);
            return None;
        }
        let mut buf = std::mem::take(&mut self.carry);
        buf.extend_from_slice(data);

        if let Some((start, end, size)) = find_reply(&buf) {
            out.extend_from_slice(&buf[..start]);
            out.extend_from_slice(&buf[end..]);
            self.pending -= 1;
            if self.pending == 0 {
                self.since = None;
            }
            return Some(size);
        }

        // Hold back a tail that can only be the beginning of a reply
        // (`ESC [ 6` or longer): terminal replies rarely straddle a read,
        // but a lone ESC or `ESC [` is common keyboard input and passes.
        let split = buf.len() - partial_reply_suffix(&buf);
        out.extend_from_slice(&buf[..split]);
        self.carry.extend_from_slice(&buf[split..]);
        None
    }
}

/// Locate `ESC [ 6 ; <height> ; <width> t`; returns its byte range and the
/// parsed size.
fn find_reply(buf: &[u8]) -> Option<(usize, usize, CellSize)> {
    let mut i = 0;
    while let Some(offset) = buf[i..].windows(4).position(|w| w == b"\x1b[6;") {
        let start = i + offset;
        let body_start = start + 4;
        if let Some(rel_end) = buf[body_start..].iter().position(|&b| b == b't') {
            let end = body_start + rel_end + 1;
            let body = &buf[body_start..body_start + rel_end];
            if let Some(size) = parse_body(body) {
                return Some((start, end, size));
            }
        }
        i = start + 1;
    }
    None
}

fn parse_body(body: &[u8]) -> Option<CellSize> {
    let separator = body.iter().position(|&b| b == b';')?;
    let (height, width) = (&body[..separator], &body[separator + 1..]);
    let parse = |digits: &[u8]| -> Option<u16> {
        (!digits.is_empty() && digits.iter().all(u8::is_ascii_digit))
            .then(|| std::str::from_utf8(digits).ok()?.parse().ok())
            .flatten()
    };
    let size = CellSize {
        height: parse(height)?,
        width: parse(width)?,
    };
    (size.width > 0 && size.height > 0).then_some(size)
}

/// Length of a suffix of `buf` that is a proper prefix of a reply, at least
/// `ESC [ 6` long; 0 when the chunk ends cleanly.
fn partial_reply_suffix(buf: &[u8]) -> usize {
    let Some(esc) = buf.iter().rposition(|&b| b == 0x1b) else {
        return 0;
    };
    let tail = &buf[esc..];
    if tail.len() < 3 || !tail.starts_with(b"\x1b[6") {
        return 0;
    }
    let rest = &tail[3..];
    let mut separators = 0;
    for &b in rest {
        match b {
            b';' => separators += 1,
            b if b.is_ascii_digit() => {}
            _ => return 0,
        }
    }
    if separators > 2 {
        return 0;
    }
    tail.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn armed() -> CellSizeProbe {
        let mut probe = CellSizeProbe::default();
        probe.query(&mut Vec::new()).unwrap();
        probe
    }

    #[test]
    fn query_writes_xtwinops_cell_size_request() {
        let mut out = Vec::new();
        let mut probe = CellSizeProbe::default();
        probe.query(&mut out).unwrap();
        assert_eq!(out, b"\x1b[16t");
        assert!(probe.is_pending());
    }

    #[test]
    fn reply_is_removed_from_stdin_and_parsed() {
        let mut probe = armed();
        let mut out = Vec::new();
        let size = probe.intercept(b"ab\x1b[6;20;9tcd", &mut out);
        assert_eq!(
            size,
            Some(CellSize {
                width: 9,
                height: 20
            })
        );
        assert_eq!(out, b"abcd");
        assert!(!probe.is_pending());
    }

    #[test]
    fn stdin_passes_untouched_when_nothing_is_pending() {
        let mut probe = CellSizeProbe::default();
        let mut out = Vec::new();
        assert_eq!(probe.intercept(b"\x1b[6;20;9t", &mut out), None);
        assert_eq!(out, b"\x1b[6;20;9t");
    }

    #[test]
    fn only_one_reply_is_taken_per_probe() {
        let mut probe = armed();
        let mut out = Vec::new();
        // The child's own forwarded CSI 16 t may be answered in the same
        // read; that copy must reach it.
        let size = probe.intercept(b"\x1b[6;20;9t\x1b[6;20;9t", &mut out);
        assert!(size.is_some());
        assert_eq!(out, b"\x1b[6;20;9t");
    }

    #[test]
    fn split_reply_is_reassembled_across_reads() {
        let mut probe = armed();
        let mut out = Vec::new();
        assert_eq!(probe.intercept(b"x\x1b[6;2", &mut out), None);
        assert_eq!(out, b"x");
        let size = probe.intercept(b"0;9ty", &mut out);
        assert_eq!(
            size,
            Some(CellSize {
                width: 9,
                height: 20
            })
        );
        assert_eq!(out, b"y");
    }

    #[test]
    fn bare_escape_and_csi_are_not_held_back() {
        let mut probe = armed();
        let mut out = Vec::new();
        assert_eq!(probe.intercept(b"\x1b", &mut out), None);
        assert_eq!(out, b"\x1b");
        assert_eq!(probe.intercept(b"\x1b[", &mut out), None);
        assert_eq!(out, b"\x1b[");
        assert_eq!(probe.intercept(b"\x1b[6;1;1x", &mut out), None);
        assert_eq!(out, b"\x1b[6;1;1x");
    }

    #[test]
    fn other_xtwinops_reports_pass_through() {
        let mut probe = armed();
        let mut out = Vec::new();
        assert_eq!(probe.intercept(b"\x1b[8;24;80t", &mut out), None);
        assert_eq!(out, b"\x1b[8;24;80t");
        assert!(probe.is_pending());
    }

    #[test]
    fn zero_sizes_are_rejected() {
        assert_eq!(parse_body(b"0;9"), None);
        assert_eq!(parse_body(b"20;0"), None);
        assert_eq!(parse_body(b"20"), None);
        assert_eq!(parse_body(b"a;b"), None);
    }

    #[test]
    fn stale_probe_stops_intercepting() {
        let mut probe = armed();
        probe.since = Some(Instant::now() - PROBE_TIMEOUT - Duration::from_millis(1));
        let mut out = Vec::new();
        assert_eq!(probe.intercept(b"\x1b[6;20;9t", &mut out), None);
        assert_eq!(out, b"\x1b[6;20;9t");
        assert!(!probe.is_pending());
    }
}
