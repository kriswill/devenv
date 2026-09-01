//! Automatic integration with the shared `devenv-proxy` daemon.

use crate::tasks;
use devenv_proxy::{ControlRequest, ControlResponse, Route};
use miette::{IntoDiagnostic, Result, WrapErr, bail, miette};
use std::{
    collections::{BTreeMap, HashSet},
    fs,
    io::{Read, Write},
    net::{SocketAddr, TcpStream},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::{Duration, Instant},
};

const START_TIMEOUT: Duration = Duration::from_secs(5);

pub(crate) fn project_routes(
    project_name: &str,
    owner: &str,
    task_configs: &[tasks::TaskConfig],
) -> Result<Vec<Route>> {
    let project = hostname_label(project_name)?;
    let mut routes = Vec::new();
    let mut hostnames = HashSet::new();

    for task in task_configs {
        let Some(process_name) = task.name.strip_prefix(devenv_tasks::PROCESS_TASK_PREFIX) else {
            continue;
        };
        let Some(process) = task.process.as_ref() else {
            continue;
        };
        if process.ports.is_empty() {
            continue;
        }

        let process_label = hostname_label(process_name)?;
        let ports: BTreeMap<&str, u16> = process
            .ports
            .iter()
            .map(|(name, port)| (name.as_str(), *port))
            .collect();
        let default_port = ports
            .get("http")
            .copied()
            .or_else(|| (ports.len() == 1).then(|| *ports.values().next().unwrap()));

        if let Some(port) = default_port {
            push_route(
                &mut routes,
                &mut hostnames,
                format!("{process_label}.{project}.localhost"),
                port,
                owner,
            )?;
        }

        // Multiple named ports remain addressable without requiring another
        // option. The conventional `http` port also receives the short URL.
        if ports.len() > 1 {
            for (port_name, port) in ports {
                let port_label = hostname_label(port_name)?;
                push_route(
                    &mut routes,
                    &mut hostnames,
                    format!("{port_label}.{process_label}.{project}.localhost"),
                    port,
                    owner,
                )?;
            }
        }
    }

    Ok(routes)
}

fn push_route(
    routes: &mut Vec<Route>,
    hostnames: &mut HashSet<String>,
    hostname: String,
    port: u16,
    owner: &str,
) -> Result<()> {
    if !hostnames.insert(hostname.clone()) {
        bail!("multiple process ports resolve to proxy hostname {hostname}");
    }
    routes.push(Route {
        hostname,
        upstream: SocketAddr::from(([127, 0, 0, 1], port)),
        owner: owner.to_owned(),
    });
    Ok(())
}

fn hostname_label(value: &str) -> Result<String> {
    let mut label = String::with_capacity(value.len());
    let mut separator = false;
    for character in value.chars() {
        if character.is_ascii_alphanumeric() {
            if separator && !label.is_empty() {
                label.push('-');
            }
            label.push(character.to_ascii_lowercase());
            separator = false;
        } else {
            separator = true;
        }
    }
    if label.is_empty() {
        bail!("{value:?} cannot be represented as a localhost hostname label");
    }
    if label.len() > 63 {
        bail!("{value:?} is too long for a localhost hostname label");
    }
    Ok(label)
}

pub(crate) async fn reconcile(owner: &str, routes: Vec<Route>) -> Result<()> {
    if routes.is_empty() {
        // Do not start a machine-wide listener for a project with no declared
        // ports, but do remove routes left by an earlier configuration.
        let _ = replace_owner(owner, routes);
        return Ok(());
    }

    ensure_running().await?;
    replace_owner(owner, routes.clone())?;
    for route in routes {
        tracing::info!("http://{} -> http://{}", route.hostname, route.upstream);
    }
    Ok(())
}

pub(crate) fn clear(owner: &str) {
    let _ = replace_owner(owner, Vec::new());
}

fn replace_owner(owner: &str, routes: Vec<Route>) -> Result<()> {
    let socket = devenv_proxy::default_control_socket();
    devenv_proxy::request(
        &socket,
        &ControlRequest::ReplaceOwner {
            owner: owner.to_owned(),
            routes,
        },
    )
    .and_then(ControlResponse::into_result)
    .map(|_| ())
    .map_err(|error| miette!("{error:#}"))
    .wrap_err_with(|| format!("failed to update localhost proxy via {}", socket.display()))
}

async fn ensure_running() -> Result<()> {
    let socket = devenv_proxy::default_control_socket();
    if proxy_ready(&socket) {
        return Ok(());
    }

    let executable = proxy_executable()?;
    let log_path = proxy_log_path(&socket);
    if let Some(parent) = log_path.parent() {
        fs::create_dir_all(parent)
            .into_diagnostic()
            .wrap_err_with(|| format!("failed to create {}", parent.display()))?;
    }
    let log = fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&log_path)
        .into_diagnostic()
        .wrap_err_with(|| format!("failed to open {}", log_path.display()))?;
    let stderr = log.try_clone().into_diagnostic()?;

    let mut command = Command::new(&executable);
    command
        .arg("--control-socket")
        .arg(&socket)
        .stdin(Stdio::null())
        .stdout(Stdio::from(log))
        .stderr(Stdio::from(stderr));
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }
    #[cfg(target_os = "linux")]
    if proxy_listen_address().is_some_and(|address| address.port() < 1024) {
        devenv_processes::configure_linux_capabilities(
            &mut command,
            &["net_bind_service".to_owned()],
        )
        .into_diagnostic()
        .wrap_err("failed to configure the proxy's Linux capabilities")?;
    }

    let mut child = command.spawn().into_diagnostic().wrap_err_with(|| {
        format!(
            "failed to start internal proxy executable {}",
            executable.display()
        )
    })?;
    let started = Instant::now();
    while started.elapsed() < START_TIMEOUT {
        if proxy_ready(&socket) {
            return Ok(());
        }
        if let Some(status) = child.try_wait().into_diagnostic()? {
            let detail = fs::read_to_string(&log_path).unwrap_or_default();
            bail!(
                "devenv-proxy exited with {status}; it must be allowed to bind 127.0.0.1:80\n{}",
                detail.trim()
            );
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    bail!(
        "devenv-proxy did not become ready within {}s; see {}",
        START_TIMEOUT.as_secs(),
        log_path.display()
    )
}

fn proxy_ready(socket: &Path) -> bool {
    let control_ready = devenv_proxy::request(socket, &ControlRequest::List)
        .and_then(ControlResponse::into_result)
        .is_ok();
    control_ready && proxy_listen_address().is_some_and(proxy_data_plane_ready)
}

fn proxy_data_plane_ready(address: SocketAddr) -> bool {
    let Ok(mut stream) = TcpStream::connect_timeout(&address, Duration::from_millis(100)) else {
        return false;
    };
    if stream
        .set_read_timeout(Some(Duration::from_millis(200)))
        .is_err()
        || stream
            .set_write_timeout(Some(Duration::from_millis(200)))
            .is_err()
    {
        return false;
    }
    let request = format!(
        "GET / HTTP/1.1\r\nHost: {}\r\nConnection: close\r\n\r\n",
        devenv_proxy::HEALTH_HOSTNAME
    );
    if stream.write_all(request.as_bytes()).is_err() {
        return false;
    }
    let mut response = [0_u8; 32];
    stream
        .read(&mut response)
        .is_ok_and(|length| response[..length].starts_with(b"HTTP/1.1 204"))
}

fn proxy_listen_address() -> Option<SocketAddr> {
    std::env::var("DEVENV_PROXY_LISTEN")
        .unwrap_or_else(|_| "127.0.0.1:80".to_owned())
        .parse()
        .ok()
}

fn proxy_executable() -> Result<PathBuf> {
    if let Some(path) = std::env::var_os("DEVENV_PROXY_BINARY") {
        return Ok(PathBuf::from(path));
    }
    bundled_proxy_executable()
}

fn bundled_proxy_executable() -> Result<PathBuf> {
    let current = std::env::current_exe()
        .into_diagnostic()
        .wrap_err("failed to locate the devenv executable")?;
    if let Some(sibling) = current.parent().map(|parent| parent.join("devenv-proxy"))
        && sibling.is_file()
    {
        return Ok(sibling);
    }
    which::which("devenv-proxy")
        .into_diagnostic()
        .wrap_err("devenv-proxy is missing from the devenv installation")
}

fn proxy_log_path(socket: &Path) -> PathBuf {
    socket
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("proxy.log")
}

#[cfg(test)]
mod tests {
    use super::*;
    use devenv_processes::ProcessConfig;

    fn process_task(name: &str, ports: &[(&str, u16)]) -> tasks::TaskConfig {
        tasks::TaskConfig {
            name: format!("{}{}", devenv_tasks::PROCESS_TASK_PREFIX, name),
            process: Some(ProcessConfig {
                ports: ports
                    .iter()
                    .map(|(name, port)| ((*name).to_owned(), *port))
                    .collect(),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    #[test]
    fn one_port_uses_process_and_project_names() {
        let routes = project_routes(
            "my_project",
            "/work/my-project",
            &[process_task("web_app", &[("server", 8080)])],
        )
        .unwrap();
        assert_eq!(routes.len(), 1);
        assert_eq!(routes[0].hostname, "web-app.my-project.localhost");
        assert_eq!(routes[0].upstream.port(), 8080);
    }

    #[test]
    fn multiple_ports_use_http_as_default_and_expose_named_urls() {
        let routes = project_routes(
            "demo",
            "/work/demo",
            &[process_task("web", &[("http", 8080), ("admin", 9000)])],
        )
        .unwrap();
        let hostnames: BTreeMap<_, _> = routes
            .into_iter()
            .map(|route| (route.hostname, route.upstream.port()))
            .collect();
        assert_eq!(hostnames.get("web.demo.localhost"), Some(&8080));
        assert_eq!(hostnames.get("http.web.demo.localhost"), Some(&8080));
        assert_eq!(hostnames.get("admin.web.demo.localhost"), Some(&9000));
    }

    #[test]
    fn multiple_ports_without_http_have_only_named_urls() {
        let routes = project_routes(
            "demo",
            "/work/demo",
            &[process_task("web", &[("public", 8080), ("admin", 9000)])],
        )
        .unwrap();
        assert_eq!(routes.len(), 2);
        assert!(
            routes
                .iter()
                .any(|route| route.hostname == "public.web.demo.localhost")
        );
    }
}
