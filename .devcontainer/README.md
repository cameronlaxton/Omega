# Omega Native Linux Devcontainer

This container is the durable runtime path for Omega Cowork work. It avoids the
Windows-to-Linux FUSE mount by running the active workspace from the Docker named
volume `omega-linux-workspace`.

## Startup

Open this repository with Dev Containers, or rebuild the container after pulling
new source. The container seeds `/workspaces/Omega` from `/opt/omega-source` when
the volume is empty, installs `.[dev,mcp]`, runs direct preflight, runs the known
bug sentinel, and verifies SQLite WAL on a native Linux filesystem.

## Sync Back

Do normal source-control sync with git from inside the container:

```bash
git status --short
git push
```

Do not treat the Windows `C:\repos\Omega` mount as the live SQLite or `.git`
workspace while the container is active. If artifacts must be mirrored back to a
Windows-visible checkout, copy reports and trace exports after the session
closes; never write live SQLite sidecars through the FUSE mount.
