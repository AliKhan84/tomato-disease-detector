"""Re-exec under an interpreter that actually has TensorFlow.

Why this module exists
----------------------
On this machine `~/.bashrc` prepends pyenv's shims to PATH *after* conda's init block, so
`streamlit` -- and `python`, and even `conda run -n tf-uetm` -- resolve to pyenv's Python
3.14. That interpreter has Streamlit but no TensorFlow, and never will: TF publishes no
3.14 wheels. `conda activate tf-uetm` sets CONDA_PREFIX and changes the prompt correctly;
only PATH is wrong, which is why the failure reads as a broken environment when the
environment is fine.

The symptom is a `ModuleNotFoundError: No module named 'tensorflow'` traceback from deep
inside the app, pointing at a line that is not the problem.

Rather than require a special launch command, the entrypoint calls `ensure_tensorflow()`
before importing anything heavy. If the current interpreter cannot import TensorFlow, this
finds one that can and re-executes the *same* command under it. `os.execv` replaces the
process image, so the port is rebound by the new process and the terminal keeps streaming
output as normal -- from the caller's point of view the command just works.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

# Set on the child so a re-exec that still cannot import TensorFlow raises a clear error
# instead of spawning itself forever.
SENTINEL = "TOMATO_INTERPRETER_GUARD"


def _can_import(python: Path, *modules: str) -> bool:
    """True if `python` can find every module, without importing them.

    `find_spec` only locates the package, so this stays fast -- actually importing
    TensorFlow costs several seconds per candidate.
    """
    probe = (
        "import importlib.util as u, sys; "
        f"sys.exit(0 if all(u.find_spec(m) for m in {list(modules)!r}) else 1)"
    )
    try:
        return subprocess.run(
            [str(python), "-c", probe],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=60,
        ).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _candidates() -> list[Path]:
    """Interpreters that might have TensorFlow, best guess first, de-duplicated."""
    home = Path.home()
    found: list[Path] = []

    explicit = os.environ.get("TOMATO_PYTHON")
    if explicit:
        found.append(Path(explicit))

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        found.append(Path(conda_prefix) / "bin" / "python")

    # The env this project was built against.
    found.append(home / "anaconda3" / "envs" / "tf-uetm" / "bin" / "python")

    # Any other conda env, then any other pyenv version.
    for root in (home / "anaconda3" / "envs", home / "miniconda3" / "envs"):
        if root.is_dir():
            found.extend(sorted(root.glob("*/bin/python")))
    versions = home / ".pyenv" / "versions"
    if versions.is_dir():
        found.extend(sorted(versions.glob("*/bin/python")))

    current = Path(sys.executable).resolve()
    unique: list[Path] = []
    for path in found:
        try:
            resolved = path.resolve()
        except OSError:
            continue
        # Skip the interpreter we already know is missing TensorFlow.
        if resolved == current or resolved in {p.resolve() for p in unique}:
            continue
        if os.access(resolved, os.X_OK) and resolved.is_file():
            unique.append(resolved)
    return unique


def _rebuild_command(python: Path) -> list[str]:
    """The current command line, re-pointed at `python`.

    Reads /proc/self/cmdline rather than sys.argv because Streamlit rewrites sys.argv to
    the *script's* arguments -- the server flags (--server.port and friends) are only
    visible in the real process command line.
    """
    argv: list[str] = []
    try:
        raw = Path("/proc/self/cmdline").read_bytes()
        argv = [part for part in raw.decode().split("\0") if part]
    except OSError:
        pass

    if "run" in argv:
        # `streamlit run <script> [flags]` -> `<python> -m streamlit run <script> [flags]`
        after_run = argv[argv.index("run") + 1:]
        return [str(python), "-m", "streamlit", "run", *after_run]

    # Not a `streamlit run` invocation (plain script, pytest, AppTest): keep argv as-is.
    return [str(python), *(argv[1:] if argv else sys.argv)]


def _close_inherited_sockets() -> None:
    """Close every open socket so the re-executed server can rebind the port.

    `os.execve` replaces the process image but keeps file descriptors that are marked
    inheritable, and uvicorn's listening socket is one of them. Without this the new
    interpreter starts, finds its own inherited socket still bound, and exits with
    "Port 8501 is not available". Closing them here is safe because the process is about
    to be replaced -- nothing else will ever read from these descriptors again.
    """
    try:
        fds = os.listdir("/proc/self/fd")
    except OSError:
        return
    for entry in fds:
        try:
            fd = int(entry)
        except ValueError:
            continue
        if fd < 3:  # keep stdin/stdout/stderr -- the new process logs to the terminal
            continue
        try:
            if os.readlink(f"/proc/self/fd/{fd}").startswith("socket:"):
                os.close(fd)
        except OSError:
            continue


def _managed_host() -> str | None:
    """Name the managed platform we appear to be running on, or None if local.

    Scanning the filesystem for a second interpreter is a local-machine fix. On a hosted
    runner there is exactly one environment, so the scan just burns a subprocess per
    candidate and then reports "searched: (none found)" -- which hides the actual cause.
    Streamlit Community Cloud clones the repo under /mount/src and builds its venv at
    /home/adminuser/venv; either is a reliable tell.
    """
    if Path("/mount/src").is_dir() or Path("/home/adminuser/venv").is_dir():
        return "Streamlit Community Cloud"
    return None


def ensure_tensorflow() -> None:
    """Re-exec under a TensorFlow-capable interpreter, or return if this one already is."""
    if importlib.util.find_spec("tensorflow") is not None:
        return

    version = f"{sys.version_info.major}.{sys.version_info.minor}"

    host = _managed_host()
    if host:
        # No second interpreter to find here, and no way to install one. The only real
        # cause is a Python the TF wheels do not cover, so say that instead of scanning.
        raise RuntimeError(
            f"TensorFlow is not installed on this {host} container, which is running "
            f"Python {version}.\n"
            "TensorFlow publishes wheels for Python 3.10-3.13 only. If the version above "
            "is outside that range, `pip install` cannot succeed here no matter what "
            "requirements.txt says.\n"
            "Fix: redeploy the app on Python 3.11, 3.12 or 3.13. The Python version is "
            "chosen in Advanced settings at deploy time and cannot be changed afterwards, "
            "so the app has to be deleted and deployed again. See README -> Deploying."
        )

    if os.environ.get(SENTINEL):
        raise RuntimeError(
            f"Re-executed under {sys.executable} but TensorFlow still will not import.\n"
            "Install the requirements into that interpreter:\n"
            f"    {sys.executable} -m pip install -r requirements.txt\n"
            "or point the app at a different one:\n"
            "    TOMATO_PYTHON=/path/to/python streamlit run src/streamlit_app.py"
        )

    needed = ("tensorflow", "streamlit")
    for python in _candidates():
        if not _can_import(python, *needed):
            continue
        print(
            f"\n[interpreter guard] {sys.executable}\n"
            "[interpreter guard]   has no TensorFlow, and never will -- TF ships no "
            f"wheels for Python {version}.\n"
            f"[interpreter guard] Restarting the server under:\n"
            f"[interpreter guard]   {python}\n"
            "[interpreter guard] Reload the page in a moment.\n",
            file=sys.stderr,
            flush=True,
        )
        env = dict(os.environ, **{SENTINEL: "1"})
        command = _rebuild_command(python)
        try:
            _close_inherited_sockets()
            os.execve(command[0], command, env)
        except OSError as exc:  # pragma: no cover -- execve rarely returns
            print(f"[interpreter guard] re-exec failed: {exc}", file=sys.stderr)
            break

    searched = "\n".join(f"    {p}" for p in _candidates()) or "    (none found)"
    raise RuntimeError(
        f"TensorFlow cannot be imported under {sys.executable}, and no interpreter with "
        "both TensorFlow and Streamlit was found.\n"
        f"Searched:\n{searched}\n"
        "Set TOMATO_PYTHON to the right interpreter, or install the requirements:\n"
        "    <python> -m pip install -r requirements.txt"
    )
