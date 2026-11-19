"""desktop/: the Linux desktop shortcut (menu entry + desktop icon + launcher).

The installer runs against a throwaway HOME, so nothing here touches the real
desktop; the launcher is only exercised on paths that need no Docker.
"""
import os
import shutil
import socket
import stat
import subprocess

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INSTALLER = os.path.join(ROOT, "desktop", "install-desktop-shortcut.sh")
LAUNCHER = os.path.join(ROOT, "desktop", "openshorts-launch.sh")

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def _env(home):
    env = {k: v for k, v in os.environ.items() if not k.startswith("XDG_")}
    env.update(HOME=str(home), XDG_DATA_HOME=str(home / ".local" / "share"),
               XDG_CONFIG_HOME=str(home / ".config"))
    env.pop("OPENSHORTS_DIR", None)
    return env


def _install(home, *args):
    return subprocess.run(["bash", INSTALLER, *args], env=_env(home),
                          capture_output=True, text=True, timeout=30)


def _entry(path):
    """The [Desktop Entry] group as a dict (later groups ignored)."""
    out, group = {}, None
    for line in path.read_text().splitlines():
        if line.startswith("["):
            group = line
        elif group == "[Desktop Entry]" and "=" in line:
            k, v = line.split("=", 1)
            out[k] = v
    return out


@pytest.mark.parametrize("script", [INSTALLER, LAUNCHER])
def test_scripts_parse_and_are_executable(script):
    assert subprocess.run(["bash", "-n", script]).returncode == 0
    assert os.stat(script).st_mode & stat.S_IXUSR


def test_install_adds_menu_entry_desktop_icon_and_launcher(tmp_path):
    r = _install(tmp_path)
    assert r.returncode == 0, r.stderr

    menu = tmp_path / ".local/share/applications/openshorts.desktop"
    icon_on_desktop = tmp_path / "Desktop/openshorts.desktop"
    app_dir = tmp_path / ".local/share/openshorts"
    launcher = app_dir / "openshorts-launch.sh"
    assert menu.is_file() and icon_on_desktop.is_file()
    assert icon_on_desktop.read_text() == menu.read_text()
    assert os.stat(icon_on_desktop).st_mode & stat.S_IXUSR  # KDE/GNOME want it executable

    entry = _entry(menu)
    assert entry["Type"] == "Application" and entry["Name"] == "OpenShorts"
    assert entry["Exec"] == f'"{launcher}" start'
    assert entry["Terminal"] == "true"  # first build output + sudo prompt are visible
    assert entry["Icon"] == str(app_dir / "openshorts.png")
    assert (app_dir / "openshorts.png").is_file()
    assert "[Desktop Action stop]" in menu.read_text()

    # The installed launcher remembers this checkout, so switching branches
    # (or a checkout without desktop/) does not break the shortcut.
    assert f'OPENSHORTS_DIR="${{OPENSHORTS_DIR:-{ROOT}}}"' in launcher.read_text()
    assert subprocess.run(["bash", "-n", str(launcher)]).returncode == 0

    if shutil.which("desktop-file-validate"):
        v = subprocess.run(["desktop-file-validate", str(menu)], capture_output=True, text=True)
        assert v.returncode == 0, v.stdout + v.stderr


def test_no_desktop_flag_only_adds_the_menu_entry(tmp_path):
    assert _install(tmp_path, "--no-desktop").returncode == 0
    assert (tmp_path / ".local/share/applications/openshorts.desktop").is_file()
    assert not (tmp_path / "Desktop/openshorts.desktop").exists()


def test_uninstall_removes_everything_it_added(tmp_path):
    assert _install(tmp_path).returncode == 0
    assert _install(tmp_path, "--uninstall").returncode == 0
    assert not (tmp_path / ".local/share/applications/openshorts.desktop").exists()
    assert not (tmp_path / "Desktop/openshorts.desktop").exists()
    assert not (tmp_path / ".local/share/openshorts").exists()


def test_installer_refuses_a_directory_that_is_not_a_checkout(tmp_path):
    env = _env(tmp_path)
    env["OPENSHORTS_DIR"] = str(tmp_path)
    r = subprocess.run(["bash", INSTALLER], env=env, capture_output=True, text=True, timeout=30)
    assert r.returncode == 1 and "docker-compose.yml" in r.stdout + r.stderr


def test_openshorts_dir_can_point_at_another_checkout(tmp_path):
    # e.g. installing from a fresh clone for a checkout on a branch without desktop/
    other = tmp_path / "other-checkout"
    other.mkdir()
    (other / "docker-compose.yml").write_text("services: {}\n")
    home = tmp_path / "home"
    env = _env(home)
    env["OPENSHORTS_DIR"] = str(other)
    r = subprocess.run(["bash", INSTALLER], env=env, capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    launcher = (home / ".local/share/openshorts/openshorts-launch.sh").read_text()
    assert f'OPENSHORTS_DIR="${{OPENSHORTS_DIR:-{other}}}"' in launcher


def _closed_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_launcher_status_reports_not_running(tmp_path):
    if shutil.which("curl") is None:
        pytest.skip("needs curl")
    env = _env(tmp_path)
    env["OPENSHORTS_URL"] = f"http://127.0.0.1:{_closed_port()}"
    r = subprocess.run(["bash", LAUNCHER, "status"], env=env,
                       capture_output=True, text=True, timeout=30)
    assert r.returncode == 1 and "Not running" in r.stdout


def test_launcher_rejects_an_unknown_command(tmp_path):
    r = subprocess.run(["bash", LAUNCHER, "explode"], env=_env(tmp_path),
                       stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=30)
    assert r.returncode == 1 and "unknown command" in r.stderr
