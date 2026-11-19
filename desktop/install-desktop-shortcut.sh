#!/usr/bin/env bash
# Add an "OpenShorts" launcher to the application menu and the desktop (Linux,
# freedesktop desktops: KDE, GNOME, XFCE, ...).
#
#   desktop/install-desktop-shortcut.sh             install
#   desktop/install-desktop-shortcut.sh --uninstall remove it again
#
# Options:
#   --no-desktop   only add the menu entry, no icon on the desktop
#
# The launcher and icon are copied to ~/.local/share/openshorts with this
# checkout's path baked in, so the shortcut keeps working whichever branch is
# checked out. Run it again after moving the checkout.
set -euo pipefail

# SRC_DIR: where the launcher and icon are copied from (this script's checkout).
# REPO_DIR: the checkout the shortcut starts; OPENSHORTS_DIR points it elsewhere.
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="${OPENSHORTS_DIR:-$SRC_DIR}"
REPO_DIR="$(cd "$REPO_DIR" 2>/dev/null && pwd || printf '%s' "$REPO_DIR")"
DATA_HOME="${XDG_DATA_HOME:-$HOME/.local/share}"
APP_DIR="$DATA_HOME/openshorts"
MENU_DIR="$DATA_HOME/applications"
MENU_FILE="$MENU_DIR/openshorts.desktop"

desktop_dir() {
    local d=""
    if command -v xdg-user-dir >/dev/null 2>&1; then
        d="$(xdg-user-dir DESKTOP 2>/dev/null || true)"
    fi
    # xdg-user-dir answers $HOME when no desktop dir is configured.
    if [ -z "$d" ] || [ "$d" = "$HOME" ]; then
        d="$HOME/Desktop"
    fi
    printf '%s\n' "$d"
}

say() { printf '[openshorts] %s\n' "$*"; }

with_desktop=1
action=install
for arg in "$@"; do
    case "$arg" in
        --uninstall) action=uninstall ;;
        --no-desktop) with_desktop=0 ;;
        -h|--help) sed -n '2,14p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) say "unknown option: $arg" >&2; exit 2 ;;
    esac
done

DESKTOP_FILE="$(desktop_dir)/openshorts.desktop"

if [ "$action" = uninstall ]; then
    rm -f "$MENU_FILE" "$DESKTOP_FILE"
    rm -rf "$APP_DIR"
    say "Removed the OpenShorts shortcut."
    exit 0
fi

[ -f "$REPO_DIR/docker-compose.yml" ] || {
    say "no docker-compose.yml in $REPO_DIR: run this from an openshorts checkout or set OPENSHORTS_DIR." >&2
    exit 1
}

mkdir -p "$APP_DIR" "$MENU_DIR"

# The launcher, with this checkout as its default OPENSHORTS_DIR.
launcher="$APP_DIR/openshorts-launch.sh"
{
    echo '#!/usr/bin/env bash'
    printf 'export OPENSHORTS_DIR="${OPENSHORTS_DIR:-%s}"\n' "$REPO_DIR"
    tail -n +2 "$SRC_DIR/desktop/openshorts-launch.sh"
} > "$launcher"
chmod 755 "$launcher"

icon="$APP_DIR/openshorts.png"
cp "$SRC_DIR/dashboard/public/logo-openshorts.png" "$icon"

# Exec values are quoted per the Desktop Entry spec so a path with spaces works.
cat > "$MENU_FILE" <<EOF
[Desktop Entry]
Type=Application
Version=1.0
Name=OpenShorts
GenericName=AI Clip Generator
Comment=Start the self-hosted OpenShorts stack and open the dashboard
Exec="$launcher" start
Icon=$icon
Terminal=true
Categories=AudioVideo;Video;AudioVideoEditing;
Keywords=video;shorts;clips;tiktok;reels;
StartupNotify=false
Actions=stop;

[Desktop Action stop]
Name=Stop OpenShorts
Exec="$launcher" stop
EOF
chmod 755 "$MENU_FILE"
say "Added OpenShorts to the application menu ($MENU_FILE)."

if [ "$with_desktop" = 1 ]; then
    mkdir -p "$(dirname "$DESKTOP_FILE")"
    cp "$MENU_FILE" "$DESKTOP_FILE"
    chmod 755 "$DESKTOP_FILE"
    # GNOME only runs a desktop launcher once it is marked trusted.
    if command -v gio >/dev/null 2>&1; then
        gio set "$DESKTOP_FILE" metadata::trusted true >/dev/null 2>&1 || true
    fi
    say "Added the OpenShorts icon to $(dirname "$DESKTOP_FILE")."
fi

command -v update-desktop-database >/dev/null 2>&1 \
    && update-desktop-database "$MENU_DIR" >/dev/null 2>&1 || true

say "Done. Launch it from the menu or the desktop; right-click it for 'Stop OpenShorts'."
