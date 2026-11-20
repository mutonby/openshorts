#!/usr/bin/env bash
# Start the self-hosted OpenShorts stack and open the dashboard.
#
#   openshorts-launch.sh [start|stop|status]     (default: start)
#
# The desktop shortcut runs this in a terminal window, so the first
# `docker compose` build is visible and a sudo prompt (for a user who is not in
# the `docker` group) can be answered. When the dashboard already answers, it
# only opens the browser.
#
# Environment:
#   OPENSHORTS_DIR   the openshorts checkout (default: this script's parent dir,
#                    or the one baked in by install-desktop-shortcut.sh)
#   OPENSHORTS_URL   dashboard URL (default: http://localhost:5175)
#   OPENSHORTS_WAIT  seconds to wait for the dashboard to answer (default: 900)
set -euo pipefail

OPENSHORTS_DIR="${OPENSHORTS_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OPENSHORTS_URL="${OPENSHORTS_URL:-http://localhost:5175}"
OPENSHORTS_WAIT="${OPENSHORTS_WAIT:-900}"

say() { printf '[openshorts] %s\n' "$*"; }

fail() {
    say "ERROR: $*" >&2
    # Launched from a desktop shortcut the terminal closes on exit: keep the
    # message on screen until it has been read.
    if [ -t 0 ]; then
        read -r -p "Press Enter to close..." _ || true
    fi
    exit 1
}

dashboard_up() { curl -fsS -o /dev/null --max-time 3 "$OPENSHORTS_URL"; }

open_browser() {
    if command -v xdg-open >/dev/null 2>&1; then
        xdg-open "$OPENSHORTS_URL" >/dev/null 2>&1 &
    elif command -v open >/dev/null 2>&1; then
        open "$OPENSHORTS_URL"
    else
        say "Open $OPENSHORTS_URL in your browser."
    fi
}

# Sets DOCKER to the command that can reach the daemon: plain `docker`, or
# `sudo docker` when the socket is there but this user may not use it.
pick_docker() {
    command -v docker >/dev/null 2>&1 || fail "docker is not installed (see README: Requirements)."
    if docker info >/dev/null 2>&1; then
        DOCKER=(docker)
    elif command -v sudo >/dev/null 2>&1; then
        say "This user cannot reach the Docker daemon; using sudo."
        say "(Add yourself to the 'docker' group to skip this prompt.)"
        DOCKER=(sudo docker)
        "${DOCKER[@]}" info >/dev/null 2>&1 || fail "the Docker daemon is not running (try: sudo systemctl start docker)."
    else
        fail "cannot reach the Docker daemon and sudo is not available."
    fi
}

compose() {
    "${DOCKER[@]}" compose --project-directory "$OPENSHORTS_DIR" \
        -f "$OPENSHORTS_DIR/docker-compose.yml" "$@"
}

cmd_start() {
    if dashboard_up; then
        say "Already running."
        open_browser
        return
    fi
    [ -f "$OPENSHORTS_DIR/docker-compose.yml" ] \
        || fail "no docker-compose.yml in $OPENSHORTS_DIR (set OPENSHORTS_DIR)."
    pick_docker
    say "Starting OpenShorts from $OPENSHORTS_DIR (the first start builds the images and takes a while)..."
    compose up -d || fail "docker compose up failed (see above)."

    say "Waiting for the dashboard at $OPENSHORTS_URL ..."
    local waited=0
    until dashboard_up; do
        if [ "$waited" -ge "$OPENSHORTS_WAIT" ]; then
            fail "the dashboard did not answer within ${OPENSHORTS_WAIT}s (check: docker compose logs)."
        fi
        sleep 2
        waited=$((waited + 2))
    done
    say "Ready."
    open_browser
    # Give xdg-open a moment to hand the URL over before the terminal closes.
    sleep 1
}

cmd_stop() {
    pick_docker
    say "Stopping OpenShorts..."
    compose stop || fail "docker compose stop failed (see above)."
    say "Stopped."
    sleep 1
}

cmd_status() {
    if dashboard_up; then
        say "Running: $OPENSHORTS_URL"
    else
        say "Not running."
        return 1
    fi
}

case "${1:-start}" in
    start)  cmd_start ;;
    stop)   cmd_stop ;;
    status) cmd_status ;;
    -h|--help) sed -n '2,17p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' ;;
    *) fail "unknown command '$1' (start, stop or status)." ;;
esac
