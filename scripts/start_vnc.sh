#!/usr/bin/env bash
# ==============================================================================
# start_vnc.sh - Xvfb + Openbox + x11vnc + noVNC inside the dev container.
#
#   start_vnc.sh [RESOLUTION] [-- CMD ...]   start, then exec CMD (or idle as PID 1)
#   start_vnc.sh start [RESOLUTION]          start and RETURN -- for `make start-vnc`
#   start_vnc.sh stop                        stop it
#   start_vnc.sh status                      report what is running
#
# The bare form is the image CMD: it must not exit, or the container would stop.
# The `start` form is for use from an already-running container, where blocking
# would wedge the caller's terminal on `tail -f /dev/null`.
#
# Everything is driven by environment variables so that nothing here is hard-wired
# to a port or display another container might also be using. Defaults match
# docker-compose.yml.
#
# Runs entirely as the unprivileged container user: the X socket, the lock file and
# the logs all live in the container's own /tmp and its own home, so two containers
# on one host never see each other's display.
# LINT.IfChange(vnc_env)
# ==============================================================================
set -uo pipefail

VNC_DISPLAY="${VNC_DISPLAY:-:1}"
VNC_PORT="${VNC_PORT:-5900}"
NOVNC_PORT="${NOVNC_PORT:-6080}"
VNC_DEPTH="${VNC_DEPTH:-24}"
# LINT.ThenChange(//docker-compose.yml:vnc_env)

DISPLAY_NUM="${VNC_DISPLAY#:}"
LOG_DIR="${HOME:-/tmp}/.twm-vnc"
mkdir -p "${LOG_DIR}" 2>/dev/null || LOG_DIR=/tmp

# Match processes on this display/port only, so `stop` in one container-like context
# never reaps another display's server.
_xvfb_pat="Xvfb ${VNC_DISPLAY} "
_x11vnc_pat="x11vnc.*-rfbport ${VNC_PORT}"
_ws_pat="websockify.*${NOVNC_PORT}"

resolve_novnc_dir() {
    for d in /usr/share/novnc /usr/share/noVNC; do
        [ -d "$d" ] && { echo "$d"; return; }
    done
}

stop_services() {
    echo "🛑 Stopping VNC services on ${VNC_DISPLAY} (rfb ${VNC_PORT}, web ${NOVNC_PORT})..."
    pkill -f "${_ws_pat}"     2>/dev/null || true
    pkill -f "${_x11vnc_pat}" 2>/dev/null || true
    pkill -x openbox          2>/dev/null || true
    pkill -f "${_xvfb_pat}"   2>/dev/null || true
    rm -f "/tmp/.X${DISPLAY_NUM}-lock" "/tmp/.X11-unix/X${DISPLAY_NUM}" 2>/dev/null || true
    echo "✅ Stopped."
}

show_status() {
    echo "=== VNC services (display ${VNC_DISPLAY}) ==="
    pgrep -fl "${_xvfb_pat}"   || echo "Xvfb:       not running"
    pgrep -xl openbox          || echo "openbox:    not running"
    pgrep -fl "${_x11vnc_pat}" || echo "x11vnc:     not running"
    pgrep -fl "${_ws_pat}"     || echo "websockify: not running"
}

FOREGROUND=1
case "${1:-}" in
    stop)   stop_services; exit 0 ;;
    status) show_status;   exit 0 ;;
    start)  FOREGROUND=0; shift ;;
esac

# Resolution: first positional arg, else VNC_RESOLUTION, else probe the host's DRM
# modes through the bind mount, else a safe default.
if [ -n "${1:-}" ] && [ "${1:-}" != "--" ]; then
    RESOLUTION="$1"; shift
elif [ -n "${VNC_RESOLUTION:-}" ]; then
    RESOLUTION="${VNC_RESOLUTION}"
else
    _detected="$(cat /sys/class/drm/*/modes 2>/dev/null | head -1 || true)"
    RESOLUTION="${_detected:-1920x1080}"
fi
[ "${1:-}" = "--" ] && shift

# Software rendering for MuJoCo/Brax/OSMesa under Xvfb.
export LIBGL_ALWAYS_SOFTWARE=1
export LIBGL_ALWAYS_INDIRECT=0
export MESA_LOADER_DRIVER_OVERRIDE=llvmpipe
export GALLIUM_DRIVER=llvmpipe
export MESA_GL_VERSION_OVERRIDE=3.3

# --- host credentials -------------------------------------------------------
# Both mounts are read-only and both have an empty stand-in as their default, so
# this is a no-op when the host supplied neither. Only ever writes into the
# container user's own home -- never chowns or walks /home/*, which as root would
# reach into other users' files.
CONTAINER_HOME="$(getent passwd "$(id -u)" 2>/dev/null | cut -d: -f6)"
CONTAINER_HOME="${CONTAINER_HOME:-${HOME:-/tmp}}"

if [ -f /tmp/host.gitconfig ]; then
    cp /tmp/host.gitconfig "${CONTAINER_HOME}/.gitconfig" 2>/dev/null \
        && echo "🔑 Imported host git configuration."
fi
# Test for actual key material, not merely a non-empty directory: the default
# stand-in (docker/empty) contains a .gitkeep, which would make an `ls -A` test
# true and print "Imported host SSH credentials" having copied nothing but that.
if [ -d /tmp/host_ssh ] && ls /tmp/host_ssh/id_* >/dev/null 2>&1; then
    mkdir -p "${CONTAINER_HOME}/.ssh"
    cp -r /tmp/host_ssh/. "${CONTAINER_HOME}/.ssh/" 2>/dev/null || true
    chmod 700 "${CONTAINER_HOME}/.ssh" 2>/dev/null || true
    find "${CONTAINER_HOME}/.ssh" -type f ! -name '*.pub' -exec chmod 600 {} + 2>/dev/null || true
    find "${CONTAINER_HOME}/.ssh" -type f   -name '*.pub' -exec chmod 644 {} + 2>/dev/null || true
    echo "🔑 Imported host SSH credentials."
fi

# The workspace is bind-mounted from the host, so its .git is owned by the host user;
# git refuses to operate on it unless the path is marked safe.
git config --global --add safe.directory /workspace 2>/dev/null || true

# --- X stack ----------------------------------------------------------------
# A stale lock with no Xvfb behind it blocks startup; clear it, but only when
# nothing is actually serving this display.
if ! pgrep -f "${_xvfb_pat}" >/dev/null 2>&1; then
    rm -f "/tmp/.X${DISPLAY_NUM}-lock" "/tmp/.X11-unix/X${DISPLAY_NUM}" 2>/dev/null || true
fi

if ! pgrep -f "${_xvfb_pat}" >/dev/null 2>&1; then
    echo "🖥️  Starting Xvfb on ${VNC_DISPLAY} (${RESOLUTION}x${VNC_DEPTH})..."
    Xvfb "${VNC_DISPLAY}" -screen 0 "${RESOLUTION}x${VNC_DEPTH}" \
        +extension GLX +extension RENDER -noreset -ac >"${LOG_DIR}/xvfb.log" 2>&1 &
    sleep 1
else
    echo "ℹ️  Xvfb already running on ${VNC_DISPLAY}."
fi

export DISPLAY="${VNC_DISPLAY}"

if ! pgrep -x openbox >/dev/null 2>&1; then
    echo "🪟 Starting Openbox..."
    openbox --sm-disable >"${LOG_DIR}/openbox.log" 2>&1 &
    sleep 0.5
fi

if ! pgrep -f "${_x11vnc_pat}" >/dev/null 2>&1; then
    echo "📡 Starting x11vnc on container port ${VNC_PORT}..."
    # -localhost is deliberately NOT set: the connection arrives from Docker's
    # userland proxy, not from loopback. Exposure is bounded on the host side
    # instead, by BIND_ADDR in docker-compose.yml (127.0.0.1 by default).
    x11vnc -display "${VNC_DISPLAY}" -rfbport "${VNC_PORT}" -shared -forever -nopw \
        -xkb -noxrecord -noxfixes -noxdamage -quiet -bg -o "${LOG_DIR}/x11vnc.log"
    sleep 0.5
fi

NOVNC_DIR="$(resolve_novnc_dir)"
if command -v websockify >/dev/null 2>&1 && ! pgrep -f "${_ws_pat}" >/dev/null 2>&1; then
    echo "🌐 Starting noVNC on container port ${NOVNC_PORT}..."
    if [ -n "${NOVNC_DIR}" ]; then
        websockify --web "${NOVNC_DIR}" "${NOVNC_PORT}" "localhost:${VNC_PORT}" \
            >"${LOG_DIR}/novnc.log" 2>&1 &
    else
        websockify "${NOVNC_PORT}" "localhost:${VNC_PORT}" >"${LOG_DIR}/novnc.log" 2>&1 &
    fi
    sleep 0.5
fi

# VNC_PORT / NOVNC_PORT are the ports bound INSIDE the container. The browser must
# use whatever host port this container was published on, which differs whenever
# another stack already holds the default -- printing the internal port as a URL
# sends you to a different container's desktop.
HOST_NOVNC="${HOST_NOVNC_PORT:-${NOVNC_PORT}}"
HOST_VNC="${HOST_VNC_PORT:-${VNC_PORT}}"

echo "=========================================================="
echo "🎉 Visualization ready (display ${VNC_DISPLAY}, ${RESOLUTION})"
echo "👉 Browser:    http://localhost:${HOST_NOVNC}/vnc_lite.html?scale=true"
echo "👉 VNC client: localhost:${HOST_VNC}"
if [ "${HOST_NOVNC}" != "${NOVNC_PORT}" ] || [ "${HOST_VNC}" != "${VNC_PORT}" ]; then
    echo "   (inside the container these are ${NOVNC_PORT} / ${VNC_PORT}; the host"
    echo "    mapping differs so this stack does not collide with its neighbours)"
fi
echo "   Logs: ${LOG_DIR}"
echo "=========================================================="

if [ "$#" -gt 0 ]; then
    exec "$@"
elif [ "${FOREGROUND}" -eq 0 ]; then
    # `start` subcommand: the services are already running in the background, so
    # return control to the caller rather than blocking.
    exit 0
else
    # Image CMD: block, because PID 1 exiting would stop the container.
    exec tail -f /dev/null
fi
