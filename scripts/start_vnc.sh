#!/usr/bin/env bash
set -e

echo "Starting VNC Server and Development environment..."

# --- Import SSH keys & Git configuration if mounted ---
for hdir in /root /home/*; do
    if [ -d "$hdir" ]; then
        if [ -d /tmp/host_ssh ]; then
            echo "Importing host SSH credentials into $hdir/.ssh..."
            mkdir -p "$hdir/.ssh"
            cp -rn /tmp/host_ssh/* "$hdir/.ssh/" 2>/dev/null || true
            chmod 700 "$hdir/.ssh" 2>/dev/null || true
            chmod 600 "$hdir/.ssh"/* 2>/dev/null || true
            chmod 644 "$hdir/.ssh"/*.pub 2>/dev/null || true
        fi
        if [ -f /tmp/host.gitconfig ]; then
            echo "Importing host Git config into $hdir/.gitconfig..."
            cp /tmp/host.gitconfig "$hdir/.gitconfig" 2>/dev/null || true
        fi
        # Ensure user ownership
        if [ "$hdir" != "/root" ]; then
            chown -R devuser:devuser "$hdir" 2>/dev/null || true
        fi
    fi
done

# --- Configure Mesa software rendering for OpenGL/MJX/Brax ---
export LIBGL_ALWAYS_SOFTWARE=1
export LIBGL_ALWAYS_INDIRECT=0
export MESA_LOADER_DRIVER_OVERRIDE=llvmpipe
export GALLIUM_DRIVER=llvmpipe
export MESA_GL_VERSION_OVERRIDE=3.3

# Clean up any residual lock files
rm -f /tmp/.X1-lock /tmp/.X11-unix/X1 2>/dev/null || true

# 1. Start Xvfb virtual framebuffer on display :1
echo "Starting Xvfb display on :1 (1920x1080)..."
Xvfb :1 -screen 0 1920x1080x24 +iglx >/tmp/xvfb.log 2>&1 &
export DISPLAY=:1
sleep 1

# 2. Start Openbox Window Manager
echo "Starting Openbox window manager..."
openbox >/dev/null 2>&1 &
sleep 0.5

# 3. Start x11vnc server on port 5900
echo "Starting x11vnc on port 5900..."
x11vnc -display :1 -forever -nopw -shared -rfbport 5900 -noxdamage -bg -o /tmp/x11vnc.log >/dev/null 2>&1 || true
sleep 0.5

# 4. Start noVNC web interface on port 6080 with auto-connect
NOVNC_DIR="/usr/share/novnc"
if [ ! -d "$NOVNC_DIR" ]; then
    NOVNC_DIR="/usr/share/noVNC"
fi

if command -v websockify > /dev/null && [ -d "$NOVNC_DIR" ]; then
    echo "Starting noVNC websockify on port 6080..."
    # Symlink lightweight vnc_lite.html to index.html for instant direct connection
    ln -sf "${NOVNC_DIR}/vnc_lite.html" "${NOVNC_DIR}/index.html" 2>/dev/null || true
    websockify --web "${NOVNC_DIR}" 6080 localhost:5900 >/tmp/novnc.log 2>&1 &
    sleep 0.5
fi

echo "=========================================================="
echo "🖥️  VNC Desktop running on port 5900 (Display :1)"
echo "🌐 Lightweight noVNC Web UI ready at:"
echo "   http://localhost:6080/vnc_lite.html?scale=true (or http://localhost:6080/)"
echo "=========================================================="

# Keep container alive
if [ $# -gt 0 ]; then
    exec "$@"
else
    exec tail -f /dev/null
fi
