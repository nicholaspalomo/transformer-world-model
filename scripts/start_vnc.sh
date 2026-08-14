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

# Clean up any residual lock files
rm -f /tmp/.X1-lock /tmp/.X11-unix/X1

# 1. Start Xvfb virtual framebuffer on display :1
Xvfb :1 -screen 0 1280x800x24 &
export DISPLAY=:1
sleep 1

# 2. Start Openbox Window Manager
openbox-session &
sleep 1

# 3. Start x11vnc server on port 5900
x11vnc -display :1 -forever -nopw -shared -rfbport 5900 -bg

# 4. Start noVNC web interface on port 6080 if available
if command -v websockify > /dev/null && [ -d /usr/share/novnc ]; then
    echo "Starting noVNC web interface on port 6080..."
    websockify --web /usr/share/novnc/ 6080 localhost:5900 &
fi

echo "=========================================================="
echo "VNC server running on port 5900 (Display :1)"
echo "noVNC web browser available on port 6080"
echo "Connect via SSH forwarding: ssh -L 5903:localhost:5903 -L 6082:localhost:6082 user@remote"
echo "=========================================================="

# Keep container alive
if [ $# -gt 0 ]; then
    exec "$@"
else
    exec tail -f /dev/null
fi
