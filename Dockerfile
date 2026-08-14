FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV DISPLAY=:1
ENV TERM=xterm-256color
ENV PYTHONPATH=/workspace

ARG USER_ID=1000
ARG GROUP_ID=1000
ARG USERNAME=devuser

# Install system dependencies, X11, VNC, noVNC, EGL/Mesa graphics libraries, git, SSH, bash-completion
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    python-is-python3 \
    python3-tk \
    git \
    openssh-client \
    bash-completion \
    sudo \
    curl \
    wget \
    make \
    build-essential \
    xvfb \
    x11vnc \
    openbox \
    novnc \
    websockify \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libosmesa6-dev \
    libegl1-mesa \
    libgles2-mesa-dev \
    patchelf \
    ffmpeg \
    less \
    locales \
    && rm -rf /var/lib/apt/lists/*

# Install modern noVNC (v1.5.0) to eliminate legacy localStorage/cookie settings bugs
RUN rm -rf /usr/share/novnc && \
    mkdir -p /usr/share/novnc && \
    curl -fsSL https://github.com/novnc/noVNC/archive/refs/tags/v1.5.0.tar.gz | tar -xz -C /usr/share/novnc --strip-components=1 && \
    ln -sf /usr/share/novnc/vnc.html /usr/share/novnc/index.html

# Install Bazelisk as /usr/local/bin/bazel
RUN curl -fsSL https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64 -o /usr/local/bin/bazel \
    && chmod +x /usr/local/bin/bazel

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Setup non-root user matching host UID/GID
RUN if ! getent group ${GROUP_ID} > /dev/null 2>&1; then \
        groupadd -g ${GROUP_ID} ${USERNAME}; \
    fi && \
    if ! getent passwd ${USER_ID} > /dev/null 2>&1; then \
        useradd -m -u ${USER_ID} -g ${GROUP_ID} -s /bin/bash ${USERNAME}; \
    else \
        EXISTING_USER=$(getent passwd ${USER_ID} | cut -d: -f1); \
        usermod -l ${USERNAME} ${EXISTING_USER} 2>/dev/null || true; \
    fi && \
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Copy shell initialization script
COPY scripts/shell_init.sh /etc/profile.d/twm_shell.sh
RUN chmod +x /etc/profile.d/twm_shell.sh && \
    echo '[ -f /etc/profile.d/twm_shell.sh ] && source /etc/profile.d/twm_shell.sh' >> /etc/bash.bashrc && \
    echo '[ -f /etc/profile.d/twm_shell.sh ] && source /etc/profile.d/twm_shell.sh' >> /home/${USERNAME}/.bashrc

WORKDIR /workspace

# Copy requirements and install
COPY requirements.txt /workspace/
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy start_vnc script
COPY scripts/start_vnc.sh /usr/local/bin/start_vnc.sh
RUN chmod +x /usr/local/bin/start_vnc.sh

# Ensure user owns their home and workspace
RUN chown -R ${USER_ID}:${GROUP_ID} /home/${USERNAME} /workspace

EXPOSE 5900 6080

CMD ["/usr/local/bin/start_vnc.sh"]
