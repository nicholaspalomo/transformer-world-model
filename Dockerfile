# syntax=docker/dockerfile:1
# ==============================================================================
# Development image for the Transformer World Model / GRPO Diffusion Policy.
#
# Built and run through docker-compose.yml, which names it after the Compose
# project (default `twm`) so it cannot be confused with another stack's image.
# Build it with `make docker-build`, which passes the host UID/GID through.
# ==============================================================================
FROM ubuntu:22.04

ARG USER_ID=1000
ARG GROUP_ID=1000
ARG USERNAME=devuser

# Pinned rather than `latest`: an unpinned release URL makes two builds of the same
# commit produce different toolchains.
ARG BAZELISK_VERSION=v1.25.0
ARG NOVNC_VERSION=1.5.0

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TERM=xterm-256color \
    LANG=C.UTF-8 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/games:/usr/games

# System dependencies: Python, X11/VNC/noVNC, Mesa software rendering for
# MuJoCo/Brax, plus the git/ssh/bash-completion niceties the dev shell expects.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \
        python3-venv \
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
        psmisc \
        procps \
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

# Modern noVNC, to avoid the legacy localStorage/cookie settings bugs in the
# distro package. The regex makes the settings-panel listeners optional so
# vnc_lite.html works without the full UI being present.
RUN rm -rf /usr/share/novnc \
    && mkdir -p /usr/share/novnc \
    && curl -fsSL "https://github.com/novnc/noVNC/archive/refs/tags/v${NOVNC_VERSION}.tar.gz" \
        | tar -xz -C /usr/share/novnc --strip-components=1 \
    && python3 -c "import re; c=open('/usr/share/novnc/app/ui.js').read(); c=re.sub(r'document\.getElementById\(([^)]+)\)\s*\.addEventListener', r'document.getElementById(\1)?.addEventListener', c).replace('settingElem.addEventListener', 'settingElem?.addEventListener'); open('/usr/share/novnc/app/ui.js','w').write(c)" \
    && ln -sf /usr/share/novnc/vnc_lite.html /usr/share/novnc/index.html

RUN curl -fsSL "https://github.com/bazelbuild/bazelisk/releases/download/${BAZELISK_VERSION}/bazelisk-linux-amd64" \
        -o /usr/local/bin/bazel \
    && chmod +x /usr/local/bin/bazel

# ------------------------------------------------------------------ python env
# A dedicated virtualenv rather than the distro site-packages. Installing this
# project's pinned jax/flax/brax over Ubuntu's python3-* packages (which is what
# `pip install --ignore-installed` into /usr did) can leave apt-managed packages
# half-replaced and unrepairable. /opt/venv is first on PATH, so `python`, `pip`
# and `pytest` all resolve to it without anyone having to spell out a path.
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip setuptools wheel

# Only the dependency metadata, so this layer is cached until the deps change.
# The source tree itself arrives at runtime through the bind mount.
COPY pyproject.toml README.md /tmp/twm-build/
RUN /opt/venv/bin/pip install --no-cache-dir "/tmp/twm-build/[dev]" \
    && /opt/venv/bin/pip uninstall -y twm \
    && rm -rf /tmp/twm-build

# -------------------------------------------------------------------- dev user
# Match the host UID/GID so files written into the bind-mounted workspace stay
# owned by the host user instead of root. If the image's base already ships a user
# at that UID (Ubuntu 24.04+ ships `ubuntu` at 1000), reuse it rather than failing.
RUN if getent passwd ${USER_ID} >/dev/null; then \
        EXISTING=$(getent passwd ${USER_ID} | cut -d: -f1); \
        if [ "${EXISTING}" != "${USERNAME}" ]; then usermod -l ${USERNAME} "${EXISTING}"; fi; \
        usermod -d /home/${USERNAME} -m ${USERNAME} 2>/dev/null || true; \
    else \
        getent group ${GROUP_ID} >/dev/null || groupadd -g ${GROUP_ID} ${USERNAME}; \
        useradd -m -u ${USER_ID} -g ${GROUP_ID} -s /bin/bash ${USERNAME}; \
    fi \
    && mkdir -p /home/${USERNAME}/.cache/bazel /home/${USERNAME}/.cache/pip /home/${USERNAME}/.twm-vnc \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME} \
    && chown -R ${USER_ID}:${GROUP_ID} /home/${USERNAME}

# Shell init: prompt, completion, PYTHONPATH. Also bind-mounted over at runtime so
# edits take effect without a rebuild; baked in so the image works standalone.
COPY scripts/shell_init.sh /etc/profile.d/twm_shell.sh
COPY scripts/start_vnc.sh /usr/local/bin/start_vnc.sh
RUN chmod +x /etc/profile.d/twm_shell.sh /usr/local/bin/start_vnc.sh \
    && printf '%s\n' '[ -f /etc/profile.d/twm_shell.sh ] && source /etc/profile.d/twm_shell.sh' \
        >> /etc/bash.bashrc

ENV HOME=/home/${USERNAME} \
    PYTHONPATH=/workspace \
    DISPLAY=:1

WORKDIR /workspace

# Default to the unprivileged user even when the image is run outside Compose
# (`docker run`, `tools/ci_local.sh`), so a stray run cannot write root-owned files
# into a bind-mounted workspace.
USER ${USER_ID}:${GROUP_ID}

# Documentation only -- the host-side mapping lives in docker-compose.yml and is
# configurable there, because these defaults collide with every other VNC container.
EXPOSE 5900 6080 6006 8888

CMD ["/usr/local/bin/start_vnc.sh"]
