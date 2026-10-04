#!/usr/bin/env bash
# ==============================================================================
# Shell initialization inside the development container.
#
# Sourced two ways: from /etc/bash.bashrc for interactive shells, and via
# BASH_ENV (set in docker-compose.yml) for NON-interactive ones -- which includes
# every `docker compose exec ... bash -lc` the Makefile issues.
#
# Everything that costs real work is therefore gated on the shell being
# interactive. In particular `bazel completion bash` starts a Bazel client, which
# in turn starts (or wakes) a ~500 MB JVM server; running that from BASH_ENV meant
# a Bazel server was spawned by every single non-interactive command in the
# container, including `make test`.
# ==============================================================================

# Environment: cheap, and wanted in both interactive and non-interactive shells.
# Idempotent: this file is sourced from /etc/bash.bashrc AND via BASH_ENV, and a
# login shell that spawns a subshell would otherwise end up with /workspace
# repeated several times over in PYTHONPATH.
case ":${PYTHONPATH-}:" in
    *:/workspace:*) ;;
    *) export PYTHONPATH="/workspace${PYTHONPATH:+:${PYTHONPATH}}" ;;
esac
export TERM="${TERM:-xterm-256color}"

# Everything below is interactive-only.
case $- in
    *i*) ;;
    *) return 0 ;;
esac

# Bash completion framework
if [ -f /usr/share/bash-completion/bash_completion ]; then
    source /usr/share/bash-completion/bash_completion
elif [ -f /etc/bash_completion ]; then
    source /etc/bash_completion
fi

# Git completion & prompt
if [ -f /usr/share/bash-completion/completions/git ]; then
    source /usr/share/bash-completion/completions/git
fi
if [ -f /usr/lib/git-core/git-sh-prompt ]; then
    source /usr/lib/git-core/git-sh-prompt
elif [ -f /etc/bash_completion.d/git-prompt ]; then
    source /etc/bash_completion.d/git-prompt
fi

GIT_PS1_SHOWDIRTYSTATE=1
GIT_PS1_SHOWUNTRACKEDFILES=1
GIT_PS1_SHOWUPSTREAM="auto"

if type __git_ps1 &>/dev/null; then
    export PS1='\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[01;33m\]$(__git_ps1 " (%s)")\[\033[00m\]\$ '
else
    export PS1='\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '
fi

# Bazel completion. Deferred behind a lazy loader: sourcing it eagerly starts a
# Bazel server on every new shell, which is minutes of JVM startup and hundreds of
# megabytes for someone who only wanted to run pytest.
_twm_lazy_bazel_completion() {
    complete -r bazel 2>/dev/null
    source <(bazel completion bash 2>/dev/null) 2>/dev/null || true
    return 124
}
if command -v bazel &>/dev/null; then
    complete -F _twm_lazy_bazel_completion -o bashdefault -o default bazel
fi

# Make target completion, read straight out of the Makefile.
_make_targets() {
    local cur="${COMP_WORDS[COMP_CWORD]}"
    if [ -f Makefile ]; then
        COMPREPLY=($(compgen -W "$(grep -oE '^[a-zA-Z0-9_-]+:' Makefile | sed 's/://' | sort -u)" -- "$cur"))
    fi
}
complete -F _make_targets make

alias ls='ls --color=auto'
alias ll='ls -la --color=auto'
alias la='ls -A --color=auto'
alias l='ls -CF --color=auto'
alias grep='grep --color=auto'
