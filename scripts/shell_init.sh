#!/usr/bin/env bash
# ==============================================================================
# Shell initialization inside Docker container
# Provides: Native colored prompt, git branch prompt, git/make/bazel tab completion
# ==============================================================================

# Enable bash completion framework
if [ -f /usr/share/bash-completion/bash_completion ]; then
    source /usr/share/bash-completion/bash_completion
elif [ -f /etc/bash_completion ]; then
    source /etc/bash_completion
fi

# Git tab completion & prompt
if [ -f /usr/share/bash-completion/completions/git ]; then
    source /usr/share/bash-completion/completions/git
fi

if [ -f /usr/lib/git-core/git-sh-prompt ]; then
    source /usr/lib/git-core/git-sh-prompt
elif [ -f /etc/bash_completion.d/git-prompt ]; then
    source /etc/bash_completion.d/git-prompt
fi

# Define native-style colored prompt with active git branch
GIT_PS1_SHOWDIRTYSTATE=1
GIT_PS1_SHOWUNTRACKEDFILES=1
GIT_PS1_SHOWUPSTREAM="auto"

if type __git_ps1 &>/dev/null; then
    export PS1='\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[01;33m\]$(__git_ps1 " (%s)")\[\033[00m\]\$ '
else
    export PS1='\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '
fi

# Bazel tab completion
if command -v bazel &>/dev/null; then
    source <(bazel completion bash 2>/dev/null) || true
fi

# Make target tab completion
_make_targets() {
    local cur="${COMP_WORDS[COMP_CWORD]}"
    local makefile="Makefile"
    if [ -f "$makefile" ]; then
        local targets=$(grep -oE '^[a-zA-Z0-9_-]+:' "$makefile" | sed 's/://' | sort -u)
        COMPREPLY=($(compgen -W "$targets" -- "$cur"))
    fi
}
complete -F _make_targets make

# Useful aliases
alias ls='ls --color=auto'
alias ll='ls -la --color=auto'
alias la='ls -A --color=auto'
alias l='ls -CF --color=auto'
alias grep='grep --color=auto'

# Workspace environment
export PYTHONPATH=/workspace:${PYTHONPATH}
export TERM=xterm-256color
