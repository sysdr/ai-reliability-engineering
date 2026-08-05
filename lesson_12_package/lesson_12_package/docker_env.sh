#!/usr/bin/env bash
# Prefer Docker Desktop (Windows docker.exe) so images/containers appear in Docker Desktop UI.
# Falls back to the local WSL docker daemon when Desktop is unavailable.
resolve_docker() {
    local candidates=(
        "/mnt/c/Program Files/Docker/Docker/resources/bin/docker.exe"
        "/mnt/c/ProgramData/DockerDesktop/version-bin/docker.exe"
        "docker.exe"
        "docker"
    )
    local c
    for c in "${candidates[@]}"; do
        if command -v "$c" &> /dev/null || [ -x "$c" ]; then
            if "$c" info &> /dev/null; then
                echo "$c"
                return 0
            fi
        fi
    done
    return 1
}

# Convert a Linux path to a Windows path usable by docker.exe volume mounts.
to_docker_path() {
    local path="$1"
    if command -v wslpath &> /dev/null; then
        wslpath -w "$path"
    else
        echo "$path"
    fi
}
