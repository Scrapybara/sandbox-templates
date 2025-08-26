# CodeSandbox supports debian & ubuntu based images
# This universal image contains almost all languages and tools
FROM ghcr.io/codesandbox/devcontainers/universal:latest

# Example: install htop by default
# RUN apt update -y && apt install -y htop
RUN apt update && apt install -y curl libreoffice ripgrep fzf tre-agrep \
gpg && \
mkdir -p /etc/apt/keyrings && \
wget -qO- https://raw.githubusercontent.com/eza-community/eza/main/deb.asc | gpg --dearmor -o /etc/apt/keyrings/gierens.gpg && \
echo "deb [signed-by=/etc/apt/keyrings/gierens.gpg] http://deb.gierens.de stable main" | tee /etc/apt/sources.list.d/gierens.list && \
chmod 644 /etc/apt/keyrings/gierens.gpg /etc/apt/sources.list.d/gierens.list && \
apt-get update && \
apt-get install -y eza && \
apt-get clean && \
rm -rf /var/lib/apt/lists/*

WORKDIR /project/workspace

# Install Bun to /project/workspace/.bun
RUN curl -fsSL https://bun.sh/install | bash

# Add Bun to PATH
ENV BUN_INSTALL="/root/.bun"
ENV PATH="$BUN_INSTALL/bin:$PATH"

# Update pnpm to latest version
RUN npm install -g pnpm@latest --force

# Install s5cmd
RUN pip install s5cmd --user --no-warn-script-location

# Add s5cmd to PATH
ENV PATH="/root/.local/bin:$PATH"

# Install language servers and FastAPI dependencies
RUN pip install pyright ruff multilspy fastapi uvicorn[standard] aiofiles
RUN npm install -g typescript @typescript-eslint/parser @typescript-eslint/eslint-plugin eslint
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Install GitHub CLI
RUN apt update && \
    mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | tee /etc/apt/keyrings/githubcli-archive-keyring.gpg >/dev/null && \
    chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list >/dev/null && \
    apt update && \
    apt install -y gh && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install code-server
RUN curl -fsSL https://code-server.dev/install.sh | sh

# The project will be mounted in /workspace by default

# Every new shell will open inside the built container

# On top of this build, we install `zsh`, `git` and `docker` as additional
# Docker layers.
