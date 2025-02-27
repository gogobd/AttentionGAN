FROM debian:bullseye-slim

ENV SHELL /bin/bash
  
# Install system dependencies
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    curl \
    wget \
    git \
    screen \
    unzip \
    vim \
    procps \
    locales \
    python3-pip \
 && apt-get clean

# Python unicode issues
RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && locale-gen
ENV LC_ALL=en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US.UTF-8

# Code server
# https://github.com/coder/code-server/releases
ARG VERSION=4.0.2
RUN mkdir -p ~/.local/lib ~/.local/bin
RUN curl -sfL https://github.com/cdr/code-server/releases/download/v$VERSION/code-server-$VERSION-linux-amd64.tar.gz | tar -C ~/.local/lib -xz
RUN mv ~/.local/lib/code-server-$VERSION-linux-amd64 ~/.local/lib/code-server-$VERSION
RUN ln -s ~/.local/lib/code-server-$VERSION/bin/code-server ~/.local/bin/code-server
RUN PATH="~/.local/bin:$PATH"

WORKDIR /app
ENV SHELL /bin/bash
CMD ~/.local/bin/code-server --bind-addr 0.0.0.0:8080 /app

# docker build -t attentiongan .
# docker run --name attentiongan --ipc host --gpus all -v <DataDir/deeplearning>:/data -v $(pwd):/app -p 8080-8089:8080-8089 -it -d attentiongan
# docker exec -it attentiongan cat /root/.config/code-server/config.yaml

