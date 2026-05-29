FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        git \
        rsync \
        sqlite3 \
    && rm -rf /var/lib/apt/lists/*

COPY . /opt/omega-source

WORKDIR /workspaces/Omega
