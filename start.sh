#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
docker buildx build --builder default --load -t solo22-materialscope:1.0 .
docker compose up -d --no-build
printf '%s\n' 'MaterialScope: http://127.0.0.1:18922/' '停止并保留数据: docker compose down'
