#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")/.."

IMAGE="ghcr.io/sinavse/multiagent_copilot_cp:latest"

export EMBEDDING_CERT_HOST_PATH="$HOME/capstone-certs/vm02-ca.crt"

echo "===== Validating configuration ====="

test -f .env || {
    echo "ERROR: .env file missing"
    exit 1
}

test -f "$EMBEDDING_CERT_HOST_PATH" || {
    echo "ERROR: SSL certificate missing"
    exit 1
}

docker compose config --quiet

echo "===== Downloading Docker image ====="

docker pull "$IMAGE"

docker tag "$IMAGE" operations-copilot:1.0

echo "===== Deploying application ====="

docker compose up -d --no-build --pull never

echo "===== Validating application health ====="

for i in $(seq 1 30); do
    if curl -fsS http://localhost:8501/_stcore/health; then
        echo ""
        echo "DEPLOYMENT SUCCESSFUL"

        docker compose ps

        if [ -n "${CODESPACE_NAME:-}" ]; then
            echo "Application URL:"
            echo "https://${CODESPACE_NAME}-8501.${GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN:-app.github.dev}"
        fi

        exit 0
    fi

    sleep 5
done

echo "ERROR: Application health check failed"

docker compose logs --tail=100

exit 1
