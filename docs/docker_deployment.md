# Docker Deployment Guide

## Overview

The Employee Operations Copilot is containerized using
Python 3.11 and Streamlit.

The container runs as a non-root user and exposes
port 8501.

## Prerequisites

- Docker Engine or Docker Desktop
- Docker Compose V2
- Valid hosted inference API credentials
- Valid hosted embedding API credentials
- Embedding endpoint CA certificate
- Network access to the approved hosted API endpoints

## Environment Configuration

Create a local .env file using .env.example.

Never commit API credentials or private certificates.

Set EMBEDDING_CERT_HOST_PATH to the absolute host
path of the approved embedding CA certificate.

Docker Compose mounts this certificate read-only at:

/run/certs/embedding-ca.crt

The application receives the container-compatible path
through EMBEDDING_SSL_CERT_PATH.

Review any other certificate or filesystem settings
and replace host-specific paths where required.

## Build

    docker compose build

## Validate Compose Configuration

    docker compose config --quiet

## Start Application

    docker compose up -d

## Access Application

    http://localhost:8501

## Check Container Health

    docker compose ps

    curl http://localhost:8501/_stcore/health

## View Logs

    docker compose logs --tail=100

## Stop Application

    docker compose down

## Persistent Storage

Named volumes persist:

- Chroma vector database
- Conversation checkpoints
- Application logs
- Evaluation results
- Hugging Face model cache

Do not use docker compose down -v unless persistent
application data should be deleted.

## Corpus and Vector Store

The provisional V2 corpus and local experimental V2
vector databases are excluded from the Docker image.

The initial deployment uses the existing V1 application
configuration.

The V1 vector database must be initialized or restored
using the project's approved ingestion process before
performing retrieval validation.

Provision approved model weights in the model cache
or permit authorized model downloads when required.

The V2 corpus can be integrated after its academic
scope, reuse permissions, and retrieval evaluation
are finalized.

## Validation Checklist

- Docker image builds successfully.
- Container starts without application errors.
- Health endpoint returns HTTP 200.
- Streamlit UI opens successfully.
- Hosted inference and embedding endpoints are accessible.
- SSL certificate validation succeeds.
- Policy retrieval returns relevant cited evidence.
- Expense calculation works as expected.
- Conversation state persists across container restarts.
- No API credentials are embedded in the image.

## Current Validation Status

Deployment configuration is implemented.

Docker build and runtime validation are pending on a
Docker-enabled machine.

The development workstation does not currently expose
the Docker CLI.

Container runtime validation must be completed before
claiming successful deployment.
