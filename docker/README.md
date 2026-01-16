## Purpose

Docker build context for inference and API services.
Contents
1) Dockerfile.api – FastAPI prediction server
2) Dockerfile.infer – CLI / batch inference
3) _bundle/ – Model bundle copied during build

Notes
This directory is passed directly to docker build.