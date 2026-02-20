# Quick Resume Deployment Guide

Use this when Docker services are mostly up, but you want the ML backend on the host (not in Docker) for better performance.

## 1) Go to project root
```bash
cd /Users/antoine/bird_leg
```

## 2) Make sure env file exists
```bash
test -f .env || cp .env.example .env
```

## 3) Start or refresh Docker services
```bash
make compose-up
```

## 4) Stop ML backend container (host mode only)
```bash
make stop-ml-backend-container
```

## 5) Run ML backend on host
Keep this terminal open.
```bash
MODEL_A_DEVICE=auto make run-ml-backend-host
```

This starts the backend on `http://127.0.0.1:9091`.

## 6) Point Label Studio to host backend
In Label Studio Machine Learning settings, use:
`http://host.docker.internal:9091`

## Quick checks
```bash
make compose-ps
curl -sS http://127.0.0.1:9091/health
```

Expected health response includes `"status":"UP"` and `"model_a_loaded":true`.

## If port 9091 is already in use
```bash
lsof -nP -iTCP:9091 -sTCP:LISTEN
kill <PID>
```

## Full restart sequence (when things look inconsistent)
```bash
make compose-down
make compose-up
make stop-ml-backend-container
MODEL_A_DEVICE=auto make run-ml-backend-host
```

## Optional after dependency changes
```bash
make bootstrap
```
