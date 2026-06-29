# AIGText Runtime Hosts

Use the shell host launchers for split deployment from Git Bash or a Linux-style shell. They set `PYTHONPATH` to the repository root, honor `$PYTHON` when set, and pass CLI arguments through to the backward-compatible Python wrappers; tests use `--help` and parser checks so no model downloads or production vectorstores are touched.

## Host A

Host A serves the phone-facing frontend and language proxy. Default bind is `0.0.0.0:8080`.

```bash
bash scripts/run_host_a.sh --host 0.0.0.0 --port 8080 --vision-backend-url http://HOST_B_IP:9101
```

Environment overrides: `AIGTEXT_ROLE`, `HOST_A_BIND`, `HOST_A_PORT`, `VISION_BACKEND_URL`, `VISION_BACKEND_API_KEY`, and `VISION_BACKEND_TIMEOUT_SECONDS`.

## Host B

Host B serves the vision endpoint. Default bind is `0.0.0.0:9101`; `VISION_BACKEND_MODE=stub` is never enabled unless explicitly set by env or `--backend-mode stub`.

```bash
bash scripts/run_host_b.sh --host 0.0.0.0 --port 9101
```

Environment overrides: `AIGTEXT_ROLE`, `VISION_BIND`, `VISION_PORT`, `VISION_BACKEND_MODE`, `VISION_FALLBACK_MODE`, `VISION_API_KEY`, and `VISUAL_TOP_K_MAX`. The safe fallback default is `VISION_FALLBACK_MODE=error`.
