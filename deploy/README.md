# ORBIT Production Deployment Guide

Target architecture (single Linux host):

```
Internet ──HTTPS──▶ Caddy :443          (only internet-facing process)
                    ├── /               → /srv/orbit/orbit-frontend/dist  (SPA, try_files)
                    └── /api/* (strip)  → 127.0.0.1:8000  (uvicorn, 1 worker)
                                              └── ThreadPool(4) → AI/Data/ML/Research
```

Constraints honored: backend loopback-only · exactly one worker (in-memory
task registry) · no Docker/Redis/SQLite/auth/Sentry · secrets only in
`/etc/orbit/orbit.env` · nothing in this directory contains real values.

## 1. Prerequisites

- Ubuntu 22.04/24.04 (or any systemd distro) with a non-root sudo user
- DNS A/AAAA record for your domain pointing at the host; ports 80/443 open
- Packages: `curl git build-essential python3.12 python3.12-venv nodejs npm caddy`
  (install Node 20+ and Caddy from upstream repos — see §6)

## 2. Directory layout

```
/srv/orbit/                  ← git clone of this repo (owned by orbit)
├── orbit-backend/           ← FastAPI app; .venv lives here; writes .data/
└── orbit-frontend/          ← dist/ built here; served by Caddy

/etc/orbit/orbit.env         ← secrets (root:orbit, 0640)
/etc/caddy/Caddyfile         ← copy of deploy/Caddyfile
/etc/systemd/system/orbit.service  ← copy of deploy/orbit.service
```

## 3. Create user, clone, build frontend

```bash
sudo useradd --system --create-home --home /srv/orbit --shell /usr/sbin/nologin orbit
sudo -iu orbit git clone <YOUR_REPO_URL> /srv/orbit
cd /srv/orbit/orbit-frontend
npm ci
npm run build          # produces dist/
```

## 4. Backend virtual environment

```bash
cd /srv/orbit/orbit-backend
python3.12 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt   # pinned to tested versions
```

## 5. Environment / secrets

```bash
sudo install -d -o root -g orbit -m 750 /etc/orbit
sudo install -o root -g orbit -m 640 deploy/orbit.env.example /etc/orbit/orbit.env
sudoedit /etc/orbit/orbit.env     # fill FREEBUFF_API_KEY, CORS_ORIGINS, DOMAIN values
```

Set `CORS_ORIGINS=https://YOUR_DOMAIN` and never commit a filled-in copy.

## 6. systemd (backend)

```bash
sudo cp /srv/orbit/deploy/orbit.service /etc/systemd/system/orbit.service
sudo systemctl daemon-reload
sudo systemctl enable --now orbit
systemctl status orbit                # expect active (running)
sudo journalctl -u orbit -f           # live logs
```

## 7. Caddy (proxy + TLS)

```bash
sudo cp /srv/orbit/deploy/Caddyfile /etc/caddy/Caddyfile
sudoedit /etc/caddy/Caddyfile         # optional: hardcode your domain instead of {$DOMAIN}
sudo systemctl reload caddy           # or: systemctl enable --now caddy on first install
```

Caddy obtains/renews certificates automatically once DNS resolves to the host
and ports 80/443 are reachable. First issuance happens on startup.

## 8. Health check & startup/restart procedures

```bash
curl -s https://YOUR_DOMAIN/api/health | jq    # expect {"ok":true,...}
sudo systemctl restart orbit                   # backend restart (Runs list resets)
sudo systemctl reload caddy                    # config-only proxy reload
```

## 9. Rollback

```bash
cd /srv/orbit && sudo -iu orbit git fetch && sudo -iu orbit git checkout <LAST_GOOD_TAG>
cd orbit-frontend && npm ci && npm run build
cd ../orbit-backend && .venv/bin/pip install -r requirements.txt
sudo systemctl restart orbit
# Caddy rollback:
sudo cp /etc/caddy/Caddyfile.bak /etc/caddy/Caddyfile && sudo systemctl reload caddy
```

Tip: `sudo cp /etc/caddy/Caddyfile /etc/caddy/Caddyfile.bak` before edits.

## 10. Log inspection

```bash
sudo journalctl -u orbit -n 200          # backend (app + access logs)
sudo journalctl -u caddy -n 200          # proxy
sudo journalctl -u orbit | grep -i error
```

Backend logs always include the `task_id` for correlation. No secrets are
logged (audited); `journalctl` output is root-readable only by default.

## 11. Verification

```bash
BASE_URL=https://YOUR_DOMAIN ./deploy/verify.sh
```

Covers: /api/health · frontend + SPA fallback · docs blocked · HTTPS/HSTS ·
AI task end-to-end · unknown task/agent · invalid + oversized CSV · Data CSV
task · CORS allow/deny. Run the Research and ML tasks via the UI once
(manual checklist items — both verified pre-release in development).

Manual checklist:

- [ ] AI task completes in the Workspace
- [ ] Research task returns verified sources
- [ ] Data CSV task renders statistics
- [ ] ML CSV task renders model comparison (needs ≥20 rows)
- [ ] Invalid CSV → inline error; oversized → clean 413/422
- [ ] Runs tab lists the session's tasks
