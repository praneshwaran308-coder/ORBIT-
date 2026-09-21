#!/usr/bin/env bash
# ORBIT production verification — run AFTER deployment, against the live URL.
# Usage: BASE_URL=https://orbit.example.com ./verify.sh
# Exit code 0 = all checks passed. No secrets are printed.
set -u

BASE="${BASE_URL:-https://orbit.example.com}"
PASS=0; FAIL=0

ok()   { echo "  PASS  $1"; PASS=$((PASS+1)); }
bad()  { echo "  FAIL  $1"; FAIL=$((FAIL+1)); }
note() { echo "  ----  $1"; }

echo "== ORBIT production verification: $BASE =="

# 1. Health through the proxy
H=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "$BASE/api/health")
[ "$H" = "200" ] && ok "/api/health -> 200" || bad "/api/health -> $H (want 200)"

# 2. Frontend loads (SPA shell)
H=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "$BASE/")
[ "$H" = "200" ] && ok "frontend / -> 200" || bad "frontend / -> $H (want 200)"
H=$(curl -s --max-time 10 "$BASE/" | grep -c '<div id="root">')
[ "$H" -ge 1 ] && ok "SPA root element present" || bad "SPA root element missing"

# 3. SPA fallback for deep links
H=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "$BASE/some/deep/link")
[ "$H" = "200" ] && ok "SPA fallback /some/deep/link -> 200" || bad "SPA fallback -> $H"

# 4. Docs endpoints blocked
for p in /api/docs /api/redoc /api/openapi.json; do
  H=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "$BASE$p")
  [ "$H" = "404" ] && ok "$p blocked (404)" || bad "$p -> $H (want 404)"
done

# 5. HTTPS + HSTS
SCHEME=$(printf '%s' "$BASE" | cut -d: -f1)
[ "$SCHEME" = "https" ] && ok "served over HTTPS" || bad "BASE_URL is not https"
H=$(curl -s -D - -o /dev/null --max-time 10 "$BASE/" | grep -ci strict-transport-security)
[ "$H" -ge 1 ] && ok "HSTS header present" || bad "HSTS header missing"

# 6. AI task (requires a working provider credential)
RUN=$(curl -s --max-time 10 -X POST "$BASE/api/run" \
  -H 'Content-Type: application/json' \
  -d '{"task":"Explain the difference between TCP and UDP"}')
TID=$(printf '%s' "$RUN" | sed -n 's/.*"task_id":"\([a-f0-9]*\)".*/\1/p')
AGENT=$(printf '%s' "$RUN" | sed -n 's/.*"agent":"\([a-z]*\)".*/\1/p')
if [ -n "$TID" ] && [ "$AGENT" = "ai" ]; then
  ok "AI task accepted (task ${TID:0:8}…, agent=ai)"
  for _ in 1 2 3 4 5 6 7 8 9 10 11 12; do
    sleep 3
    ST=$(curl -s --max-time 10 "$BASE/api/status/$TID" | sed -n 's/.*"status":"\([a-z_]*\)".*/\1/p' | head -1)
    case "$ST" in completed|failed|partial|insufficient_data) break;; esac
  done
  [ "$ST" = "completed" ] && ok "AI task completed" || bad "AI task status: $ST"
else
  bad "AI task accept failed: $(printf '%s' "$RUN" | head -c 120)"
fi

# 7. Unknown task id -> 404
H=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "$BASE/api/status/does-not-exist")
[ "$H" = "404" ] && ok "unknown task id -> 404" || bad "unknown task id -> $H"

# 8. Unknown agent -> 422
H=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 -X POST "$BASE/api/run" \
  -H 'Content-Type: application/json' -d '{"task":"hi","agent":"hacker"}')
[ "$H" = "422" ] && ok "unknown agent -> 422" || bad "unknown agent -> $H"

# 9. Invalid CSV -> 422
H=$(curl -s -o /dev/null -w '%{http_code}' --max-time 15 -X POST "$BASE/api/run/file" \
  -F 'task=analyze this' -F 'agent=data' \
  -F 'file=@/dev/null;filename=x.txt;type=text/plain' 2>/dev/null)
[ "$H" = "422" ] && ok "invalid CSV upload -> 422" || bad "invalid CSV upload -> $H"

# 10. Oversized upload -> 413 (Caddy body limit) or 422 (app limit)
dd if=/dev/zero of=/tmp/orbit_oversize.csv bs=1M count=30 2>/dev/null
H=$(curl -s -o /dev/null -w '%{http_code}' --max-time 30 -X POST "$BASE/api/run/file" \
  -F 'task=analyze this' -F 'agent=data' \
  -F 'file=@/tmp/orbit_oversize.csv;type=text/csv')
rm -f /tmp/orbit_oversize.csv
[ "$H" = "413" ] || [ "$H" = "422" ] && ok "oversized upload -> $H (blocked)" || bad "oversized upload -> $H (want 413/422)"

# 11. Data CSV task end-to-end (small valid CSV)
printf 'sqft,price\n1200,185000\n1450,210000\n1600,260000\n1800,275000\n2000,310000\n2200,340000\n' > /tmp/orbit_smoke.csv
RUN=$(curl -s --max-time 15 -X POST "$BASE/api/run/file" \
  -F 'task=Analyze this CSV: statistics, correlations and insights' -F 'agent=data' \
  -F 'file=@/tmp/orbit_smoke.csv;type=text/csv')
TID=$(printf '%s' "$RUN" | sed -n 's/.*"task_id":"\([a-f0-9]*\)".*/\1/p')
rm -f /tmp/orbit_smoke.csv
if [ -n "$TID" ]; then
  for _ in 1 2 3 4 5 6 7 8; do
    sleep 3
    ST=$(curl -s --max-time 10 "$BASE/api/status/$TID" | sed -n 's/.*"status":"\([a-z_]*\)".*/\1/p' | head -1)
    case "$ST" in completed|failed|partial|insufficient_data) break;; esac
  done
  [ "$ST" = "completed" ] && ok "Data CSV task completed" || bad "Data CSV task status: $ST"
else
  bad "Data CSV task accept failed"
fi

note "Research/ML tasks: exercise via the UI or repeat the pattern above"
note "  Research: POST /api/run {\"task\":\"Research the current state of fusion energy with sources\"}"
note "  ML:       POST /api/run/file with a >=20-row CSV and 'Train a model to predict price from this CSV'"

# 12. CORS: allowed origin blessed, foreign origin gets no ACAO
A=$(curl -s -D - -o /dev/null --max-time 10 -H "Origin: $BASE" "$BASE/api/health" \
     | grep -i '^access-control-allow-origin:' | tr -d '\r' | awk '{print $2}')
[ "$A" = "$BASE" ] && ok "CORS: configured origin blessed ($A)" \
                    || note "CORS: no ACAO for $BASE (only wrong if CORS_ORIGINS should include it)"
E=$(curl -s -D - -o /dev/null --max-time 10 -H "Origin: http://evil.example" "$BASE/api/health" \
     | grep -ci '^access-control-allow-origin:')
[ "$E" = "0" ] && ok "CORS: evil origin gets no ACAO" || bad "CORS: evil origin blessed!"

echo "== $PASS passed, $FAIL failed =="
[ "$FAIL" -eq 0 ]
