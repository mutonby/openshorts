# API endpoints and agent access (MCP, API keys, OAuth, webhooks)

## Main endpoints

- `POST /api/process`: submit a video for processing.
- `GET /api/status/{job_id}`: poll job status and logs.
- `POST /api/edit`: apply AI video effects.
- `POST /api/subtitle`: generate and apply subtitles (auto-transcribes dubbed videos).
- `POST /api/hook`: add text hook overlays.
- `POST /api/translate`: AI voice dubbing via ElevenLabs.
- `GET /api/translate/languages`: supported dubbing languages.
- `POST /api/social/post`: post to social media (async upload).
- `POST /mcp`: MCP server (JSON-RPC), the pipeline as agent tools.
- `POST/GET/DELETE /api/keys`: user API keys (cloud mode, session JWT only).
- `DELETE /api/account`: erase the account and everything in it (GDPR art. 17).
- `GET/PUT /api/autopilot`: Autopilot settings, connected accounts, recent runs (cloud only).
- `GET /api/autopilot/videos`: latest uploads of the connected YouTube channel + what Autopilot did.
- `POST /api/autopilot/run`: clip one channel video now (`{video_id}`).

## API keys (`cloud/api_keys.py`)

`osk_...` tokens, sha256-stored, created in the dashboard account page.
`cloud/auth.get_current_user_optional` accepts them (`Bearer osk_...` or
`X-API-Key`) and resolves the owner, so metering, entitlement, plan priority and
job ownership apply to agents with zero endpoint changes. Key management itself
refuses API-key auth: a leaked key cannot mint replacements.

## MCP server (`mcp_server.py`, always mounted)

Stateless Streamable-HTTP JSON-RPC at `/mcp`, no SDK dependency, ~3 methods + 8
tools. Each tool calls back into this same app in-process
(`httpx.ASGITransport`) forwarding the caller's auth headers, so it can never
drift from the REST behaviour. Cloud mode 401s without a resolvable user;
self-host stays BYOK-open.

## stdio transport (`mcp_stdio.py`)

The same `handle_message` / `call_tool` as a subprocess, for hosts that only
launch MCP servers as a command (e.g. Glama's Dockerfile deployments; a local
client can skip the web server). Two invariants:

- `sys.stdout` is swapped for stderr **before `app` is imported**: the pipeline
  prints everywhere and one stray line corrupts the JSON-RPC stream.
- The app's lifespan is entered (`router.lifespan_context`), which
  `ASGITransport` does not do on its own.

## OAuth for MCP clients (`cloud/mcp_oauth.py`, cloud mode only)

claude.ai and ChatGPT connect by URL, so the server publishes RFC 9728/8414
metadata under `/.well-known/`, accepts dynamic client registration
(`POST /oauth/register`, public clients, PKCE S256 mandatory) and bounces
`GET /oauth/authorize` to the dashboard consent screen (`#/oauth/authorize`),
because the session JWT lives in localStorage on the frontend host. `POST
/api/oauth/authorize` (session auth) mints the code; `POST /oauth/token` redeems
it by **minting an ordinary `osk_` key** named after the client and returning it
as the access token. No new auth path, no refresh tokens: the key shows up in
Account → API keys and revoking it disconnects the app. The `/mcp` 401 carries
`WWW-Authenticate: Bearer resource_metadata=...` so clients find the flow.
`oauth_codes` is in `USER_OWNED_TABLES`; `oauth_clients` deliberately not.

## Webhooks

`POST /api/process` takes `webhook_url` + optional `webhook_secret`
(HMAC-SHA256, `X-OpenShorts-Signature`). Validated with
`security_utils.assert_public_url` at submit AND at delivery (DNS rebinding).
Fired once per job from `run_job_wrapper` after the R2 archive so the payload can
carry durable download links; survives redeploys via the resume manifest.
`PUBLIC_API_URL` sets the absolute-URL base when behind a proxy.
