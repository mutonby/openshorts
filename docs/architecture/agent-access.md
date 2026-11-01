# API endpoints and agent access (MCP, API keys, OAuth, webhooks)

## Main endpoints

- `POST /api/process`: submit a video for processing. Optional `source_start` /
  `source_end` (seconds) clip and bill only that stretch.
- `GET /api/status/{job_id}`: poll job status and logs.
- `POST /api/edit`: apply AI video effects.
- `POST /api/subtitle`: generate and apply subtitles (auto-transcribes dubbed videos).
- `POST /api/hook`: add text hook overlays.
- `POST /api/translate`: AI voice dubbing via ElevenLabs.
- `GET /api/translate/languages`: supported dubbing languages.
- `POST /api/social/post`: post to social media (async upload). Cloud posts
  through one managed Upload-Post account, one `os_<hex>` profile per user,
  created only by `/api/social/connect` (reads never create it). Capped at
  `SOCIAL_POSTS_PER_HOUR` per user and `SOCIAL_POSTS_PER_DAY_FREE` on the free
  plan (429 with the reason); thumbnail and SaaSShorts publishing count too.
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
job ownership apply to agents with zero endpoint changes. Key management, OAuth
consent, account erasure and billing (cancel, resume, retention offer, end
trial, Stripe portal, invoices) refuse API-key auth
(`cloud/auth.get_session_user_required`): a leaked key cannot mint
replacements nor touch money.

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
/api/oauth/authorize` (session auth, never a key) mints the code, and Cancel asks
`POST /api/oauth/deny` for the redirect: the consent page's own URL is whatever
the link said, so it never navigates to a `redirect_uri` the API did not check
against the client's registered ones. `POST /oauth/token` redeems
it by **minting an ordinary `osk_` key** named after the client and returning it
as the access token. No new auth path: the key shows up in Account → API keys
and revoking it disconnects the app. The key never expires, but a refresh token
(`osr_`, table `oauth_refresh_tokens`, hashed) comes with it for clients that
want one (Gemini connected apps require "persistent linking"): `grant_type=
refresh_token` swaps the key's secret in place and returns the same refresh
token, and a revoked key refuses it. `POST /oauth/revoke` (RFC 7009, always 200)
revokes the OAuth key behind an access or refresh token; hand-made keys are
never revocable there. The `/mcp` 401 carries
`WWW-Authenticate: Bearer resource_metadata=...` so clients find the flow.
The key carries the granted scope (`api_keys.scope`, `cloud/scopes.py`). The only
scope is `openshorts`, the default every client asks for (unknown scopes are
dropped, not refused); it reaches `/mcp`, the endpoints the MCP tools call
in-process and the download paths, and anything else is a 403 naming the
scope. OAuth keys minted before the column are recognised by their name
suffix. Hand-made keys have no scope and keep the whole API. A new MCP tool
calling a new endpoint must add it to `scopes._MCP_ROUTES` (a test reads the
tool code and fails otherwise).
`oauth_codes` and `oauth_refresh_tokens` are in `USER_OWNED_TABLES`; `oauth_clients` deliberately not.
Registration is capped per client IP (`OAUTH_REGISTRATIONS_PER_HOUR`, 429
`slow_down`): a client registers once per connection, the cap only stops one
stuck in a loop. Agent upload slots (`POST /api/uploads`) are capped at
`MAX_PENDING_UPLOADS_PER_USER` open at once.

## Webhooks

`POST /api/process` takes `webhook_url` + optional `webhook_secret`
(HMAC-SHA256, `X-OpenShorts-Signature`). Validated with
`security_utils.assert_public_url` at submit AND at delivery (DNS rebinding).
Fired once per job from `run_job_wrapper` after the R2 archive so the payload can
carry durable download links; survives redeploys via the resume manifest.
`PUBLIC_API_URL` sets the absolute-URL base when behind a proxy.

## Media links (`/videos`, `/thumbnails`; `media_auth.py`, `app._authorize_media`)

Cloud mode serves a job's files only to a request that has a path signature
(`?exp=&sig=`, 24 h, `MEDIA_URL_TTL_SECONDS`), the owner's media token (`?mt=`
or the `os_media` HttpOnly cookie, both from `POST /api/media-token`, 12 h) or
the owner's own `Authorization` / `X-API-Key`. Owner: live job record, the
`.owner` sidecar (clip, SaaSShorts and thumbnail directories all write one),
then the `projects` row. Everything handed outside the dashboard is signed:
MCP `video_url` / `download_url`, webhook `video_url`, and a per-clip
`download_url` in `GET /api/status`. The dashboard appends `mt` in
`getApiUrl`. Bare links to jobs created before `MEDIA_LEGACY_CUTOFF` keep
working until `MEDIA_LEGACY_UNTIL` (links sent before signing existed).
Self-host is open. The R2 library links are separate: public objects behind an
unguessable key.

Jobs sent with an `osk_` key (REST, MCP, OAuth clients) do not trigger the
clips-ready email: the caller polls or gets the webhook. The job records
`via_api_key` and the resume manifest carries it.

## Payment-required errors

Every 402 (`quota_exceeded`, `no_plan`) goes through `app.payment_required`: it adds `upgrade_url` (`/#/pricing?plan=starter&src=api`, which opens the Starter checkout) and repeats the link inside `message`, since MCP clients and SDKs often show only the message. The MCP `get_quota` tool returns the same `upgrade_url` for free or unentitled users, and the server instructions tell the agent to hand that link to the user.

