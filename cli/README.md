# OpenShorts CLI

Clip long videos into vertical 9:16 shorts from the terminal. Zero
dependencies; talks to the same API the dashboard, the MCP server and the
webhooks use.

```bash
pip install openshorts        # or: uvx openshorts / pipx run openshorts

export OPENSHORTS_API_KEY=osk_...   # from your account page at openshorts.app

openshorts process "https://youtube.com/watch?v=..." --wait
openshorts clips <job_id>
openshorts publish <job_id> 0 --platforms tiktok,youtube
openshorts quota
```

Self-hosted instance? Point it at your own machine and skip the key:

```bash
export OPENSHORTS_API_URL=http://localhost:8000
openshorts process "https://youtube.com/watch?v=..." --wait
```

For pipelines, prefer the webhook to `--wait`: pass `--webhook` and
`--webhook-secret` and OpenShorts POSTs once (HMAC-signed,
`X-OpenShorts-Signature: sha256=<hex>`) when the job ends, with clip titles
and durable download links.

Use the global `--json` option before the command for machine-readable output:

```bash
openshorts --json process "https://youtube.com/watch?v=..." --wait > result.json
openshorts --json status <job_id> --watch > result.json
```

While waiting, progress messages go to stderr and stdout contains only the
completed result as JSON. Failed jobs exit with code 1 without a JSON result.

The hosted free tier is 20 minutes/month with a watermark; paid plans from
$12/month. The self-hosted edition is MIT and has no meter. Agent-native
version of the same surface: [openshorts.app/mcp](https://www.openshorts.app/mcp).
