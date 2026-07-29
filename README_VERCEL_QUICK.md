# Vercel deployment notes (quick)

To deploy this repo to Vercel and use OpenRouter for AI text/chat:

1) Connect your GitHub repo to Vercel (Import Project).
2) In Vercel project settings -> Environment Variables, add:
   - OPENROUTER_API_KEY = <your OpenRouter API key>

3) The project includes two serverless endpoints under /api/openrouter:
   - POST /api/openrouter/chat  -> proxy to OpenRouter chat completions (accepts { messages: [...] })
   - GET  /api/openrouter/health -> shows cached free models fetched from OpenRouter

4) In your frontend, call /api/openrouter/chat with the messages payload.

Notes:
- Only OPENROUTER_API_KEY is required for this integration; default rotation/TTL/retries are built-in.
- Video processing (FFmpeg, workers) cannot run on Vercel; run heavy jobs locally or on a separate always-on host when needed.
