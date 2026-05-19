import express from "express";
import { v4 as uuidv4 } from "uuid";
import { z } from "zod";
import { initBundle } from "./bundle.js";
import { executeRender } from "./render-worker.js";

// --- Render status types ---

export type RenderStatus = "queued" | "rendering" | "done" | "error";

export interface RenderJob {
  renderId: string;
  jobId: string;
  clipIndex: number;
  status: RenderStatus;
  progress: number;
  outputUrl?: string;
  error?: string;
}

// In-memory render job map
export const renderJobs = new Map<string, RenderJob>();

// --- Request validation schema ---

const renderRequestSchema = z.object({
  jobId: z.string().min(1),
  clipIndex: z.number().int().min(0),
  props: z.object({
    videoUrl: z.string(),
    durationInFrames: z.number().int().positive(),
    fps: z.number().positive(),
    width: z.number().int().positive(),
    height: z.number().int().positive(),
    subtitles: z.any().nullable().optional(),
    hook: z.any().nullable().optional(),
    effects: z.any().nullable().optional(),
  }),
});

// --- Express app ---

const app = express();
app.use(express.json({ limit: "10mb" }));

const PORT = parseInt(process.env.PORT || "3100", 10);
const OUTPUT_DIR = process.env.OUTPUT_DIR || "/output";

// Serve video files from the shared output volume so Remotion can access them via HTTP
app.use("/output", express.static(OUTPUT_DIR));

// Health check
app.get("/health", (_req, res) => {
  res.json({ ok: true });
});

// Submit a render job
app.post("/render", (req, res) => {
  const parsed = renderRequestSchema.safeParse(req.body);

  if (!parsed.success) {
    res.status(400).json({
      error: "Invalid request body",
      details: parsed.error.issues,
    });
    return;
  }

  const { jobId, clipIndex, props } = parsed.data;
  const renderId = uuidv4();

  const job: RenderJob = {
    renderId,
    jobId,
    clipIndex,
    status: "queued",
    progress: 0,
  };

  renderJobs.set(renderId, job);

  console.log(
    `[render] Queued render ${renderId} for job=${jobId} clip=${clipIndex}`
  );

  // Resolve video URL: convert frontend/backend URLs to renderer's own static server
  // The renderer serves /output/* from the shared Docker volume
  let resolvedVideoUrl = props.videoUrl;
  const videoPathMatch = props.videoUrl.match(/\/videos\/([^/]+)\/(.+)$/);
  if (videoPathMatch) {
    resolvedVideoUrl = `http://localhost:${PORT}/output/${videoPathMatch[1]}/${videoPathMatch[2]}`;
    console.log(`[render] Resolved video URL: ${props.videoUrl} -> ${resolvedVideoUrl}`);
  }

  // Fire and forget - render runs in background
  executeRender({
    renderId,
    jobId,
    clipIndex,
    props: {
      videoUrl: resolvedVideoUrl,
      durationInFrames: props.durationInFrames,
      fps: props.fps,
      width: props.width,
      height: props.height,
      subtitles: props.subtitles ?? null,
      hook: props.hook ?? null,
      effects: props.effects ?? null,
    },
  }).catch((err) => {
    console.error(`[render] Unhandled error for ${renderId}:`, err);
    const existingJob = renderJobs.get(renderId);
    if (existingJob) {
      existingJob.status = "error";
      existingJob.error =
        err instanceof Error ? err.message : "Unknown error";
    }
  });

  res.status(202).json({ renderId, status: "queued" });
});

// Get render status
app.get("/render/:renderId", (req, res) => {
  const { renderId } = req.params;
  const job = renderJobs.get(renderId);

  if (!job) {
    res.status(404).json({ error: "Render not found" });
    return;
  }

  const response: Record<string, unknown> = {
    renderId: job.renderId,
    status: job.status,
  };

  if (job.progress !== undefined) {
    response.progress = job.progress;
  }
  if (job.outputUrl) {
    response.outputUrl = job.outputUrl;
  }
  if (job.error) {
    response.error = job.error;
  }

  res.json(response);
});

// --- Start server ---

async function main() {
  console.log("[render-service] Initializing Remotion bundle...");
  await initBundle();
  console.log("[render-service] Bundle ready.");

  app.listen(PORT, () => {
    console.log(`[render-service] Listening on port ${PORT}`);
  });
}

main().catch((err) => {
  console.error("[render-service] Fatal error during startup:", err);
  process.exit(1);
});
