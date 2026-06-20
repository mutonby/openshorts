import os
import uuid
import json
import shutil
import asyncio
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends

from learning.models import (
    LearningProfile,
    ClipFeedback,
    AutoAnalysisResult,
    UploadFeedbackResponse,
)
from learning.vector_store import VectorStore
from learning.embedder import LocalEmbedder
from learning.profile_manager import ProfileManager
from learning.rag_injector import RAGInjector
from learning.example_builder import FewShotBuilder
from learning.config import LEARNING_ENABLED

router = APIRouter(prefix="/api/learning", tags=["learning"])

_learning_dir = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "uploads"
)
os.makedirs(_learning_dir, exist_ok=True)

_store: VectorStore = None
_embedder: LocalEmbedder = None
_profiles: ProfileManager = None
_builder: FewShotBuilder = None
_rag: RAGInjector = None


def init_learning(app_state: dict = None):
    global _store, _embedder, _profiles, _builder, _rag

    if not LEARNING_ENABLED:
        print("  [Learning] System disabled (LEARNING_ENABLED=false)")
        return

    _store = VectorStore()
    _embedder = LocalEmbedder()
    _profiles = ProfileManager(_store)
    _builder = FewShotBuilder()
    _rag = RAGInjector(_store, _embedder, _builder)

    print(f"  [Learning] System initialized ({len(_profiles.list_profiles())} profiles)")


def get_rag_injector() -> Optional[RAGInjector]:
    return _rag


def get_profile_manager() -> Optional[ProfileManager]:
    return _profiles


# ====== Profile Endpoints ======


@router.get("/profiles")
async def list_profiles():
    if not _profiles:
        raise HTTPException(503, "Learning system not initialized")
    return [p.model_dump(mode="json") for p in _profiles.list_profiles()]


@router.post("/profiles", status_code=201)
async def create_profile(name: str = Form(...), base_category: str = Form(...), description: str = Form("")):
    if not _profiles:
        raise HTTPException(503, "Learning system not initialized")
    profile = _profiles.create_profile(name, base_category, description)
    return profile.model_dump(mode="json")


@router.get("/profiles/{profile_id}")
async def get_profile(profile_id: str):
    if not _profiles:
        raise HTTPException(503, "Learning system not initialized")
    profile = _profiles.get_profile(profile_id)
    if not profile:
        raise HTTPException(404, "Profile not found")
    return profile.model_dump(mode="json")


@router.put("/profiles/{profile_id}")
async def update_profile(
    profile_id: str,
    name: Optional[str] = Form(None),
    base_category: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
):
    if not _profiles:
        raise HTTPException(503, "Learning system not initialized")
    updates = {}
    if name is not None:
        updates["name"] = name
    if base_category is not None:
        updates["base_category"] = base_category
    if description is not None:
        updates["description"] = description
    profile = _profiles.update_profile(profile_id, updates)
    if not profile:
        raise HTTPException(404, "Profile not found")
    return profile.model_dump(mode="json")


@router.delete("/profiles/{profile_id}")
async def delete_profile(profile_id: str, delete_clips: bool = True):
    if not _profiles:
        raise HTTPException(503, "Learning system not initialized")
    _profiles.delete_profile(profile_id, delete_clips=delete_clips)
    return {"status": "deleted", "profile_id": profile_id}


# ====== Feedback Endpoints ======


@router.post("/feedback", status_code=201)
async def upload_feedback(
    video: UploadFile = File(...),
    profile_id: str = Form(...),
    platform: str = Form("youtube"),
    views: int = Form(0),
    likes: int = Form(0),
    shares: int = Form(0),
    watch_time_seconds: float = Form(0.0),
    comments: str = Form(""),
):
    if not _profiles or not _store or not _embedder:
        raise HTTPException(503, "Learning system not initialized")

    profile = _profiles.get_profile(profile_id)
    if not profile:
        raise HTTPException(404, "Profile not found")

    clip_id = str(uuid.uuid4())

    # Save video file
    ext = os.path.splitext(video.filename or "clip.mp4")[1] or ".mp4"
    video_path = os.path.join(_learning_dir, f"{clip_id}{ext}")
    with open(video_path, "wb") as f:
        content = await video.read()
        f.write(content)

    comments_list = [c.strip() for c in comments.split("\n") if c.strip()]

    clip = ClipFeedback(
        clip_id=clip_id,
        profile_id=profile_id,
        job_id="",
        clip_index=0,
        views=views,
        likes=likes,
        shares=shares,
        watch_time_seconds=watch_time_seconds,
        comments=comments_list,
        platform=platform,
    )

    # Store the clip data for auto-analysis status
    _pending_feedback[clip_id] = {
        "clip": clip,
        "video_path": video_path,
        "status": "analyzing",
    }

    # Schedule auto-analysis in background
    asyncio.create_task(_run_auto_analysis(clip_id))

    return UploadFeedbackResponse(
        clip_id=clip_id,
        status="analyzing",
    )


@router.get("/feedback/{clip_id}")
async def get_feedback_status(clip_id: str):
    if clip_id not in _pending_feedback:
        raise HTTPException(404, "Feedback not found")
    entry = _pending_feedback[clip_id]
    return {
        "clip_id": clip_id,
        "status": entry["status"],
        "auto_analysis": entry.get("auto_analysis"),
    }


@router.post("/feedback/{clip_id}/confirm")
async def confirm_feedback(
    clip_id: str,
    transcript_segment: Optional[str] = Form(None),
    hook_text: Optional[str] = Form(None),
    viral_pattern: Optional[str] = Form(None),
    ai_score: Optional[float] = Form(None),
    clip_duration: Optional[float] = Form(None),
    scene_count: Optional[int] = Form(None),
    key_visuals: Optional[str] = Form(""),
):
    if not _store or not _embedder:
        raise HTTPException(503, "Learning system not initialized")

    entry = _pending_feedback.get(clip_id)
    if not entry:
        raise HTTPException(404, "Feedback not found")

    clip: ClipFeedback = entry["clip"]

    if transcript_segment is not None:
        clip.transcript_segment = transcript_segment
    if hook_text is not None:
        clip.hook_text = hook_text
    if viral_pattern is not None:
        clip.viral_pattern = viral_pattern
    if ai_score is not None:
        clip.ai_score = ai_score
    if clip_duration is not None:
        clip.clip_duration = clip_duration
    if scene_count is not None:
        clip.scene_count = scene_count
    if key_visuals:
        clip.key_visuals = [v.strip() for v in key_visuals.split(",") if v.strip()]

    text_to_embed = (
        f"Hook: {clip.hook_text}\n"
        f"Pattern: {clip.viral_pattern}\n"
        f"Transcript: {clip.transcript_segment[:1500]}\n"
        f"Comments: {' '.join(clip.comments[:5])}"
    )

    embedding = _embedder.embed_query(text_to_embed)

    metadata = {
        "profile_id": clip.profile_id,
        "base_category": _profiles.get_profile(clip.profile_id).base_category if _profiles else "",
        "hook_text": clip.hook_text,
        "viral_pattern": clip.viral_pattern,
        "ai_score": clip.ai_score,
        "clip_duration": clip.clip_duration,
        "views": clip.views,
        "likes": clip.likes,
        "shares": clip.shares,
        "watch_time_seconds": clip.watch_time_seconds,
        "engagement_rate": clip.engagement_rate,
        "retention_rate": clip.retention_rate,
        "is_success": clip.is_success,
        "platform": clip.platform,
        "language": clip.language,
        "scene_count": clip.scene_count,
        "transcript_segment": clip.transcript_segment[:500],
    }

    _store.add_clip(clip_id, embedding, metadata)
    entry["status"] = "stored"
    entry["auto_analysis"] = None

    return {
        "clip_id": clip_id,
        "status": "stored",
        "message": "Clip added to learning pool. It will influence future generations.",
    }


@router.delete("/feedback/{clip_id}")
async def delete_feedback(clip_id: str):
    if _store:
        _store.delete_clip(clip_id)
    _pending_feedback.pop(clip_id, None)

    # Clean up video file
    for ext in [".mp4", ".webm", ".mov"]:
        path = os.path.join(_learning_dir, f"{clip_id}{ext}")
        if os.path.exists(path):
            os.remove(path)

    return {"status": "deleted", "clip_id": clip_id}


# ====== Stats Endpoints ======


@router.get("/stats")
async def get_global_stats():
    if not _profiles or not _store:
        raise HTTPException(503, "Learning system not initialized")
    profiles = _profiles.list_profiles()
    total_clips = sum(p.clip_count for p in profiles)
    avg_engagement = (
        sum(p.avg_engagement for p in profiles) / len(profiles) if profiles else 0
    )

    return {
        "total_profiles": len(profiles),
        "total_clips": total_clips,
        "avg_engagement": round(avg_engagement, 4),
        "profiles": [
            {
                "profile_id": p.profile_id,
                "name": p.name,
                "category": p.base_category,
                "clip_count": p.clip_count,
                "avg_engagement": p.avg_engagement,
            }
            for p in profiles
        ],
    }


@router.get("/profiles/{profile_id}/stats")
async def get_profile_stats(profile_id: str):
    if not _store:
        raise HTTPException(503, "Learning system not initialized")
    stats = _store.get_profile_stats(profile_id)
    return stats


# ====== Internal State ======

_pending_feedback: dict = {}


async def _run_auto_analysis(clip_id: str):
    """
    Background task: auto-extract features from uploaded clip.
    For now, marks the clip as ready for manual confirmation.
    Full auto-analysis (transcription, visual extraction) will be added in Phase 2.
    """
    entry = _pending_feedback.get(clip_id)
    if not entry:
        return

    try:
        video_path = entry["video_path"]
        clip: ClipFeedback = entry["clip"]

        # Extract basic info from filename
        clip.clip_duration = 0.0

        # Placeholder for future auto-analysis:
        # 1. Transcribe with faster-whisper
        # 2. Detect scenes with PySceneDetect
        # 3. Extract visual features with YOLO

        entry["auto_analysis"] = AutoAnalysisResult(
            transcript_segment="",
            hook_text="",
            viral_pattern="",
            ai_score=0.0,
            clip_duration=clip.clip_duration,
            language=clip.language,
            scene_count=0,
            key_visuals=[],
        )
        entry["status"] = "ready_for_confirmation"
        print(f"  [Learning] Auto-analysis complete for {clip_id}")

    except Exception as e:
        print(f"  [Learning] Auto-analysis failed for {clip_id}: {e}")
        entry["status"] = "awaiting_manual_input"
