from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime


class LearningProfile(BaseModel):
    profile_id: str
    name: str
    base_category: str
    description: str = ""
    created_at: datetime = None
    clip_count: int = 0
    avg_engagement: float = 0.0
    is_preset: bool = False

    def __init__(self, **data):
        if "created_at" not in data or data["created_at"] is None:
            data["created_at"] = datetime.utcnow()
        super().__init__(**data)


class ClipFeedback(BaseModel):
    clip_id: str
    profile_id: str
    job_id: str
    clip_index: int

    views: int = 0
    likes: int = 0
    shares: int = 0
    watch_time_seconds: float = 0.0
    comments: List[str] = []
    platform: str = "youtube"
    posted_at: Optional[datetime] = None

    transcript_segment: str = ""
    hook_text: str = ""
    viral_pattern: str = ""
    ai_score: float = 0.0
    clip_duration: float = 0.0
    language: str = "id"
    scene_count: int = 0
    key_visuals: List[str] = []

    engagement_rate: float = 0.0
    retention_rate: float = 0.0
    is_success: bool = False

    created_at: datetime = None

    def __init__(self, **data):
        if "created_at" not in data or data["created_at"] is None:
            data["created_at"] = datetime.utcnow()
        if "engagement_rate" not in data or data["engagement_rate"] == 0.0:
            data["engagement_rate"] = (
                (data.get("likes", 0) + data.get("shares", 0)) / max(data.get("views", 1), 1)
            )
        if "retention_rate" not in data or data["retention_rate"] == 0.0:
            dur = data.get("clip_duration", 1)
            data["retention_rate"] = data.get("watch_time_seconds", 0) / max(dur, 1)
        if "is_success" not in data:
            from learning.config import RAG_SUCCESS_THRESHOLD
            data["is_success"] = data.get("engagement_rate", 0) >= RAG_SUCCESS_THRESHOLD
        super().__init__(**data)


class AutoAnalysisResult(BaseModel):
    transcript_segment: str
    hook_text: str
    viral_pattern: str
    ai_score: float
    clip_duration: float
    language: str
    scene_count: int
    key_visuals: List[str]


class UploadFeedbackResponse(BaseModel):
    clip_id: str
    status: str
    auto_analysis: Optional[AutoAnalysisResult] = None
