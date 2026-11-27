"""Per-job AI batch checkpoints. Prompts and credentials stay out of the files."""
import hashlib
import json
import os
from pathlib import Path

from jsonfile import write_json_atomic


class AnalysisUnavailable(RuntimeError):
    pass


class AnalysisBudgetExceeded(RuntimeError):
    pass


def _limit(name):
    try:
        return max(0, int(os.environ.get(name, "0")))
    except ValueError:
        return 0


class AnalysisCheckpoint:
    def __init__(self, directory, provider, *, enabled=True):
        self.directory = Path(directory, ".analysis") if directory and enabled else None
        self.provider = provider
        self.requests = 0
        self.prompt_bytes = 0
        self.reused = 0
        self.max_requests = _limit("HIGHLIGHT_MAX_AI_REQUESTS")
        self.max_prompt_bytes = _limit("HIGHLIGHT_MAX_PROMPT_BYTES")

    def charge(self, prompt):
        size = len(prompt.encode("utf-8"))
        if self.max_requests and self.requests >= self.max_requests:
            raise AnalysisBudgetExceeded("AI request limit reached. Completed batches were saved.")
        if self.max_prompt_bytes and self.prompt_bytes + size > self.max_prompt_bytes:
            raise AnalysisBudgetExceeded("AI prompt budget reached. Completed batches were saved.")
        self.requests += 1
        self.prompt_bytes += size

    def batch(self, model, schema, prompt, submit):
        if self.directory is None:
            return submit()
        spec = json.dumps([self.provider, model, schema.model_json_schema(), prompt],
                          sort_keys=True, ensure_ascii=False)
        digest = hashlib.sha256(spec.encode("utf-8")).hexdigest()
        path = self.directory / f"{digest}.json"
        try:
            with path.open() as handle:
                saved = json.load(handle)
            result = schema.model_validate(saved["result"]).model_dump()
            cost = saved.get("cost")
            if cost is not None and not isinstance(cost, dict):
                raise ValueError("invalid cached cost")
        except (OSError, ValueError, KeyError, TypeError):
            pass
        else:
            self.reused += 1
            return result, cost
        result, cost = submit()
        result = schema.model_validate(result).model_dump()
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        write_json_atomic(path, {"result": result, "cost": cost})
        return result, cost

    def report(self):
        return {"requests": self.requests, "reused_batches": self.reused,
                "prompt_bytes": self.prompt_bytes}
