# src/observability/event_logger.py

import json
import threading
from datetime import datetime, timezone
from pathlib import Path


class EventLogger:

    def __init__(self, log_path=None):

        project_root = Path(__file__).resolve().parents[2]

        self.log_path = (
            Path(log_path)
            if log_path
            else project_root / "data" / "logs" / "execution_logs.jsonl"
        )

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def log_event(self, event: dict):

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **event,
        }

        with self._lock:
            with self.log_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, default=str) + "\n")