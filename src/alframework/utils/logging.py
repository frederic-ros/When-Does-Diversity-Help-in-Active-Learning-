from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import json
from typing import Any, Dict, List, Optional

@dataclass
class ResultLogger:
    out_dir: Path

    def __post_init__(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def save_config(self, config: Dict[str, Any], filename: str = "config.json") -> None:
        (self.out_dir / filename).write_text(json.dumps(config, indent=2), encoding="utf-8")

    def save_history(self, history: List[Dict[str, Any]], filename: str = "history.json") -> None:
        (self.out_dir / filename).write_text(json.dumps(history, indent=2, default=str), encoding="utf-8")
