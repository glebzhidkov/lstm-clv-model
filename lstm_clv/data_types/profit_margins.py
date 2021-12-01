from __future__ import annotations

import json
from typing import Dict, List

import numpy as np


class Margins:
    def __init__(self, margins: Dict[str, float]):
        self.margins = margins

    @classmethod
    def load(cls, path: str) -> Margins:
        with open(path) as f:
            values = json.load(f)
        return cls(values["margins"])

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({"margins": self.margins}, f, indent=4)

    def as_array(self, events: List[str]) -> np.ndarray:
        if len(events) == 1 and events[0] == "ALL":
            return np.array([1.0])
        return np.array([self.margins[event] for event in events])

    def as_dict(self) -> Dict[str, float]:
        return self.margins
