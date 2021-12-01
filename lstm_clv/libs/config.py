from __future__ import annotations

import json
from typing import Any, Dict, Optional


class Config(Dict[str, Any]):
    def __init__(self, values: Dict[str, Any], path: str) -> None:
        self.config = values
        self.path = path

    def set_values(self, values: Dict[str, Any]) -> Config:
        """Replace content of config"""
        self.config = values
        return self

    @classmethod
    def create(cls, path: str) -> Config:
        """Create a new config stored at `path`"""
        return cls(values={}, path=path)

    @classmethod
    def load(cls, path: str) -> Config:
        """Load an existing config stored at `path`"""
        with open(path) as f:
            values = json.load(f)
        return cls(values=values, path=path)

    def save(self, path: Optional[str] = None) -> None:
        """Save config to its original location or a new `path`"""
        path = path or self.path
        with open(path, "w") as f:
            json.dump(self.config, f, indent=4)

    def __getitem__(self, k: str) -> Any:
        try:
            return self.config[k]
        except KeyError:
            raise KeyError(f"Config at {self.path} has no key={k}")

    def __setitem__(self, k: str, v: Any) -> None:
        self.config[k] = v

    def __repr__(self) -> str:
        return f"Config: {self.config}"
