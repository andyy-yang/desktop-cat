import json
from pathlib import Path


class PersistenceStore:
    """Dumb JSON layer: stores exactly what it receives."""

    def __init__(self, path: Path):
        self._path = Path(path)

    def load(self) -> dict | None:
        if not self._path.exists():
            return None
        return json.loads(self._path.read_text(encoding="utf-8"))

    def save(self, data: dict) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(data, indent=1), encoding="utf-8")
