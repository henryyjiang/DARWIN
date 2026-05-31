"""Whitelisted dataset retrieval (ARCHITECTURE.md §8.3 / §9.3 `data.*`).

`data_search(query)` / `data_fetch(dataset_id, revision)` over the Hugging Face Hub API (a
whitelisted host, §8.3). This enacts the **data philosophy**: models acquire training data by
pulling *existing, precompiled, license-clear* datasets — they do **not** write scrapers. Each
result returns the dataset **card + license string** so provenance (the dataset id@revision +
license) can be recorded in the genome and memory (`datasets_used`), the same attribution
discipline as papers (§8.4).

Pure JSON parsing (`parse_dataset`, `parse_search`) is unit-tested against canned HF responses;
the `Transport` (default `UrllibTransport`) is injected and whitelist-gated.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from darwin.sources.transport import Transport, UrllibTransport

_HF_API = "https://huggingface.co/api/datasets"


@dataclass
class DatasetRef:
    """A retrieved dataset + its license/card (§8.3 provenance)."""

    dataset_id: str
    revision: str = "main"
    license: str = ""
    description: str = ""
    downloads: int = 0
    likes: int = 0
    tags: list[str] = field(default_factory=list)
    gated: bool = False

    @property
    def pinned_id(self) -> str:
        """The id@revision pin to record in `datasets_used` (§8.3)."""
        return f"{self.dataset_id}@{self.revision}"

    @property
    def url(self) -> str:
        return f"https://huggingface.co/datasets/{self.dataset_id}"

    def to_dict(self) -> dict:
        return {
            "dataset_id": self.dataset_id,
            "revision": self.revision,
            "pinned_id": self.pinned_id,
            "license": self.license,
            "description": self.description,
            "downloads": self.downloads,
            "likes": self.likes,
            "tags": list(self.tags),
            "gated": self.gated,
            "url": self.url,
        }


def _license_from(card_data: dict, tags: list[str]) -> str:
    if isinstance(card_data, dict) and card_data.get("license"):
        lic = card_data["license"]
        return ", ".join(lic) if isinstance(lic, list) else str(lic)
    # fall back to a `license:xxx` tag
    for tag in tags:
        if isinstance(tag, str) and tag.startswith("license:"):
            return tag.split(":", 1)[1]
    return ""


def parse_dataset(obj: dict, *, revision: str = "main") -> DatasetRef:
    """Build a DatasetRef from an HF `/api/datasets/{id}` object (pure)."""
    tags = [t for t in obj.get("tags", []) if isinstance(t, str)]
    card = obj.get("cardData") or {}
    sha = obj.get("sha")
    return DatasetRef(
        dataset_id=obj.get("id", ""),
        revision=revision if revision != "main" else (sha or "main"),
        license=_license_from(card, tags),
        description=(card.get("pretty_name") if isinstance(card, dict) else "") or obj.get("description", "") or "",
        downloads=int(obj.get("downloads", 0) or 0),
        likes=int(obj.get("likes", 0) or 0),
        tags=tags,
        gated=bool(obj.get("gated", False)),
    )


def parse_search(text: str) -> list[DatasetRef]:
    """Parse an HF `/api/datasets?search=` JSON list (pure)."""
    data = json.loads(text)
    if not isinstance(data, list):
        return []
    return [parse_dataset(obj) for obj in data if isinstance(obj, dict)]


class DataSource:
    """Whitelisted dataset retrieval over the HF Hub API (§9.3 `data.*`)."""

    def __init__(self, transport: Transport | None = None):
        self.transport = transport or UrllibTransport()

    def search(self, query: str, *, limit: int = 5) -> list[DatasetRef]:
        if not query.strip():
            return []
        text = self.transport.get_text(
            _HF_API, {"search": query, "limit": str(max(1, limit)), "full": "true"}
        )
        return parse_search(text)[:limit]

    def fetch(self, dataset_id: str, revision: str = "main") -> DatasetRef | None:
        base = f"{_HF_API}/{dataset_id}"
        if revision and revision != "main":
            base = f"{base}/revision/{revision}"
        text = self.transport.get_text(base)
        obj = json.loads(text)
        if not isinstance(obj, dict) or not obj.get("id"):
            return None
        return parse_dataset(obj, revision=revision)
