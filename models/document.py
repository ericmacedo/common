from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from hashlib import md5
from pathlib import Path
from typing import Any, Dict, Iterable, List, ClassVar

from dateutil import parser

from ..utils.miscellaneous import defaut_json_serializer


@dataclass
class Document:
    id: str
    doi: str
    url: str = None
    title: str = None
    authors: List[str] = field(default_factory=list)
    content: str = None
    abstract: str = None
    citations: int = None
    source: str = None
    date: datetime = None
    references: List[str] = field(default_factory=list)
    embedding: Iterable[float] = field(default_factory=list)
    ngrams: Dict = field(default_factory=dict)
    
    FIELDS: ClassVar[List[str]] = [
        "id", "doi", "url", "title", "authors", "content", "embedding",
        "abstract", "citations", "source", "date", "references", "ngrams"]

    def asdict(self) -> Dict:
        return asdict(self)

    @property
    def error(self) -> str:
        return self.__error if hasattr(self, "__error") else None

    @error.setter
    def error(self, error_message: str):
        self.__error = error_message

    def __getitem__(self, index: str) -> Any:
        if isinstance(index, str) and hasattr(self, index):
            return getattr(self, index)
        return None

    def __eq__(self, document: Document) -> bool:
        return document.id == self.id

    def save(self, path: str) -> None:
        path = Path(path).resolve()
        with open(path.joinpath(f"{self.id}.json"), "w", encoding="utf-8") as jsonFile:
            json.dump(self.asdict(), jsonFile, default=defaut_json_serializer)

    @classmethod
    def load(cls, path: Path) -> Document:
        with open(path, "r", encoding="utf-8") as jsonFile:
            jsonDoc = json.load(jsonFile)
        doc = Document(**jsonDoc)
        doc.date = parser.parse(doc.date)
        return doc

    @classmethod
    def hash(cls, s: str) -> str:
        return md5(s.encode("utf-8")).hexdigest()
