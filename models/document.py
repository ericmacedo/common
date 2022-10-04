import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from dateutil import parser

from utils import defaut_json_serializer


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

    def save(self, path: Path) -> None:
        with open(path.joinpath(f"{self.id}.json"), "w", encoding="utf-8") as jsonFile:
            json.dump(self.asdict(), jsonFile, default=defaut_json_serializer)

    @classmethod
    def load(cls, path: Path) -> Any:
        with open(path, "r", encoding="utf-8") as jsonFile:
            jsonDoc = json.load(jsonFile)
        doc = Document(**jsonDoc)
        doc.date = parser.parse(doc.date)
        return doc
