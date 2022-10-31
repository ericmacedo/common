from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import ClassVar, Dict, List


from sqlalchemy import JSON, Column, DateTime, Integer, String

from ..helpers.orm import Base, Engine, MapperRegistry
from ..mixins.orm import MixinORM


@MapperRegistry.mapped
@dataclass
class Document(MixinORM):
    __tablename__ = "documents"
    __sa_dataclass_metadata_key__ = "sa"

    id: int = field(init=False,
                    metadata={"sa": Column(Integer, primary_key=True)})
    doi: str = field(metadata={"sa": Column(
        String, nullable=False, unique=True)})
    url: str = field(metadata={"sa": Column(String, nullable=False)})
    title: str = field(metadata={"sa": Column(String, nullable=False)})
    content: str = field(metadata={"sa": Column(String, nullable=False)})
    abstract: str = field(metadata={"sa": Column(String, nullable=False)})
    citations: int = field(metadata={"sa": Column(Integer, nullable=False)})
    source: str = field(metadata={"sa": Column(String, nullable=False)})
    date: datetime = field(metadata={"sa": Column(DateTime, nullable=False)})
    references: List[str] = field(default_factory=list,
                                  metadata={"sa": Column(JSON, nullable=False)})
    embedding: List[float] = field(init=False, default_factory=list,
                                   metadata={"sa": Column(JSON, nullable=False)})
    authors: List[str] = field(default_factory=list,
                               metadata={"sa": Column(JSON, nullable=False)})
    ngrams: Dict[str, int] = field(init=False, default_factory=dict,
                                   metadata={"sa": Column(JSON, nullable=False)})

    FIELDS: ClassVar[List[str]] = [
        "id", "doi", "url", "title", "authors", "content", "embedding",
        "abstract", "citations", "source", "date", "references", "ngrams"]

    @property
    def error(self) -> str:
        return self.__error if hasattr(self, "__error") else None

    @error.setter
    def error(self, error_message: str):
        self.__error = error_message

    def __eq__(self, document: Document) -> bool:
        return document and document.id == self.id


Base.metadata.create_all(Engine)
