from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import ClassVar, Dict, List

from sqlalchemy import JSON, Column, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, relationship

from ..helpers.orm import Base, Engine, MapperRegistry
from ..mixins.orm import MixinORM
from .types.dictionary import Dictionary
from .types.embedding import Embedding

DOCUMENT_TABLE = "documents"


def empty_embedding():
    return None


@MapperRegistry.mapped
@dataclass
class DocumentEmbedding(MixinORM):
    __tablename__ = f"{DOCUMENT_TABLE}_embeddings"
    __sa_dataclass_metadata_key__ = "sa"

    id: Mapped[int] = field(init=False,
                            metadata={"sa": Column(Integer, primary_key=True)})
    document_id: Mapped[int] = field(
        metadata={"sa": Column(Integer,
                               ForeignKey(f"{DOCUMENT_TABLE}.id"),
                               unique=True)})
    document: Mapped["Document"] = field(
        init=False,
        metadata={"sa": relationship("Document", back_populates="embedding")})
    embedding: Embedding = field(
        metadata={"sa": Column(Embedding, nullable=False)})

    def as_dict(self) -> Dict:
        return asdict(self)


@MapperRegistry.mapped
@dataclass
class Document(MixinORM):
    __tablename__ = DOCUMENT_TABLE
    __sa_dataclass_metadata_key__ = "sa"

    id: Mapped[int] = field(init=False,
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
    references: List[str] = field(
        default_factory=list,
        metadata={"sa": Column(JSON, nullable=False)})
    embedding: Mapped["DocumentEmbedding"] = field(
        init=False,
        default_factory=empty_embedding,
        metadata={"sa": relationship("DocumentEmbedding",
                                     uselist=False,
                                     back_populates="document",
                                     cascade="all, delete")})
    authors: List[str] = field(default_factory=list,
                               metadata={"sa": Column(JSON, nullable=False)})
    ngrams: Dict[str, int] = field(default_factory=dict,
                                   metadata={"sa": Column(Dictionary, nullable=False)})

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

    def as_dict(self) -> Dict:
        obj = asdict(self)
        obj["embedding"] = self.embedding.embedding if self.embedding else None
        return obj


Base.metadata.create_all(Engine)
