from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import ClassVar, Dict, List

from sqlalchemy import Column, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, relationship

from ..helpers.db import MapperRegistry, MixinORM
from .types.embedding import Embedding


def empty_embedding():
    return None


def hash_id(context) -> str:
    return MixinORM.hash(context.current_parameters["ngram"])


NGRAM_TABLE = "ngrams"


@MapperRegistry.mapped
@dataclass
class NGramEmbedding(MixinORM):
    __tablename__ = f"{NGRAM_TABLE}_embeddings"
    __sa_dataclass_metadata_key__ = "sa"

    id: Mapped[int] = field(init=False,
                            metadata={"sa": Column(Integer, primary_key=True)})
    ngram_id: Mapped[str] = field(
        metadata={"sa": Column(String(32),
                               ForeignKey(f"{NGRAM_TABLE}.id",
                                          ondelete="CASCADE"),
                               unique=True)})
    ngram: Mapped["NGram"] = field(
        init=False,
        metadata={"sa": relationship("NGram", back_populates="embedding")})
    embedding: Embedding = field(
        metadata={"sa": Column(Embedding, nullable=False)})

    def as_dict(self) -> Dict:
        return asdict(self)


@MapperRegistry.mapped
@dataclass
class NGram(MixinORM):
    __tablename__ = NGRAM_TABLE
    __sa_dataclass_metadata_key__ = "sa"

    id: Mapped[str] = field(init=False,
                            metadata={"sa": Column(String(32),
                                                   primary_key=True,
                                                   default=hash_id)})
    ngram: str = field(metadata={"sa": Column(
        String, nullable=False, unique=True)})
    embedding: Mapped["NGramEmbedding"] = field(
        init=False,
        default_factory=empty_embedding,
        metadata={"sa": relationship("NGramEmbedding",
                                     uselist=False,
                                     back_populates="ngram",
                                     cascade="all, delete")})
    frequency: int = field(default=0,
                           metadata={"sa": Column(Integer, nullable=False)})
    occurence: int = field(default=0,
                           metadata={"sa": Column(Integer, nullable=False)})

    FIELDS: ClassVar[List[str]] = [
        "id", "ngram", "frequency", "embedding", "occurence"]

    def __eq__(self, ngram: NGram) -> bool:
        return ngram and ngram.ngram == self.ngram

    def as_dict(self) -> Dict:
        obj = asdict(self)
        obj["embedding"] = self.embedding.embedding if self.embedding else None
        return obj


# Base.metadata.create_all(Engine)
