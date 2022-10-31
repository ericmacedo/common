from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, List
from ..mixins.orm import MixinORM
from ..helpers.orm import Base, Engine, MapperRegistry

from sqlalchemy import JSON, Column, Integer, String


@MapperRegistry.mapped
@dataclass
class NGram(MixinORM):
    __tablename__ = "ngrams"
    __sa_dataclass_metadata_key__ = "sa"

    id: int = field(init=False,
                    metadata={"sa": Column(Integer, primary_key=True)})
    ngram: str = field(metadata={"sa": Column(
        String, nullable=False, unique=True)})
    embedding: List[float] = field(init=False, default_factory=list,
                                   metadata={"sa": Column(JSON, nullable=False)})
    frequency: int = field(default=0,
                           metadata={"sa": Column(Integer, nullable=False)})

    FIELDS: ClassVar[List[str]] = ["id", "ngram", "frequency", "embedding"]

    def __eq__(self, ngram: NGram) -> bool:
        return ngram and ngram.ngram == self.ngram


Base.metadata.create_all(Engine)
