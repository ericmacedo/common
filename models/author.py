from dataclasses import dataclass, field

from sqlalchemy import Column, Integer, String

from ..helpers.orm import MapperRegistry


@MapperRegistry.mapped
@dataclass
class Author:
    __tablename__ = "authors"
    __sa_dataclass_metadata_key__ = "sa"

    id: int = field(
        init=False, metadata={"sa": Column(Integer, primary_key=True)})
    name: str = field(metadata={"sa": Column(String, nullable=False)})
    affiliation: str = field(metadata={"sa": Column(String, nullable=False)})
