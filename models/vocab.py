from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from hashlib import md5
from pathlib import Path
from typing import Any, ClassVar, Dict, Generator, Iterable, List

import pandas as pd
from common.utils.itertools import SubscriptableGenerator
from tabulate import tabulate

from ..utils.miscellaneous import defaut_json_serializer


@dataclass
class NGram:
    ngram: str
    frequency: int
    embedding: List[float] = field(default_factory=list)

    FIELDS: ClassVar[List[str]] = ["ngram", "frequency", "embedding"]

    def asdict(self) -> Dict:
        return asdict(self)

    def save(self, path: str) -> None:
        path = Path(path).resolve()
        id = NGram.hash(self.ngram)
        with open(path.joinpath(f"{id}.json"), "w", encoding="utf-8") as jsonFile:
            json.dump(self.asdict(), jsonFile, default=defaut_json_serializer)

    @classmethod
    def load(cls, path: Path) -> NGram:
        with open(path, "r", encoding="utf-8") as jsonFile:
            jsonNGram = json.load(jsonFile)
        ngram = NGram(**jsonNGram)
        return ngram

    @classmethod
    def hash(cls, s: str) -> str:
        return md5(s.encode("utf-8")).hexdigest()


class VocabBase(ABC):
    def __init__(self, output_dir: List[str] = [".", "output"]) -> None:
        # Output directory
        self._output_dir = Path(*output_dir).resolve()
        self._vocab_dir = self._output_dir.joinpath("vocab")
        self._vocab_dir.mkdir(parents=True, exist_ok=True)

    @property
    @abstractmethod
    def index(self) -> Iterable[str]:
        pass

    @property
    def ngrams(self) -> Iterable[NGram]:
        return SubscriptableGenerator(
            NGram.load(self._vocab_dir.joinpath(f"{ngram}.json"))
            for ngram in self.index)

    def __iter__(self) -> Generator[NGram, None, None]:
        for id in self.index:
            yield NGram.load(self._vocab_dir.joinpath(f"{id}.json"))

    def __getitem__(self, indexer: int | str) -> NGram:
        if isinstance(indexer, str):
            if indexer not in self:
                raise KeyError(f"Key '{indexer}' not found.")
            indexer = NGram.hash(indexer)
        elif isinstance(indexer, int):
            indexer = self.index[indexer]
        else:
            raise KeyError(f"Key '{indexer}' is not a valid indexer.")
        return NGram.load(self._vocab_dir.joinpath(f"{indexer}.json"))

    def as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame.from_records(ngram.asdict() for ngram in self.ngrams)

    def __len__(self) -> int:
        return len(self.index)

    def __repr__(self) -> str:
        return tabulate([
            ["Vocabulary size", len(self)],
            ["Location", self._vocab_dir]])

    def __contains__(self, ngram: str) -> bool:
        return self._vocab_dir.joinpath(f"{NGram.hash(ngram)}.json").exists()


class VocabView(VocabBase):
    def __init__(self, index: Iterable[str], output_dir: List[str] = [".", "output"]) -> None:
        super(VocabView, self).__init__(output_dir)

        self.__index = index

    @property
    def index(self) -> Iterable[str]:
        return self.__index


class Vocab(VocabBase):
    def __init__(self, output_dir: List[str] = [".", "output"]) -> None:
        super(Vocab, self).__init__(output_dir)

    @property
    def index(self) -> Iterable[str]:
        return SubscriptableGenerator(ngram.stem for ngram in sorted(
            self._vocab_dir.glob("*.json")))

    def update_ngram(self, ngram: NGram):
        ngram.save(self._vocab_dir)

    def clear_vocab(self):
        for f in self._vocab_dir.glob("*.json"):
            f.unlink()
