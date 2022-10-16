from abc import ABC, abstractmethod
import os
from pathlib import Path
from typing import Any, Generator, Iterable, List, Tuple

import pandas as pd
from tabulate import tabulate

from common.models.vocab import Vocab

from ..utils.itertools import SubscriptableGenerator, sample
from ..utils.miscellaneous import are_instances_of
from ..utils.text import extract_ngrams
from .document import Document


class CorpusBase(ABC):
    INDEXERS = str | int | slice | Tuple[str]
    FIELDS = [
        "id", "doi", "url", "title", "authors", "content", "embedding",
        "abstract", "citations", "source", "date", "references"]

    def __init__(self, output_dir: List[str] = [".", "output"]):
        # Output directory
        self._output_dir = Path(*output_dir).resolve()
        self._corpus_dir = self._output_dir.joinpath("corpus")
        self._corpus_dir.mkdir(parents=True, exist_ok=True)

    @property
    @abstractmethod
    def index(self) -> List[str]:
        pass

    @property
    def documents(self) -> Generator[Document, None, None]:
        for doc in self.index:
            yield Document.load(self._corpus_dir.joinpath(f"{doc}.json"))

    def __getitem__(self, indexer: INDEXERS) -> Document | Iterable[Any]:
        if isinstance(indexer, str):
            if not indexer in self.FIELDS:
                raise KeyError(f"Key '{indexer}' not found.")
            return SubscriptableGenerator(doc[indexer] for doc in self.documents)
        elif isinstance(indexer, int):
            return Document.load(
                self._corpus_dir.joinpath(f"{self.index[indexer]}.json"))
        elif isinstance(indexer, slice):
            return SubscriptableGenerator(
                Document.load(self._corpus_dir.joinpath(f"{id}.json"))
                for id in self.index[indexer])
        elif isinstance(indexer, tuple) and are_instances_of(indexer, str):
            frame = {f"{key}": [] for key in indexer}
            for doc in self.documents:
                for key in indexer:
                    frame[key].append(doc[key])
            return pd.DataFrame(frame)
        else:
            raise KeyError(f"Key '{indexer}' is not a valid indexer.")

    def __iter__(self) -> Generator:
        for id in self.index:
            yield Document.load(self._corpus_dir.joinpath(f"{id}.json"))

    def __len__(self) -> int:
        return len(self.index)

    def as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame.from_records(doc.asdict() for doc in self.documents)

    def as_series(self, key: str) -> pd.Series:
        return pd.Series(self[key])

    def __repr__(self) -> str:
        return tabulate([
            ["Number of documents", len(self)],
            ["Location", self._corpus_dir]])


class CorpusView(CorpusBase):
    def __init__(self, index: Iterable, output_dir: List[str] = [".", "output"]) -> None:
        super(CorpusView, self).__init__(output_dir)

        self.__index = index

    @property
    def index(self) -> List[str]:
        return self.__index


class Corpus(CorpusBase):
    def __init__(self, output_dir: List[str] = [".", "output"]) -> None:
        super(Corpus, self).__init__(output_dir)

        self.__vocab = Vocab(output_dir)

    @property
    def index(self) -> List[str]:
        return [doc.stem for doc in sorted(self._corpus_dir.glob("*.json"))]

    @property
    def vocab(self) -> Vocab:
        return self.__vocab

    def generate_ngrams(self) -> None:
        vocab: pd.DataFrame = extract_ngrams(self["content"])
        vocab.to_csv(**self.__vocab_dict)

    def sample(self, size: float | int, random_state: int = None) -> CorpusView:
        return CorpusView(
            output_dir=[p for p in str(
                self._output_dir.relative_to(self._output_dir.parent)).split(os.sep)],
            index=sample(self.index, size=size, random_state=random_state))
