from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

import pandas as pd
from tabulate import tabulate

from ..helpers.db import DB
from ..models.ngram import NGram, NGramEmbedding


class VocabBase(ABC):
    def __init__(self, index: Iterable = None):
        self._db = DB(NGram, index)
        self._db_embeddings = DB(NGramEmbedding, index)

    @property
    @abstractmethod
    def index(self) -> Iterable[int]:
        pass

    @property
    def ngrams(self) -> Iterable[NGram]:
        return self._db.rows(self.index)

    def __iter__(self) -> Iterable[NGram]:
        for ngram in self.ngrams:
            yield ngram

    def __getitem__(self, indexer: int | str) -> NGram:
        if isinstance(indexer, str):
            return self._db.find_by_match(ngram=indexer)
        elif isinstance(indexer, int):
            return self._db.find_by_index(indexer)
        else:
            raise KeyError("Key '{0}' ({1}) is not a valid indexer.".format(
                indexer, type(indexer)))

    def as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame.from_records(ngram.asdict() for ngram in self.ngrams)

    def __len__(self) -> int:
        return len(self.index)

    def __repr__(self) -> str:
        return tabulate([["Vocabulary size", len(self)]])

    def __contains__(self, ngram: str) -> bool:
        return next(
            (it for it in self if it.ngram == ngram.ngram), None
        ) is not None


class VocabView(VocabBase):
    def __init__(self, index: Iterable[str]) -> None:
        super(VocabView, self).__init__(index)
        self._index = index

    @property
    def index(self) -> Iterable[str]:
        return self._index


class Vocab(VocabBase):
    def __init__(self) -> None:
        super(Vocab, self).__init__()

    @property
    def index(self) -> Iterable[int]:
        return self._db.index

    def clear_vocab(self):
        self._db_embeddings.drop_table()
        self._db.drop_table()

        self._db.create_table()
        self._db_embeddings.create_table()
