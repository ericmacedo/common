from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

import pandas as pd
from tabulate import tabulate

from common.embeddings.sbert import SBert

from ..helpers.db import DB
from ..models.ngram import NGram, NGramEmbedding


class VocabBase(ABC):
    def __init__(self, index: Iterable = None, **kwargs):
        self._db = DB(NGram, index, **kwargs)
        self._db_embeddings = DB(NGramEmbedding, index, **kwargs)

        self._kwargs = kwargs

    @property
    @abstractmethod
    def index(self) -> Iterable[int]:
        pass

    @property
    def ngrams(self) -> Iterable[NGram]:
        return self._db.rows()

    def __iter__(self) -> Iterable[NGram]:
        for ngram in self.ngrams:
            yield ngram

    def __getitem__(self, indexer: int | str) -> NGram:
        if isinstance(indexer, str):
            return self._db.find(NGram.hash(indexer))
        elif isinstance(indexer, int):
            return self._db.find_by_index(indexer)
        elif isinstance(indexer, slice):
            return self._db.find_by_slice(indexer)
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
    def __init__(self, index: Iterable[str], **kwargs) -> None:
        super(VocabView, self).__init__(index, **kwargs)
        self._index = index

    @property
    def index(self) -> Iterable[str]:
        return self._index


class Vocab(VocabBase):
    def __init__(self, **kwargs) -> None:
        super(Vocab, self).__init__(**kwargs)

    @property
    def index(self) -> Iterable[int]:
        return self._db.index

    def calculate_embeddings(self):
        encoder: SBert = SBert()

        vocab_size = len(self)
        batch_size = 1000
        for i in range(0, vocab_size, 1000):
            print(f"Processing NGrams {i:_}-{i+batch_size:_}/{vocab_size:_}",
                  end="\r")
            ngrams = [*self[i:i+batch_size]]
            embeddings = encoder.encode_ngrams(
                [ngram.ngram for ngram in ngrams])

            ngram_embeddings = [
                NGramEmbedding(ngram_id=ngram.id, embedding=embedding)
                for ngram, embedding in zip(ngrams, embeddings)]
            self._db_embeddings.bulk_update(ngram_embeddings)

            del ngrams, embeddings, ngram_embeddings

    def clear_vocab(self):
        self._db_embeddings.drop_table()
        self._db.drop_table()

        self._db.create_table()
        self._db_embeddings.create_table()
