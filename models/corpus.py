from abc import ABC
from typing import Any, Generator, Iterable, Tuple

import pandas as pd
from pyparsing import abstractmethod
from sqlalchemy.exc import OperationalError
from tabulate import tabulate

from common.helpers.db import DriverDB
from common.models.settings import Settings

from ..embeddings.sbert import SBert
from ..helpers.db import DB
from ..utils.concurrency import batch_processing
from ..utils.itertools import sample
from ..utils.miscellaneous import are_instances
from .document import Document, DocumentEmbedding
from .vocab import NGram, Vocab, VocabBase, VocabView


def process_ngrams(data: Any, **kwargs):
    ngram, frequency = data

    with DriverDB.session_scope(**kwargs) as session:
        while True:
            try:
                if (new_ngram := session.get(NGram, NGram.hash(ngram))) != None:
                    new_ngram.frequency += frequency
                else:
                    new_ngram = NGram(ngram, frequency)

                new_ngram.occurence += 1

                value = new_ngram
                break
            except OperationalError:
                continue

    del db_driver
    return value


class CorpusBase(ABC):
    INDEXERS = str | int | slice | Tuple[str]

    def __init__(self, index: Iterable = None, **kwargs) -> None:
        self._db = DB(Document, index, **kwargs)
        self._db_embeddings = DB(DocumentEmbedding, index, **kwargs)
        self._db_settings = Settings

        self._kwargs = kwargs

    @property
    @abstractmethod
    def index(self) -> Iterable[int]:
        pass

    @property
    def documents(self) -> Iterable[Document]:
        return self._db.rows()

    @property
    def vocab(self) -> VocabBase:
        return self._vocab_view

    def __getitem__(self, indexer: INDEXERS) -> Document | Iterable[Any]:
        is_int, is_slice, is_str, is_iterable = (
            isinstance(indexer, int),
            isinstance(indexer, slice),
            isinstance(indexer, str),
            isinstance(indexer, Iterable) and are_instances(indexer, str))

        if is_int:
            return self._db.find_by_index(indexer)
        elif is_slice:
            return self._db.find_by_slice(indexer)
        elif is_str or is_iterable:
            if is_str:
                indexer = [indexer]

            if any(i for i in indexer if i not in Document.FIELDS):
                raise KeyError(f"Key '{indexer}' not found.")

            return self._db.select_columns(indexer)
        else:
            raise KeyError(f"Key '{indexer}' is not a valid indexer.")

    def __iter__(self) -> Generator:
        for document in self.documents:
            yield document

    def __len__(self) -> int:
        return len(self._db)

    def as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame.from_records(doc.asdict() for doc in self.documents)

    def as_series(self, key: str) -> pd.Series:
        return pd.Series(self[key])

    def __repr__(self) -> str:
        return tabulate([["Number of documents", len(self)]])

    def __contains__(self, doc: Document) -> bool:
        return next((it for it in self if it.id == doc.id), None) is not None


class CorpusView(CorpusBase):
    def __init__(self, index: Iterable, **kwargs) -> None:
        super(CorpusView, self).__init__(index=index, **kwargs)
        self._index = index

        if self._db.is_custom_index():
            ngrams = set()
            for document_ngrams in self["ngrams"]:
                ngrams.update({*document_ngrams.keys()})

            ngram_ids = [NGram.hash(ngram) for ngram in ngrams]

            del ngrams
        else:
            ngram_ids = None

        self._vocab_view: VocabView = VocabView(index=ngram_ids, **kwargs)

    @property
    def index(self) -> Iterable[int]:
        return self._index


class Corpus(CorpusBase):
    def __init__(self, **kwargs) -> None:
        super(Corpus, self).__init__(**kwargs)

        self._vocab: Vocab = Vocab(**kwargs)

    @property
    def index(self) -> Iterable[int]:
        return self._db.index

    def sample(self, size: float | int, random_state: int = None) -> CorpusView:
        return CorpusView(
            [*sample(self.index, size=size, random_state=random_state)],
            **self._kwargs)

    def calculate_document_embeddings(self):
        encoder: SBert = SBert()

        for doc_id, embedding in zip(self["id"],
                                     encoder.encode_documents(self["content"])):
            DocumentEmbedding(document_id=doc_id, embedding=embedding).save()

        del encoder

    def build_vocab(self, resume: bool = False):
        if not resume:
            self.clear_vocab()
            self._db_settings.set("last_document_processed", 0)

        corpus_len = len(self)
        last_document_processed = self._db_settings.get(
            "last_document_processed").value
        for index, document in enumerate(self, 1):
            print(f"Processing document {index}/{corpus_len}", end="\r")

            if resume and index <= last_document_processed:
                continue

            ngrams = batch_processing(fn=process_ngrams,
                                      data=[item for item in document.ngrams.items()
                                            if item[1] > 1],
                                      **self._kwargs)

            self._vocab._db.bulk_update(ngrams)
            ngrams.clear()
            self._db_settings.set("last_document_processed", index)

        self._vocab._db.delete_where(NGram.occurence == 1)

        self._vocab.calculate_embeddings()

    def clear_corpus(self):
        self._db_embeddings.drop_table()
        self._db.drop_table()

        self._db.create_table()
        self._db_embeddings.create_table()

    def clear_vocab(self):
        self._vocab.clear_vocab()
