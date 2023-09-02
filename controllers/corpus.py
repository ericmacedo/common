from abc import ABC
from datetime import datetime
from typing import Any, Generator, Iterable, List, Optional, Tuple

import pandas as pd
from pydantic import BaseModel, Field
from pyparsing import abstractmethod
from sqlalchemy import desc
from sqlalchemy.exc import OperationalError
from tabulate import tabulate

from common.database.connector import DriverDB
from common.models.settings import Settings

from ..database.service import ServiceDB
from ..embeddings.sbert import SBert
from ..models.document import Document, DocumentEmbedding
from ..utils.concurrency import batch_processing
from ..utils.miscellaneous import are_instances
from .vocab import NGram, Vocab


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

    return value


class CorpusBaseQuery(BaseModel):
    db_name: str


class FilterBase(BaseModel):
    doi_filter: Optional[str]
    url_filter: Optional[str]
    title_filter: Optional[str]
    abstract_filter: Optional[str]
    citatitons_filter: Optional[str]
    date_filter: Optional[str]


class SortBase(BaseModel):
    doi_sort: Optional[int]
    url_sort: Optional[int]
    title_sort: Optional[int]
    abstract_sort: Optional[int]
    citations_sort: Optional[int]
    date_sort: Optional[int]


class CorpusQuery(CorpusBaseQuery, FilterBase, SortBase):
    page: Optional[int] = Field(None, description="Page number to query")
    page_size: Optional[int] = Field(
        None, description="Number of rows per page")
    metadata: Optional[bool] = Field(
        False, description="Whether to return only metadata")


class CorpusControllerBase(ABC):
    INDEXERS = str | int | slice | Tuple[str]

    def __init__(self, **kwargs) -> None:
        DriverDB(**kwargs).create_all()

        self._db = ServiceDB(Document, **kwargs)
        self._db_embeddings = ServiceDB(DocumentEmbedding, **kwargs)
        self._db_settings = Settings

        self._kwargs = kwargs

    @property
    @abstractmethod
    def index(self) -> Iterable[int]:
        pass

    @property
    def documents(self) -> Iterable[Document]:
        return self._db.rows()

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

    def len(self, query: CorpusBaseQuery = None) -> int:
        return self._db.len(self.__get_filter_params(query)
                            if query else query)

    def min(self, column: str) -> int:
        min = self._db.min(column)
        return int(min.year if isinstance(min, datetime) else min)

    def max(self, column: str) -> int:
        max = self._db.max(column)
        return int(max.year if isinstance(max, datetime) else max)

    def find(self, doc_id: int) -> Document:
        return self._db.find(doc_id)

    def find_where(self, query: CorpusQuery) -> Iterable[Document]:
        return self._db.find_where(
            page=query.page,
            page_size=query.page_size,
            filters=self.__get_filter_params(query),
            sort=self.__get_sort_params(query))

    def as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame.from_records(doc.asdict() for doc in self.documents)

    def as_series(self, key: str) -> pd.Series:
        return pd.Series(self[key])

    def __repr__(self) -> str:
        return tabulate([["Number of documents", len(self)]])

    def __contains__(self, doc: Document) -> bool:
        return next((it for it in self if it.id == doc.id), None) is not None

    def __del__(self):
        del self._db, self._db_embeddings

    def __get_filter_params(self, query: FilterBase) -> List:
        params = []

        if (doi := getattr(query, "doi_filter", None)):
            params.append(Document.doi.contains(doi))

        if (url := getattr(query, "url_filter", None)):
            params.append(Document.url.contains(url))

        if (title := getattr(query, "title_filter", None)):
            params.append(Document.title.contains(title))

        if (abstract := getattr(query, "abstract_filter", None)):
            params.append(Document.abstract.contains(abstract))

        if (citations := getattr(query, "citations_filter", None)):
            citations = citations.split(",")
            params.append(Document.citations.between(*citations))

        if (date := getattr(query, "date_filter", None)):
            date = date.split(",")
            date = map(lambda d: datetime.strptime(str(d), '%Y'), date)
            params.append(Document.date.between(*date))

        return params

    def __get_sort_params(self, query: SortBase) -> List:
        fields = ["doi", "url", "title", "abstract", "citations", "date"]

        params = []
        for field in fields:
            if sort := getattr(query, f"{field}_sort", None):
                sort = int(sort)
                doc_field = Document[field]
                params.append(doc_field if sort > 0 else desc(doc_field))

        return params


class CorpusController(CorpusControllerBase):
    def __init__(self, **kwargs) -> None:
        super(CorpusController, self).__init__(**kwargs)

        self._vocab: Vocab = Vocab(**kwargs)

    @property
    def index(self) -> Iterable[int]:
        return self._db.index

    def calculate_document_embeddings(self):
        encoder: SBert = SBert()

        for doc_id, embedding in zip(self["id"],
                                     encoder.encode_documents(self["content"])):
            DocumentEmbedding(document_id=doc_id,
                              embedding=embedding).save(**self._kwargs)

        del encoder

    def build_vocab(self, resume: bool = False):
        if not resume:
            self.clear_vocab()
            self._db_settings.set("last_document_processed", 0, **self._kwargs)

        corpus_len = len(self)
        last_document_processed = self._db_settings.get(
            "last_document_processed", **self._kwargs).value
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
            self._db_settings.set(
                "last_document_processed", index, **self._kwargs)

        self._vocab._db.delete_where(NGram.occurence == 1)

        self._vocab.calculate_embeddings()

    def clear_corpus(self):
        self._db_embeddings.drop_table()
        self._db.drop_table()

        self._db.create_table()
        self._db_embeddings.create_table()

    def clear_vocab(self):
        self._vocab.clear_vocab()
