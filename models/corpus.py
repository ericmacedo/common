from abc import ABC
from typing import Any, Generator, Iterable, Tuple

import pandas as pd
from pyparsing import abstractmethod
from tabulate import tabulate

from ..embeddings.sbert import SBert
from ..helpers.db import DB
from ..helpers.orm import session_scope
from ..utils.concurrency import batch_processing
from ..utils.itertools import sample
from ..utils.miscellaneous import are_instances
from .document import Document, DocumentEmbedding
from .ngram import NGramEmbedding
from .vocab import NGram, Vocab, VocabBase, VocabView


class CorpusBase(ABC):
    INDEXERS = str | int | slice | Tuple[str]

    def __init__(self, index: Iterable = None) -> None:
        self._db = DB(Document, index)
        self._db_embeddings = DB(DocumentEmbedding, index)

    @property
    @abstractmethod
    def index(self) -> Iterable[int]:
        pass

    @property
    def documents(self) -> Iterable[Document]:
        return self._db.rows(self.index)

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
    def __init__(self, index: Iterable) -> None:
        super(CorpusView, self).__init__(index)
        self._index = index

        ngrams = set()
        for document in self:
            for ngram, _ in document.ngrams.items():
                ngrams.add(ngram)

        with session_scope() as s:
            ngram_ids = s.query(NGram).filter(
                NGram.ngram.in_(ngrams)
            ).order_by(NGram.id).with_entities(NGram.id).all()

        self._vocab_view: VocabView = VocabView(
            index=[i[0] for i in ngram_ids])

    @property
    def index(self) -> Iterable[int]:
        return self._index


class Corpus(CorpusBase):
    def __init__(self) -> None:
        super(Corpus, self).__init__()

        self._vocab: Vocab = Vocab()
        self._vocab_view: VocabView = VocabView(self._vocab.index)

    @property
    def index(self) -> Iterable[int]:
        return self._db.index

    def sample(self, size: float | int, random_state: int = None) -> CorpusView:
        return CorpusView(sample(self.index, size=size, random_state=random_state))

    def calculate_document_embeddings(self):
        encoder: SBert = SBert()

        for doc_id, embedding in zip(self["id"],
                                     encoder.encode_documents(self["content"])):
            DocumentEmbedding(document_id=doc_id, embedding=embedding).save()

        del encoder

    def build_vocab(self, resume: bool = False):

        encoder: SBert = SBert()

        if not resume:
            self._vocab.clear_vocab()

        corpus_len = len(self)
        for i, document in enumerate(self):
            print(f"Processing document {i+1}/{corpus_len}", end="\r")

            if resume and document.ngrams:
                continue

            ngrams = []
            for ngram, frequency in document.ngrams.items():
                if (new_ngram := self._vocab[ngram]) != None:
                    new_ngram.frequency += frequency
                else:
                    new_ngram = NGram(ngram, frequency)

                new_ngram.occurence += 1
                ngrams.append(new_ngram)

            # calculates embedding for new ngrams
            ngrams_to_calculate_embedding = [
                index for index, ngram in enumerate(ngrams)
                if not getattr(ngram, "id", None)]
            ngrams_embeddings = zip(
                ngrams_to_calculate_embedding,
                [*encoder.encode_ngrams(
                    [ngrams[i].ngram for i in ngrams_to_calculate_embedding])])

            document.update({"ngrams": document.ngrams})
            self._vocab._db.bulk_update(ngrams)
            self._vocab._db_embeddings.bulk_update([
                NGramEmbedding(ngram_id=ngrams[index].id, embedding=embedding)
                for index, embedding in ngrams_embeddings])

            ngrams.clear()
            del ngrams, ngrams_to_calculate_embedding, ngrams_embeddings

        del encoder

    def clear_corpus(self):
        self._db_embeddings.drop_table()
        self._db.drop_table()

        self._db.create_table()
        self._db_embeddings.create_table()

        self._vocab.clear_vocab()
