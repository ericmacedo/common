from pathlib import Path
from typing import Any, Generator, List, Tuple

import pandas as pd
from ..utils.text import extract_ngrams

from ..utils.miscellaneous import are_instances_of
from .document import Document

from tabulate import tabulate


class Corpus:
    def __init__(self, output_dir: List[str] = [".", "output"]) -> None:
        # Output directory
        self.__output_dir = Path(*output_dir).resolve()
        self.__corpus_dir = self.__output_dir.joinpath("corpus")
        self.__corpus_dir.mkdir(parents=True, exist_ok=True)

        self.__vocab_dict = {
            "path_or_buf": self.__output_dir.joinpath("vocab.tsv"),
            "encoding": "utf-8", "sep": "\t", "index": False, "header": False}

    @property
    def __FIELDS(self) -> List[str]:
        return [
            "id", "doi", "url", "title", "authors", "content",
            "abstract", "citations", "source", "date", "references"]

    @property
    def index(self) -> List[str]:
        return [
            doc.name.split(".")[0]
            for doc in sorted(self.__corpus_dir.glob("*.json"))]

    @property
    def documents(self) -> Generator[Document, None, None]:
        for doc in self.index:
            yield Document.load(self.__corpus_dir.joinpath(f"{doc}.json"))

    def __getitem__(self, indexer: str | int | slice | Tuple[str]) -> Document | List[Any]:
        if isinstance(indexer, str):
            if not indexer in self.__FIELDS:
                raise KeyError(f"Key '{indexer}' not found.")
            return [doc[indexer] for doc in self.documents]
        elif isinstance(indexer, int):
            return Document.load(
                self.__corpus_dir.joinpath(f"{self.index[indexer]}.json"))
        elif isinstance(indexer, slice):
            return [
                Document.load(self.__corpus_dir.joinpath(f"{id}.json"))
                for id in self.index[indexer]]
        elif isinstance(indexer, tuple) and are_instances_of(indexer, str):
            frame = {f"{key}": [] for key in indexer}
            for doc in self.documents:
                for key in indexer:
                    frame[key].append(doc[key])
            return pd.DataFrame(frame)
        else:
            raise KeyError(f"Key '{indexer}' is not a valid indexer.")

    def as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame.from_records(doc.asdict() for doc in self.documents)

    def as_series(self, key: str) -> pd.Series:
        return pd.Series(self[key])

    def generate_ngrams(self) -> None:
        vocab: pd.DataFrame = extract_ngrams(self["content"])
        vocab.to_csv(**self.__vocab_dict)

    def __repr__(self) -> str:
        return tabulate([
            ["Number of documents", len(self.index)],
            ["Location", self.__corpus_dir]])
