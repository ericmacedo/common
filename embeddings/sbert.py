import pickle
from dataclasses import dataclass, field
from typing import Generator, Iterable

from sentence_transformers import SentenceTransformer

from ..utils.text import split_sentences


@dataclass
class SBert:
    model_name: str
    embeddings: list = field(default_factory=list)

    def __init__(self, model_name: str = "sentence-transformers/allenai-specter"):
        self.model_name = model_name
        self.embeddings = []

    def encode_documents(self, corpus: Iterable[str]) -> Generator[Iterable[float], None, None]:
        transformer = SentenceTransformer(self.model_name)

        size = len(corpus)
        for index, document in enumerate(corpus):
            print(f"Processing embedding for document {index + 1}/{size}",
                  end="\r")
            yield [*transformer.encode(split_sentences(document)).mean(axis=0)]

        del transformer
        SBert.clear_memory()

    def encode_ngrams(self, ngrams: Iterable[str]) -> Generator[Iterable[float], None, None]:
        transformer = SentenceTransformer(self.model_name)

        embeddings = transformer.encode(ngrams).tolist()
        while embeddings:
            yield embeddings.pop(0)

        del transformer
        SBert.clear_memory()

    @classmethod
    def load(cls, path: str):
        return pickle.load(open(path, "rb"))

    @classmethod
    def clear_memory(cls):
        from gc import collect

        from torch.cuda import empty_cache, ipc_collect

        collect()
        ipc_collect()
        empty_cache()

    def save(self, path: str):
        with open(path, "wb") as pkl_file:
            pickle.dump(
                obj=self,
                file=pkl_file,
                protocol=pickle.DEFAULT_PROTOCOL,
                fix_imports=True)
