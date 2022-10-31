import pickle
from dataclasses import dataclass, field
from multiprocessing import Process, Queue
from typing import Iterable, List

from ..utils.text import split_sentences


@dataclass
class SBert:
    model_name: str
    embeddings: list = field(default_factory=list)

    def __init__(self, model_name: str = "sentence-transformers/allenai-specter"):
        self.model_name = model_name
        self.embeddings = []

    def _train(self, queue: Queue, corpus: Iterable[str]):
        from sentence_transformers import SentenceTransformer

        transformer = SentenceTransformer(self.model_name)

        embeddings = [
            transformer.encode(
                split_sentences(document)
            ).mean(axis=0).tolist() for document in corpus]

        del transformer
        SBert.clear_memory()

        queue.put(embeddings)

    def train(self, corpus: Iterable[str]) -> list:
        queue = Queue()
        p = Process(
            target=self._train,
            kwargs={"queue": queue, "corpus": corpus})
        p.start()
        self.embeddings = queue.get()
        p.join()

        return self.embeddings

    def predict(self, data: Iterable[str]) -> List[List[float]]:
        queue = Queue()
        p = Process(
            target=self._train,
            kwargs={"queue": queue, "corpus": data})
        p.start()
        embeddings = queue.get()
        p.join()

        return embeddings

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
