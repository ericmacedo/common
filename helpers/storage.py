from abc import ABC
from pathlib import Path
from typing import Generic, Iterable, List, TypeVar

T = TypeVar('T')


class ResourcesManager(Generic[T]):
    def __init__(self, db: bool, path: str | List[str] = [".", "output"]):
        if not db:
            self.resource_path = Path(
                *path if isinstance(path, Iterable) else dir
            ).resolve()
            self.resource_path.mkdir(parents=True, exist_ok=True)

        self.__db: bool = db

    @property
    def is_db(self) -> bool:
        return self.__db

    def get_id(self, id: int | str) -> int | str:
        return int(id) if self.is_db else self.__resolve_path(id)

    def __resolve_path(self, id: int | str) -> str:
        return self.resource_path.joinpath(f"{id}.json")
