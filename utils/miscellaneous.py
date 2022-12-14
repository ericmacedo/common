from typing import Any, Callable, Dict, Iterable, Literal


def are_instances(it: Iterable, dtype: object) -> bool:
    it = iter(it)
    while current := next(it, None):
        if not isinstance(current, dtype):
            return False
    return True


def apply(s: str, *args: Callable) -> str:
    for fn in args:
        s = fn(s)
    return s


def count_dictionary(it: Iterable[Any],
                     sort_by: Literal["key", "value"] = "value",
                     reversed: bool = False,
                     n: int = None) -> Dict[str, int]:
    d = {}
    for i in it:
        for j in i:
            d[j] = d.get(j, 0) + 1
    return {
        k: v for k, v in sorted(
            d.items(),
            key=lambda item: item[0 if sort_by == "key" else 1],
            reverse=reversed)[:n]}
