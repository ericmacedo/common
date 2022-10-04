from datetime import date, datetime
from typing import Iterable


def defaut_json_serializer(obj: object) -> object:
    """JSON serializer for objects not serializable by default json code"""

    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    else:
        return str(obj)


def are_instances_of(it: Iterable, dtype: object) -> bool:
    it = iter(it)
    while current := next(it, None):
        if not isinstance(current, dtype):
            return False
    return True
