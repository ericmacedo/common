from datetime import datetime
from typing import Any


def datetime_to_intYear(date: datetime) -> int:
    return int(date.year)


def default_json_parser(item: Any):
    if isinstance(item, datetime):
        return datetime_to_intYear(item)
    else:
        return str(item)
