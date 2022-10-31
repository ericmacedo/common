import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict

import brotli


def defaut_json_serializer(obj: object) -> object:
    """JSON serializer for objects not serializable by default json code"""

    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    else:
        return str(obj)


def load_json(path: str) -> Dict:
    try:
        path = Path(path).resolve()
        with open(path, "rb") as jsonFile:
            decompressed = brotli.decompress(jsonFile.read().decode('utf-8'))
            obj = json.load(decompressed)
    except:
        obj = None
    finally:
        return obj


def save_json(path: str, obj: Any) -> Dict:
    try:
        path = Path(path).resolve()
        with open(path, "wb") as jsonFile:
            compressed = brotli.compress(json.dumps(
                obj, default=defaut_json_serializer
            ).encode("utf-8"))
            jsonFile.write(compressed)
    except:
        obj = None
    finally:
        return obj
