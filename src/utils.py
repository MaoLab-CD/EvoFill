import json
from types import SimpleNamespace

def load_config(path: str) -> SimpleNamespace:
    def boolify(obj):
        """把字符串 true/false 转成 Python 布尔值，其余原样返回。"""
        if isinstance(obj, str):
            lower = obj.lower()
            if lower == "true":
                return True
            if lower == "false":
                return False
        return obj

    def hook(d):
        return SimpleNamespace(**{
            k: (hook(v) if isinstance(v, dict) else boolify(v))
            for k, v in d.items()
        })

    with open(path, encoding="utf-8") as f:
        return json.load(f, object_hook=hook)