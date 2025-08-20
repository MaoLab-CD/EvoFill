import json
from types import SimpleNamespace

def load_config(path: str) -> SimpleNamespace:
    def hook(d):
        return SimpleNamespace(**{k: hook(v) if isinstance(v, dict) else v
                                  for k, v in d.items()})
    with open(path) as f:
        return json.load(f, object_hook=hook)