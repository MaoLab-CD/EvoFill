import json
from types import SimpleNamespace
from pathlib import Path
import torch
import numpy as np

def load_config(path: str) -> SimpleNamespace:
    def str2bool(v):
        """把字符串 'true'/'false' 转成布尔值，其余原样返回。"""
        if isinstance(v, str) and v.lower() in ('true', 'false'):
            return v.lower() == 'true'
        return v

    def hook(d):
        # 先转布尔，再递归
        d = {k: str2bool(v) for k, v in d.items()}
        return SimpleNamespace(**{k: hook(v) if isinstance(v, dict) else v
                                  for k, v in d.items()})

    with open(path) as f:
        return json.load(f, object_hook=hook)

def export_config(obj, *, f=None, indent=None, ensure_ascii=False, **kwargs):
    """
    将任意嵌套对象（含 SimpleNamespace / Path / Tensor / ndarray）序列化为 JSON。
    参数
    ----
    obj : 任意对象
    f   : None -> 返回 str
          传入文件路径(str/Path) -> 落盘
    indent, ensure_ascii, **kwargs 同 json.dumps
    """
    def _convert(o):
        if isinstance(o, SimpleNamespace):
            return {k: _convert(v) for k, v in vars(o).items()}
        if isinstance(o, dict):
            return {k: _convert(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_convert(i) for i in o]
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, torch.Tensor):
            return o.detach().cpu().tolist()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o

    data = _convert(obj)
    if f is None:
        return json.dumps(data, indent=indent, ensure_ascii=ensure_ascii, **kwargs)
    # 落盘
    Path(f).parent.mkdir(parents=True, exist_ok=True)
    with open(f, "w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=indent, ensure_ascii=ensure_ascii, **kwargs)