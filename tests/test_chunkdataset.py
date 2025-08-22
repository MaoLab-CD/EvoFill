# test_dataset.py

from src.utils import load_config
from src.model import ChunkDataset

cfg = load_config('config/config.json')
ds = ChunkDataset(cfg.data.val_chunks_dir,
                    cfg.data.val_dismat,
                    cfg)
count = 0
for seq1, seq2, m1, m2, d in ds:
    count += 1
    if count >= 5:
        break
    print(seq1.shape,seq2.shape,m1.shape,m2.shape,d)