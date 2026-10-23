from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

SYSTEM_CONTENT = (
    "You are a waypoint recommendation model. You analyse a user's past trips and "
    "the current trip to determine which POIs will be visited on the way."
)

_POI_ID = re.compile(r"POI id (\d+)")


class PoiVocab:
    def __init__(self, poi_ids: Sequence[str]):
        self.ids: List[str] = [str(p) for p in poi_ids]
        self.index: Dict[str, int] = {p: i for i, p in enumerate(self.ids)}

    def __len__(self) -> int:
        return len(self.ids)

    def decode(self, indices: Sequence[int]) -> List[str]:
        return [self.ids[i] for i in indices]

    @staticmethod
    def load(path) -> "PoiVocab":
        return PoiVocab(json.loads(Path(path).read_text())["poi_ids"])


class PromptDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_len: int):
        self.rows = [json.loads(l) for l in
                     Path(jsonl_path).read_text().splitlines() if l.strip()]
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.rows)

    def tokens(self, i: int) -> List[int]:
        ids = self.tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM_CONTENT},
             {"role": "user", "content": self.rows[i]["question"]}],
            add_generation_prompt=False, tokenize=True)
        return ids[-self.max_len:] if len(ids) > self.max_len else ids


class WaypointDataset(PromptDataset):
    def __init__(self, jsonl_path, tokenizer, max_len: int, vocab: PoiVocab,
                 user2row: Optional[Dict[str, int]] = None):
        super().__init__(jsonl_path, tokenizer, max_len)
        self.user2row = user2row or {}
        side = Path(jsonl_path).with_name("memory_" + Path(jsonl_path).name)
        if not side.exists():
            raise FileNotFoundError(f"{side} missing: run preprocess/build_dataset.py")
        feats = {}
        for line in open(side):
            d = json.loads(line)
            feats[str(d["trajectory_id"])] = (d["idx"], d["cnt"], d["dt"])
        self.memory = [feats[str(r["trajectory_id"])] for r in self.rows]
        self.se = []
        for r in self.rows:
            q = r["question"]
            j = q.find("The following data")
            m = _POI_ID.findall(q[j:] if j >= 0 else q)[:2]
            ok = len(m) == 2
            self.se.append((vocab.index.get(m[0], -1) if ok else -1,
                            vocab.index.get(m[1], -1) if ok else -1))

    def __getitem__(self, i: int) -> Dict:
        r = self.rows[i]
        return {"input_ids": self.tokens(i),
                "label_indices": r["label_indices"],
                "memory": self.memory[i],
                "gold": r["gold_poi_ids"],
                "se": self.se[i],
                "user_row": self.user2row.get(str(r.get("user_id")), -1)}


class WaypointCollator:
    def __init__(self, pad_id: int, n_poi: int):
        self.pad_id, self.n_poi = pad_id, n_poi

    def __call__(self, batch: List[Dict]):
        maxlen = max(len(b["input_ids"]) for b in batch)
        B, V = len(batch), self.n_poi
        input_ids, attn = [], []
        labels = torch.zeros(B, V)
        memory = torch.zeros(B, V, 3)
        for i, b in enumerate(batch):
            pad = maxlen - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [self.pad_id] * pad)
            attn.append([1] * len(b["input_ids"]) + [0] * pad)
            labels[i, b["label_indices"]] = 1.0
            idx, cnt, dt = b["memory"]
            if idx:
                ii = torch.tensor(idx)
                memory[i, ii, 0] = 1.0
                memory[i, ii, 1] = torch.log1p(torch.tensor(cnt, dtype=torch.float))
                memory[i, ii, 2] = torch.tensor(dt, dtype=torch.float)
        return {"input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attn, dtype=torch.long),
                "labels": labels,
                "memory": memory,
                "se": torch.tensor([b["se"] for b in batch], dtype=torch.long),
                "user_idx": torch.tensor([b["user_row"] for b in batch], dtype=torch.long),
                "gold": [b["gold"] for b in batch]}


def load_user_embeddings(npz_path: str):
    z = np.load(npz_path, allow_pickle=True)
    users = [str(u) for u in z["user_ids"]]
    return {u: i for i, u in enumerate(users)}, torch.from_numpy(z["U"].astype(np.float32))


def load_route_matrix(npz_path: str) -> torch.Tensor:
    return torch.from_numpy(np.load(npz_path)["duration_min"].astype("float32"))
