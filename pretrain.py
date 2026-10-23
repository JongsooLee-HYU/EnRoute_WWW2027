#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from data import PromptDataset
from model import ActivityModel


class ActivityDataset(PromptDataset):
    def __init__(self, jsonl_path, tokenizer, max_len: int):
        super().__init__(jsonl_path, tokenizer, max_len)
        self.rows = [r for r in self.rows if r.get("has_type")]

    def __getitem__(self, i: int):
        return {"input_ids": self.tokens(i), "type_indices": self.rows[i]["type_indices"]}


class ActivityCollator:
    def __init__(self, pad_id: int, n_type: int):
        self.pad_id, self.n_type = pad_id, n_type

    def __call__(self, batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        input_ids, attn = [], []
        y = torch.zeros(len(batch), self.n_type)
        for i, b in enumerate(batch):
            pad = maxlen - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [self.pad_id] * pad)
            attn.append([1] * len(b["input_ids"]) + [0] * pad)
            y[i, b["type_indices"]] = 1.0
        return torch.tensor(input_ids), torch.tensor(attn), y


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    tot, n, tp, pos = 0.0, 0, 0.0, 0.0
    for input_ids, attn, y in loader:
        logits = model(input_ids.to(device), attn.to(device))
        y = y.to(device)
        tot += float(F.binary_cross_entropy_with_logits(logits, y, reduction="sum")) / y.shape[1]
        n += y.shape[0]
        tp += float(((torch.sigmoid(logits) >= 0.5).float() * y).sum())
        pos += float(y.sum())
    model.train()
    return {"type_loss": tot / max(n, 1), "type_recall": tp / max(pos, 1e-9)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.1)
    ap.add_argument("--learning_rate", type=float, default=1e-4)
    ap.add_argument("--head_learning_rate", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--num_train_epochs", type=float, default=3.0)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=4)
    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--logging_steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)
    activity_types = json.loads((data_dir / "type_vocab.json").read_text())["types"]
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    from peft import LoraConfig
    lora_cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha,
                          lora_dropout=args.lora_dropout,
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                          bias="none", task_type="FEATURE_EXTRACTION")
    model = ActivityModel(args.model, len(activity_types), lora_cfg).to(device)

    train_ds = ActivityDataset(data_dir / "train.jsonl", tok, args.max_length)
    valid_ds = ActivityDataset(data_dir / "valid.jsonl", tok, args.max_length)
    collate = ActivityCollator(tok.pad_token_id, len(activity_types))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate, drop_last=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate)
    print(f"activity types {len(activity_types)} ({', '.join(activity_types)}) | "
          f"train {len(train_ds)} | valid {len(valid_ds)}")

    lora_params = [p for p in model.llm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        [{"params": lora_params, "lr": args.learning_rate},
         {"params": list(model.act_head.parameters()), "lr": args.head_learning_rate}],
        weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    total_steps = max(int(steps_per_epoch * args.num_train_epochs), 1)
    sched = get_cosine_schedule_with_warmup(opt, int(total_steps * args.warmup_ratio), total_steps)
    print(f"optimizer steps: {total_steps} ({steps_per_epoch}/epoch)")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train_args.json").write_text(json.dumps(vars(args), indent=2))

    def checkpoint(m):
        nonlocal best
        if -m["type_loss"] > best:
            best = -m["type_loss"]
            model.save(str(out_dir))
            print(f"  saved (best valid activity loss {m['type_loss']:.4f})", flush=True)

    best, step, running, n_running, history, t0 = float("-inf"), 0, 0.0, 0, [], time.time()
    accum, n_batches = args.gradient_accumulation_steps, len(train_loader)
    model.train()
    for _ in range(math.ceil(args.num_train_epochs)):
        for i, (input_ids, attn, y) in enumerate(train_loader):
            group_start = i - i % accum
            group_size = min(accum, n_batches - group_start)
            logits = model(input_ids.to(device), attn.to(device))
            loss = F.binary_cross_entropy_with_logits(logits, y.to(device))
            (loss / group_size).backward()
            running += float(loss.detach())
            n_running += 1
            if i + 1 == group_start + group_size:
                torch.nn.utils.clip_grad_norm_(
                    [p for g in opt.param_groups for p in g["params"]], 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % args.logging_steps == 0:
                    print(f"step {step}/{total_steps} | activity loss {running / max(n_running, 1):.4f} "
                          f"| lr {sched.get_last_lr()[0]:.2e} "
                          f"| {(time.time() - t0) / 60:.1f} min", flush=True)
                    running, n_running = 0.0, 0
                if args.eval_every and step % args.eval_every == 0:
                    m = evaluate(model, valid_loader, device)
                    print(f"  [valid @ {step}] activity loss {m['type_loss']:.4f} "
                          f"recall {m['type_recall']:.3f}", flush=True)
                    history.append({"step": step, **m})
                    checkpoint(m)
                if step >= total_steps:
                    break
        if step >= total_steps:
            break

    m = evaluate(model, valid_loader, device)
    print(f"[final valid] activity loss {m['type_loss']:.4f} recall {m['type_recall']:.3f}")
    history.append({"step": step, **m})
    checkpoint(m)
    (out_dir / "valid_history.json").write_text(json.dumps(history, indent=2))
    print(f"Done in {(time.time() - t0) / 60:.1f} min. -> {out_dir}")


if __name__ == "__main__":
    main()
