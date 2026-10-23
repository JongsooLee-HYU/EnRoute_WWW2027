#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from asl import AsymmetricLoss
from data import (PoiVocab, WaypointCollator, WaypointDataset, load_route_matrix,
                  load_user_embeddings)
from metrics import summarize
from model import EnRoute

STAGE3_TRAINABLE = ("adapter_W", "adapter_gate", "mem_theta", "mem_b", "mem_log_tau")


def run_model(model, batch, device):
    return model(batch["input_ids"].to(device), batch["attention_mask"].to(device),
                 batch["memory"].to(device), batch["se"].to(device),
                 batch["user_idx"].to(device))


@torch.no_grad()
def evaluate(model, loader, loss_fn, vocab, device, top_n=20):
    model.eval()
    tot = gsum = 0.0
    n = 0
    rows = []
    for batch in loader:
        logits, g = run_model(model, batch, device)
        b = logits.shape[0]
        tot += float(loss_fn(logits, batch["labels"].to(device).to(logits.dtype))) * b
        gsum += float(g.sum())
        n += b
        for idxs, gold in zip(logits.topk(top_n, dim=-1).indices.tolist(), batch["gold"]):
            rows.append({"scope": "trajectory", "gold": gold,
                         "ranked_predictions": vocab.decode(idxs)})
    model.train()
    m = summarize(rows, (1, 5, 20), 20)
    m["loss"] = tot / max(n, 1)
    m["copy_gate_mean"] = gsum / max(n, 1)
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, choices=[2, 3], required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--route_matrix", required=True, help="OSRM travel-time npz")
    ap.add_argument("--init_lora", default=None, help="stage 2: Stage-1 checkpoint dir")
    ap.add_argument("--init_from", default=None, help="stage 3: Stage-2 checkpoint dir")
    ap.add_argument("--user_emb", default=None, help="stage 3: LightGCN npz (user_ids, U)")
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--proj_dim", type=int, default=256)
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
    ap.add_argument("--asl_gamma_neg", type=float, default=4.0)
    ap.add_argument("--asl_gamma_pos", type=float, default=1.0)
    ap.add_argument("--asl_clip", type=float, default=0.05)
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--logging_steps", type=int, default=20)
    ap.add_argument("--save_every", type=int, default=200,
                    help="resumable checkpoint every N optimizer steps; 0 disables")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.stage == 2 and not args.init_lora:
        ap.error("--stage 2 needs --init_lora (Stage-1 dir)")
    if args.stage == 3 and not (args.init_from and args.user_emb):
        ap.error("--stage 3 needs --init_from (Stage-2 dir) and --user_emb")
    if (args.stage == 2 and (args.init_from or args.user_emb)) or (args.stage == 3 and args.init_lora):
        ap.error("stage 2 takes --init_lora; stage 3 takes --init_from and --user_emb")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)
    vocab = PoiVocab.load(data_dir / "poi_vocab.json")
    n_poi = len(vocab)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    route_T = load_route_matrix(args.route_matrix)
    user2row, user_E = load_user_embeddings(args.user_emb) if args.user_emb else ({}, None)

    from peft import LoraConfig
    lora_cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha,
                          lora_dropout=args.lora_dropout,
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                          bias="none", task_type="FEATURE_EXTRACTION")
    model = EnRoute(args.model, n_poi, route_T, user_E, lora_cfg, True,
                    proj_dim=args.proj_dim).to(device)
    if args.stage == 2:
        model.load_lora(args.init_lora)
        print(f"LoRA initialised from {args.init_lora}")
    if args.stage == 3:
        model.init_from(args.init_from)
        for name, prm in model.named_parameters():
            prm.requires_grad = name.startswith(STAGE3_TRAINABLE)
        print(f"Stage 3: initialised from {args.init_from}; trainable: "
              + ", ".join(n for n, p in model.named_parameters() if p.requires_grad))

    train_ds = WaypointDataset(data_dir / "train.jsonl", tok, args.max_length, vocab, user2row)
    valid_ds = WaypointDataset(data_dir / "valid.jsonl", tok, args.max_length, vocab, user2row)
    collate = WaypointCollator(tok.pad_token_id, n_poi)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate, drop_last=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate)
    print(f"stage {args.stage} | POI vocab {n_poi} | train {len(train_ds)} | valid {len(valid_ds)}"
          + (f" | users with embedding {len(user2row)}" if args.stage == 3 else ""))

    loss_fn = AsymmetricLoss(args.asl_gamma_neg, args.asl_gamma_pos, args.asl_clip)
    lora_params = [p for p in model.llm.parameters() if p.requires_grad]
    head_params = [p for n, p in model.named_parameters()
                   if not n.startswith("llm.") and p.requires_grad]
    opt = torch.optim.AdamW([{"params": lora_params, "lr": args.learning_rate},
                             {"params": head_params, "lr": args.head_learning_rate}],
                            weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    total_steps = int(steps_per_epoch * args.num_train_epochs)
    sched = get_cosine_schedule_with_warmup(opt, int(total_steps * args.warmup_ratio), total_steps)
    print(f"optimizer steps: {total_steps} ({steps_per_epoch}/epoch)")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train_args.json").write_text(json.dumps(vars(args), indent=2))
    resume_dir = out_dir / "resume"
    step, running, best, history, elapsed0, t0 = 0, 0.0, float("-inf"), [], 0.0, time.time()

    def save_resume():
        resume_dir.mkdir(parents=True, exist_ok=True)
        trainable = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
        torch.save({"params": trainable, "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "step": step, "best": best, "history": history,
                    "elapsed": elapsed0 + (time.time() - t0)}, resume_dir / "state.pt.tmp")
        (resume_dir / "state.pt.tmp").replace(resume_dir / "state.pt")

    if (resume_dir / "state.pt").exists():
        st = torch.load(resume_dir / "state.pt", map_location="cpu", weights_only=False)
        own = dict(model.named_parameters())
        with torch.no_grad():
            for n, v in st["params"].items():
                own[n].copy_(v.to(own[n].dtype))
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        step, best, history, elapsed0 = st["step"], st["best"], st["history"], st["elapsed"]
        print(f"RESUMED from step {step} (best NDCG@20 {best:.4f})", flush=True)

    def validate():
        nonlocal best
        m = evaluate(model, valid_loader, loss_fn, vocab, device)
        print(f"  [valid @ {step}] loss {m['loss']:.4f} copy-gate {m['copy_gate_mean']:.3f} "
              f"Recall@20 {m['Recall@20']:.4f} NDCG@20 {m['NDCG@20']:.4f}", flush=True)
        history.append({"step": step, **m})
        if m["NDCG@20"] > best:
            best = m["NDCG@20"]
            model.save(str(out_dir))
            print(f"  saved (best NDCG@20 {best:.4f})", flush=True)

    accum, n_batches, n_running = args.gradient_accumulation_steps, len(train_loader), 0
    model.train()
    for _ in range(math.ceil(args.num_train_epochs)):
        for i, batch in enumerate(train_loader):
            group_start = i - i % accum
            group_size = min(accum, n_batches - group_start)
            logits, _ = run_model(model, batch, device)
            loss = loss_fn(logits, batch["labels"].to(device).to(logits.dtype))
            (loss / group_size).backward()
            running += loss.detach().item()
            n_running += 1
            if i + 1 == group_start + group_size:
                torch.nn.utils.clip_grad_norm_(
                    [p for g in opt.param_groups for p in g["params"]], 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % args.logging_steps == 0:
                    print(f"step {step}/{total_steps} | loss {running / max(n_running, 1):.4f} "
                          f"| lr {sched.get_last_lr()[0]:.2e} "
                          f"| {(elapsed0 + time.time() - t0) / 60:.1f} min", flush=True)
                    running, n_running = 0.0, 0
                if args.eval_every and step % args.eval_every == 0:
                    validate()
                if args.save_every and step % args.save_every == 0:
                    save_resume()
                if step >= total_steps:
                    break
        if step >= total_steps:
            break

    validate()
    (out_dir / "valid_history.json").write_text(json.dumps(history, indent=2))
    shutil.rmtree(resume_dir, ignore_errors=True)
    print(f"Done in {(elapsed0 + time.time() - t0) / 60:.1f} min. -> {out_dir}")


if __name__ == "__main__":
    main()
