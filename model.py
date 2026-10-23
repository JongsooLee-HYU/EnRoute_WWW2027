from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

STATE_FILE = "enroute.pt"


def build_llm(base_model: str, lora_cfg=None, gradient_checkpointing: bool = True):
    llm = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.bfloat16,
                                               device_map=None)
    llm.config.use_cache = False
    if gradient_checkpointing:
        llm.gradient_checkpointing_enable()
        llm.enable_input_require_grads()
    if lora_cfg is not None:
        from peft import get_peft_model
        llm = get_peft_model(llm, lora_cfg)
        llm.print_trainable_parameters()
    return llm


def encode(llm, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    out = llm(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    H = out.hidden_states[-1]
    a = attention_mask.unsqueeze(-1).to(H.dtype)
    return (H * a).sum(dim=1) / a.sum(dim=1).clamp_min(1e-6)


class ActivityModel(nn.Module):
    def __init__(self, base_model: str, n_activity: int, lora_cfg):
        super().__init__()
        self.llm = build_llm(base_model, lora_cfg, True)
        self.act_head = nn.Linear(self.llm.config.hidden_size, n_activity, dtype=torch.float32)
        nn.init.normal_(self.act_head.weight, std=0.02)
        nn.init.zeros_(self.act_head.bias)

    def forward(self, input_ids, attention_mask) -> torch.Tensor:
        h = encode(self.llm, input_ids, attention_mask).to(self.act_head.weight.dtype)
        return self.act_head(h)

    def save(self, out_dir: str) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.llm.save_pretrained(str(out / "lora"))
        torch.save(self.act_head.state_dict(), out / "activity_head.pt")


class EnRoute(nn.Module):
    def __init__(self, base_model: str, n_poi: int, route_matrix: torch.Tensor,
                 user_emb: Optional[torch.Tensor] = None, lora_cfg=None,
                 gradient_checkpointing: bool = True, proj_dim: int = 256):
        super().__init__()
        llm = build_llm(base_model, lora_cfg, gradient_checkpointing)
        self.llm = llm
        d = llm.config.hidden_size
        f32 = torch.float32
        self.n_poi, self.proj_dim = n_poi, proj_dim

        self.head = nn.Linear(d, n_poi, dtype=f32)
        self.copy_A = nn.Linear(d, proj_dim, bias=False, dtype=f32)
        self.copy_B = nn.Linear(d, proj_dim, bias=False, dtype=f32)
        self.copy_gate = nn.Linear(d, 1, dtype=f32)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)
        nn.init.normal_(self.copy_A.weight, std=0.02)
        nn.init.normal_(self.copy_B.weight, std=0.02)
        nn.init.zeros_(self.copy_gate.weight)
        nn.init.zeros_(self.copy_gate.bias)

        self.mem_theta = nn.Parameter(torch.zeros(2))
        self.mem_b = nn.Parameter(torch.tensor(math.log(math.e - 1)))
        self.mem_log_tau = nn.Parameter(torch.tensor(math.log(30.0)))

        assert route_matrix.shape == (n_poi, n_poi)
        self.register_buffer("route_T", route_matrix.float(), persistent=False)
        self.detour_gate = nn.Linear(d, 1, dtype=f32)
        nn.init.zeros_(self.detour_gate.weight)
        nn.init.zeros_(self.detour_gate.bias)

        self.use_adapter = user_emb is not None
        if self.use_adapter:
            self.register_buffer("user_E", user_emb.float(), persistent=False)
            self.adapter_W = nn.Linear(user_emb.shape[1], d, bias=False, dtype=f32)
            self.adapter_gate = nn.Linear(d, 1, dtype=f32)
            nn.init.normal_(self.adapter_W.weight, std=0.02)
            nn.init.zeros_(self.adapter_gate.weight)
            nn.init.zeros_(self.adapter_gate.bias)

    def temporal_weight(self, memory: torch.Tensor) -> torch.Tensor:
        mask, log_n, delta = memory[..., 0], memory[..., 1], memory[..., 2]
        tau = torch.exp(self.mem_log_tau)
        rec = torch.where(delta >= 0, torch.exp(-delta.clamp_min(0) / tau),
                          torch.zeros_like(delta))
        th_f, th_r = self.mem_theta[0], self.mem_theta[1]
        return F.softplus(th_f * log_n + th_r * rec + self.mem_b) * mask

    def log_detour(self, se: torch.Tensor) -> torch.Tensor:
        V = self.route_T.shape[0]
        s, e = se[:, 0], se[:, 1]
        ok = (s >= 0) & (e >= 0)
        s_, e_ = s.clamp(0, V - 1), e.clamp(0, V - 1)
        direct = self.route_T[s_, e_]
        ok = ok & torch.isfinite(direct)
        delta = (self.route_T[s_] + self.route_T[:, e_].T - direct.unsqueeze(1)).clamp_min(0.0)
        out = torch.log1p(delta)
        finite = torch.isfinite(out)
        worst = torch.where(finite, out, torch.full_like(out, -1.0)).amax(dim=1, keepdim=True)
        out = torch.where(finite, out, worst.clamp_min(0.0))
        return out * ok.unsqueeze(1).to(out.dtype)

    def forward(self, input_ids, attention_mask, memory, se, user_idx
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.use_adapter):
            h = encode(self.llm, input_ids, attention_mask).to(self.head.weight.dtype)
        if self.use_adapter:
            has_u = user_idx.ge(0).to(h.dtype).unsqueeze(1)
            e_u = self.user_E[user_idx.clamp_min(0)]
            h = h + F.softplus(self.adapter_gate(h)) * self.adapter_W(e_u) * has_u
        z = self.head(h)
        g = F.softplus(self.copy_gate(h))
        alpha = (self.copy_A(h) @ self.copy_B(self.head.weight).T) / math.sqrt(self.proj_dim)
        c = g * alpha * self.temporal_weight(memory).to(z.dtype)
        lam = F.softplus(self.detour_gate(h))
        r = -lam * self.log_detour(se).to(z.dtype)
        return z + c + r, g

    def train(self, mode: bool = True) -> "EnRoute":
        super().train(mode)
        if self.use_adapter:
            self.llm.eval()
        return self

    def config(self) -> dict:
        return {"n_poi": self.n_poi, "proj_dim": self.proj_dim, "use_adapter": self.use_adapter}

    def head_state(self) -> dict:
        return {k: v.detach().cpu() for k, v in self.state_dict().items()
                if not k.startswith("llm.")}

    def save(self, out_dir: str) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.llm.save_pretrained(str(out / "lora"))
        torch.save({"config": self.config(), "state": self.head_state()}, out / STATE_FILE)

    def load_lora(self, ckpt_dir: str) -> None:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        weights = Path(ckpt_dir) / "lora" / "adapter_model.safetensors"
        if not weights.exists():
            raise FileNotFoundError(f"no LoRA adapter at {weights}")
        set_peft_model_state_dict(self.llm, load_file(str(weights)))

    def init_from(self, ckpt_dir: str) -> None:
        self.load_lora(ckpt_dir)
        ck = torch.load(Path(ckpt_dir) / STATE_FILE, map_location="cpu")
        missing, unexpected = self.load_state_dict(ck["state"], strict=False)
        bad = [k for k in missing if not k.startswith(("llm.", "adapter_"))]
        if bad or unexpected:
            raise RuntimeError(f"init_from {ckpt_dir}: missing {bad}, unexpected {unexpected}")

    @staticmethod
    def load(ckpt_dir: str, base_model: str, route_matrix: torch.Tensor,
             user_emb: Optional[torch.Tensor] = None, device: str = "cuda") -> "EnRoute":
        ck = torch.load(Path(ckpt_dir) / STATE_FILE, map_location="cpu")
        c = ck["config"]
        if c["use_adapter"] and user_emb is None:
            raise ValueError(f"{ckpt_dir} has a user adapter: pass the LightGCN embeddings")
        model = EnRoute(base_model, c["n_poi"], route_matrix,
                        user_emb if c["use_adapter"] else None,
                        lora_cfg=None, gradient_checkpointing=False, proj_dim=c["proj_dim"])
        from peft import PeftModel
        model.llm = PeftModel.from_pretrained(model.llm, str(Path(ckpt_dir) / "lora"))
        missing, unexpected = model.load_state_dict(ck["state"], strict=False)
        bad = [k for k in missing if not k.startswith("llm.")]
        if bad or unexpected:
            raise RuntimeError(f"{ckpt_dir}: missing {bad}, unexpected {unexpected}")
        model.llm.config.use_cache = False
        return model.to(device).eval()
