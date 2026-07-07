from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HypergraphMMConfig:
    n_roi: int = 68
    n_query: int = 8
    d_model: int = 128
    n_hg_layers: int = 2
    ffn_mult: int = 4
    dropout: float = 0.1


    predict_amy: bool = True
    predict_tau_global: bool = True
    predict_tau_roi: bool = True


    d_plasma_in: int = 1
    d_apoe_in: int = 1
    d_demo_in: int = 3


class MLP(nn.Module):
    def __init__(self, din: int, dout: int, hidden: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        if hidden is None:
            hidden = max(dout, din * 2)
        self.net = nn.Sequential(
            nn.Linear(din, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AttnPool(nn.Module):
    """Attention pooling over tokens X (B,K,D)."""
    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)
        self.scorer = nn.Linear(d_model, 1, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        s = torch.tanh(self.proj(x))                  # (B,K,D)
        logits = self.scorer(s).squeeze(-1)           # (B,K)
        if mask is not None:
            logits = logits.masked_fill(~mask, -1e9)
        a = torch.softmax(logits, dim=1)              # (B,K)
        a = self.drop(a)
        h = torch.sum(x * a.unsqueeze(-1), dim=1)     # (B,D)
        return h, a


class HypergraphLayer(nn.Module):
    """
    Hypergraph message passing with soft incidence matrices:
      ROI nodes T: (B,R,D)
      Mod nodes U: (B,M,D)  (M=3: plasma/apoe/demo)
      Query nodes E: (B,K,D)

    Incidence:
      H: (B,R,K) ROI->Query
      G: (B,M,K) Mod->Query (learned gating)
    """
    def __init__(self, d_model: int, ffn_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)

        self.roi_to_q = nn.Linear(d_model, d_model, bias=False)
        self.mod_to_q = nn.Linear(d_model, d_model, bias=False)

        self.q_to_roi = nn.Linear(d_model, d_model, bias=False)
        self.q_to_mod = nn.Linear(d_model, d_model, bias=False)

        self.ln_E = nn.LayerNorm(d_model)
        self.ln_T = nn.LayerNorm(d_model)
        self.ln_U = nn.LayerNorm(d_model)

        self.ffn_E = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mult * d_model, d_model),
        )
        self.ffn_T = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mult * d_model, d_model),
        )
        self.ffn_U = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mult * d_model, d_model),
        )

    def forward(self, T, U, E, H, G):
        # ROI -> Query
        msg_E_from_T = torch.einsum("bkr,brd->bkd", H.transpose(1, 2), self.roi_to_q(T))  # (B,K,D)
        # Mod -> Query
        msg_E_from_U = torch.einsum("bkm,bmd->bkd", G.transpose(1, 2), self.mod_to_q(U))  # (B,K,D)

        E = self.ln_E(E + self.drop(F.gelu(msg_E_from_T + msg_E_from_U)))
        E = self.ln_E(E + self.drop(self.ffn_E(E)))

        # Query -> ROI
        msg_T = torch.einsum("brk,bkd->brd", H, self.q_to_roi(E))  # (B,R,D)
        T = self.ln_T(T + self.drop(F.gelu(msg_T)))
        T = self.ln_T(T + self.drop(self.ffn_T(T)))

        # Query -> Mod
        msg_U = torch.einsum("bmk,bkd->bmd", G, self.q_to_mod(E))  # (B,M,D)
        U = self.ln_U(U + self.drop(F.gelu(msg_U)))
        U = self.ln_U(U + self.drop(self.ffn_U(U)))

        return T, U, E


class HypergraphMMNet(nn.Module):

    def __init__(self, cfg: HypergraphMMConfig):
        super().__init__()
        self.cfg = cfg
        R, K, D = cfg.n_roi, cfg.n_query, cfg.d_model

        # ROI scalar -> embedding (per-ROI affine)
        self.roi_w = nn.Parameter(torch.randn(R, D) * 0.02)
        self.roi_b = nn.Parameter(torch.zeros(R, D))

        # Modality encoders -> D
        self.plasma_mlp = MLP(cfg.d_plasma_in, D, dropout=cfg.dropout)
        self.apoe_mlp = MLP(cfg.d_apoe_in, D, dropout=cfg.dropout)
        self.demo_mlp = MLP(cfg.d_demo_in, D, dropout=cfg.dropout)

        # Query node initial embeddings
        self.E0 = nn.Parameter(torch.randn(K, D) * 0.02)

        # ROI->Query incidence prototypes
        self.Qproto = nn.Parameter(torch.randn(K, D) * 0.02)
        self.qproj = nn.Linear(D, D, bias=False)

        # Mod->Query gating
        self.mod2K = nn.Linear(D, K)

        # HG layers
        self.hg_layers = nn.ModuleList([
            HypergraphLayer(D, ffn_mult=cfg.ffn_mult, dropout=cfg.dropout)
            for _ in range(cfg.n_hg_layers)
        ])

        # Pool over queries
        self.pool = AttnPool(D, dropout=cfg.dropout)

        # Heads (keep key names compatible with your original train.py)
        self.head_tauG = nn.Linear(D, 1) if cfg.predict_tau_global else None
        self.head_amy = nn.Linear(D, 1) if cfg.predict_amy else None
        self.head_tauROI = nn.Linear(D, 1) if cfg.predict_tau_roi else None

    def _roi_tokens(self, x_mri: torch.Tensor) -> torch.Tensor:
        return x_mri.unsqueeze(-1) * self.roi_w.unsqueeze(0) + self.roi_b.unsqueeze(0)

    def _mod_tokens(self, x_plasma, x_apoe, x_demo) -> torch.Tensor:
        u_pl = self.plasma_mlp(x_plasma)
        u_ap = self.apoe_mlp(x_apoe)
        u_de = self.demo_mlp(x_demo)
        return torch.stack([u_pl, u_ap, u_de], dim=1)  # (B,3,D)

    def _build_H(self, T: torch.Tensor) -> torch.Tensor:
        Tp = self.qproj(T)
        logits = torch.einsum("brd,kd->brk", Tp, self.Qproto) / (self.cfg.d_model ** 0.5)
        return torch.softmax(logits, dim=-1)

    def _build_G(self, U: torch.Tensor, avail: Optional[torch.Tensor]) -> torch.Tensor:
        logits = self.mod2K(U)  # (B,3,K)
        G = torch.softmax(logits, dim=-1)
        if avail is not None:
            G = G * avail.unsqueeze(-1)
        return G

    def forward(
        self,
        x_mri: torch.Tensor,
        x_plasma: torch.Tensor,
        x_apoe: torch.Tensor,
        x_demo: torch.Tensor,
        avail_plasma: Optional[torch.Tensor] = None,
        avail_apoe: Optional[torch.Tensor] = None,
        avail_demo: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ) -> Dict[str, torch.Tensor]:

        B = x_mri.size(0)

        T = self._roi_tokens(x_mri)                 # (B,R,D)
        U = self._mod_tokens(x_plasma, x_apoe, x_demo)  # (B,3,D)

        avail = None
        if (avail_plasma is not None) and (avail_apoe is not None) and (avail_demo is not None):
            avail = torch.stack([avail_plasma, avail_apoe, avail_demo], dim=1).to(U.dtype)  # (B,3)

        H = self._build_H(T)                        # (B,R,K)
        G = self._build_G(U, avail)                 # (B,3,K)

        E = self.E0.unsqueeze(0).expand(B, -1, -1)  # (B,K,D)

        for layer in self.hg_layers:
            T, U, E = layer(T, U, E, H, G)

        h, pool_attn = self.pool(E)

        out: Dict[str, torch.Tensor] = {}

        if self.head_tauG is not None:
            out["tau_global_logit"] = self.head_tauG(h).squeeze(-1)

        if self.head_amy is not None:
            out["amy_logit"] = self.head_amy(h).squeeze(-1)

        if self.head_tauROI is not None:
            out["tau_roi_logits"] = self.head_tauROI(T).squeeze(-1)

        if return_attn:
            out["H_roi_to_query"] = H
            out["G_mod_to_query"] = G
            out["pool_attn"] = pool_attn
            # compatibility for your previous export function (B,1,K,R)
            out["attn_query_to_roi"] = H.permute(0, 2, 1).unsqueeze(1)

        return out
