
from __future__ import annotations

import copy
import os
import time
from typing import Any, Callable, Dict, Optional, Sequence, Union

import numpy as np
from tqdm import tqdm
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from ._tuning import (
    normalize_architectures,
    normalize_options,
    record_resolved_params,
    report_and_prune,
    require_optuna,
    suggest_architecture,
    suggest_categorical,
    suggest_float,
    validate_minimize_study,
    validate_n_jobs,
)


class TextMLPEncoder(nn.Module):
    """Encode a vector representation independently for each segment."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (1024, 256),
        out_dim: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        dims = [int(input_dim), *[int(h) for h in hidden_dims], int(out_dim)]
        if any(d <= 0 for d in dims):
            raise ValueError("Text encoder dimensions must all be positive.")

        layers: list[nn.Module] = []
        for i, (in_size, out_size) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(in_size, out_size))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Video3DEncoder(nn.Module):
    """Encode a pooled video latent tensor ``[C, D, H, W]`` per segment."""

    def __init__(
        self,
        in_channels: int = 1,
        channels: Sequence[int] = (8, 16, 32),
        out_dim: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        dims = [int(in_channels), *[int(c) for c in channels]]
        if len(dims) < 2 or any(d <= 0 for d in dims) or int(out_dim) <= 0:
            raise ValueError("Video encoder dimensions must all be positive.")

        layers: list[nn.Module] = []
        for i, (in_size, out_size) in enumerate(zip(dims[:-1], dims[1:])):
            layers.extend(
                [
                    nn.Conv3d(in_size, out_size, kernel_size=3, padding=1),
                    nn.ReLU(),
                ]
            )
            if i < len(dims) - 2:
                layers.append(nn.MaxPool3d(kernel_size=2, stride=2))
        layers.append(nn.AdaptiveAvgPool3d((1, 1, 1)))
        self.conv = nn.Sequential(*layers)
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(dims[-1], int(out_dim)),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.conv(x))


class DynamicTarNetBase(nn.Module):
    """
    Dynamic TarNet-style model used to learn a deconfounder representation and an
    outcome model.
    """

    def __init__(
        self,
        sizes_z: Sequence[int],
        sizes_y: Sequence[int],
        outcome_dim: int = 1,
        dropout: float | None = None,
        bn: bool = False,
        include_Si_in_head: bool = True,
        include_Si_in_rep: bool = False,
        include_C_in_head: bool = True,
        include_C_in_rep: bool = False,
        multimodal: bool = False,
        text_input_dim: int | None = None,
        text_hidden_dims: Sequence[int] = (1024, 256),
        text_out_dim: int = 128,
        video_in_channels: int = 1,
        video_channels: Sequence[int] = (8, 16, 32),
        video_out_dim: int = 128,
    ):
        super().__init__()
        sizes_z = tuple(int(width) for width in sizes_z)
        sizes_y = tuple(int(width) for width in sizes_y)
        if not sizes_z or any(width <= 0 for width in sizes_z):
            raise ValueError("sizes_z must contain positive layer widths.")
        if not sizes_y or any(width <= 0 for width in sizes_y):
            raise ValueError("sizes_y must contain positive layer widths.")
        if sizes_y[-1] != int(outcome_dim):
            raise ValueError(
                f"The final sizes_y width must equal outcome_dim={outcome_dim}."
            )
        self.bn = bn
        self.dropout = dropout
        self.include_Si_in_head = include_Si_in_head
        self.include_Si_in_rep = include_Si_in_rep
        self.include_C_in_head = include_C_in_head
        self.include_C_in_rep = include_C_in_rep
        self.multimodal = bool(multimodal)

        encoder_dropout = 0.0 if dropout is None else float(dropout)
        if self.multimodal:
            if text_input_dim is None:
                raise ValueError("text_input_dim is required when multimodal=True.")
            self.text_encoder = TextMLPEncoder(
                input_dim=int(text_input_dim),
                hidden_dims=text_hidden_dims,
                out_dim=int(text_out_dim),
                dropout=encoder_dropout,
            )
            self.video_encoder = Video3DEncoder(
                in_channels=int(video_in_channels),
                channels=video_channels,
                out_dim=int(video_out_dim),
                dropout=encoder_dropout,
            )
        else:
            self.text_encoder = None
            self.video_encoder = None

        # Keep the registered module names for checkpoint compatibility while
        # using the same sizes_z/sizes_y convention as TNutil.TarNetBase.
        self.model_fr = self._build_mlp(sizes_z, dropout, bn)
        self.model_outcome = self._build_mlp(
            sizes_y, dropout, bn, last_linear=True
        )

    def _build_mlp(self, layer_sizes, dropout, bn, last_linear=False):
        layers = []
        for i, out_size in enumerate(layer_sizes):
            layers.append(nn.LazyLinear(out_size))
            is_last = i == (len(layer_sizes) - 1)
            if not (is_last and last_linear):
                if bn:
                    layers.append(nn.BatchNorm1d(out_size))
                layers.append(nn.ReLU())
                if dropout is not None:
                    layers.append(nn.Dropout(dropout))
        return nn.Sequential(*layers)

    def forward(
        self,
        r: torch.Tensor,
        w: torch.Tensor,
        mask: torch.Tensor,
        c: Optional[torch.Tensor] = None,
        return_rep: bool = False,
        r_video: Optional[torch.Tensor] = None,
    ):
        if self.multimodal:
            if r_video is None:
                raise ValueError("r_video is required when multimodal=True.")
            if r.ndim != 3:
                raise ValueError(f"Multimodal r must be [B,T,F], got {tuple(r.shape)}")
            if r_video.ndim == 5:
                r_video = r_video.unsqueeze(2)
            if r_video.ndim != 6:
                raise ValueError(
                    "r_video must be [B,T,C,D,H,W] or [B,T,D,H,W]; "
                    f"got {tuple(r_video.shape)}"
                )
            if r_video.shape[:2] != r.shape[:2]:
                raise ValueError(
                    "r and r_video must agree on [B,T]; "
                    f"got {tuple(r.shape)} and {tuple(r_video.shape)}"
                )

            B, T, text_dim = r.shape
            if mask.shape != (B, T) or w.shape != (B, T):
                raise ValueError(
                    f"w and mask must both be [B,T]=({B},{T}); "
                    f"got {tuple(w.shape)} and {tuple(mask.shape)}"
                )
            mask = mask.to(device=r.device)
            mask_bool = mask if mask.dtype == torch.bool else (mask != 0)
            r = r.float().masked_fill(~mask_bool.unsqueeze(-1), 0.0)
            r_video = r_video.float().masked_fill(
                ~mask_bool.view(B, T, 1, 1, 1, 1), 0.0
            )

            text_flat = r.reshape(B * T, text_dim)
            video_flat = r_video.reshape(B * T, *r_video.shape[2:])
            text_emb = self.text_encoder(text_flat).reshape(B, T, -1)
            video_emb = self.video_encoder(video_flat).reshape(B, T, -1)
            r = torch.cat([text_emb, video_emb], dim=-1)
        else:
            if r_video is not None:
                raise ValueError("r_video was provided, but this model is not multimodal.")
            r = r.permute(1, 2, 0).float()  # [B, T, F]
            B, T, _ = r.shape
            if mask.shape != (B, T) or w.shape != (B, T):
                raise ValueError(
                    f"w and mask must both be [B,T]=({B},{T}); "
                    f"got {tuple(w.shape)} and {tuple(mask.shape)}"
                )
            mask = mask.to(device=r.device)
            mask_bool = mask if mask.dtype == torch.bool else (mask != 0)
            r = r.masked_fill(~mask_bool.unsqueeze(-1), 0.0)

        w = w.to(device=r.device, dtype=r.dtype)
        mask_f = mask_bool.to(dtype=r.dtype)

        Si = mask_f.sum(dim=1, keepdim=True)
        s = (torch.arange(T, device=r.device, dtype=r.dtype) + 1.0).view(1, T, 1).expand(B, T, 1)

        c_time = None
        c_static = None
        if c is not None:
            if c.ndim == 3:
                if c.shape[0] == B and c.shape[1] == T:
                    c_time = c
                elif c.shape[1] == B and c.shape[2] == T:
                    c_time = c.permute(1, 2, 0)
                else:
                    raise ValueError(f"c has incompatible shape {tuple(c.shape)} for B={B}, T={T}")
            elif c.ndim == 2:
                if c.shape[0] == B and c.shape[1] == T:
                    c_time = c.unsqueeze(-1)
                elif c.shape[0] == B:
                    c_static = c
                else:
                    raise ValueError(f"c has incompatible shape {tuple(c.shape)} for B={B}, T={T}")
            elif c.ndim == 1:
                if c.shape[0] == B:
                    c_static = c.view(B, 1)
                else:
                    raise ValueError(f"c has incompatible shape {tuple(c.shape)} for B={B}")
            else:
                raise ValueError("c must be 1D, 2D, or 3D.")

            if c_time is not None:
                c_time = c_time.to(device=r.device, dtype=r.dtype).masked_fill(
                    ~mask_bool.unsqueeze(-1), 0.0
                )
            if c_static is not None:
                c_static = c_static.to(device=r.device, dtype=r.dtype)

        r = r.masked_fill(~mask_bool.unsqueeze(-1), 0.0)
        rep_parts = [r, s]
        if self.include_Si_in_rep:
            Si_rep = Si.view(B, 1, 1).expand(B, T, 1)
            rep_parts.append(Si_rep)

        if self.include_C_in_rep:
            if c_time is not None:
                rep_parts.append(c_time)
            elif c_static is not None:
                rep_parts.append(c_static.view(B, 1, -1).expand(B, T, -1))

        x_rep = torch.cat(rep_parts, dim=-1).reshape(B * T, -1)
        h = self.model_fr(x_rep).reshape(B, T, -1)

        h = h * mask_f.unsqueeze(-1)
        w = w.masked_fill(~mask_bool, 0.0)

        h_flat = h.reshape(B, -1)

        head_parts = [h_flat, w]
        if self.include_C_in_head and (c_time is not None):
            head_parts.append(c_time.reshape(B, -1))
        if self.include_C_in_head and (c_static is not None):
            head_parts.append(c_static)
        if self.include_Si_in_head:
            head_parts.append(Si)

        x_out = torch.cat(head_parts, dim=1)
        y_pred = self.model_outcome(x_out)

        if return_rep:
            return y_pred, h, h_flat
        return y_pred


def mse_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    return ((y_true - y_pred) ** 2).mean()


class DynamicTarNet:
    """
    Wrapper for DynamicTarNetBase (sequence inputs with padding + mask).

    Expected user-facing inputs:
      R: [F, N, T] or [N, T, F]
      W: [N, T]
      mask: [N, T]
      Y: [N] or [N,1]

    With ``multimodal=True``, ``R`` is the vector/text modality [N,T,F] and
    ``R_video`` supplies an aligned video modality [N,T,C_video,D,H,W] (or
    [N,T,D,H,W] for a single input channel). The two modalities are encoded and
    fused before the same DynamicTarNet representation and outcome heads are
    applied.

    Optional:
      C (structured confounders):
        - segment-varying: [N,T,C] or [C,N,T] or scalar [N,T]
        - static:       [N,C] or scalar [N]
    """

    def __init__(
        self,
        architecture_y: Sequence[int] = (16, 1),
        architecture_z: Sequence[int] = (64, 32),
        outcome_dim: int = 1,
        epochs: int = 200,
        batch_size: int = 32,
        learning_rate: float = 2e-5,
        dropout: float | None = None,
        bn: bool = False,
        step_size: int | None = None,
        patience: int = 5,
        min_delta: float = 0.01,
        model_dir: str | None = None,
        verbose: bool = True,
        random_state: int = 42,
        include_Si_in_head: bool = True,
        include_Si_in_rep: bool = False,
        include_C_in_head: bool = True,
        include_C_in_rep: bool = False,
        device: str | torch.device | None = "auto",
        multimodal: bool = False,
        text_input_dim: int = 4096,
        text_hidden_dims: Sequence[int] = (1024, 256),
        text_out_dim: int = 128,
        video_in_channels: int = 1,
        video_channels: Sequence[int] = (8, 16, 32),
        video_out_dim: int = 128,
    ):
        if int(outcome_dim) != 1:
            raise ValueError("DynamicTarNet supports one scalar outcome per unit (outcome_dim=1).")
        self.device = _resolve_torch_device(device)
        if verbose:
            print(f"Device: Using {self.device}")

        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.step_size = step_size
        self.patience = patience
        self.min_delta = min_delta
        self.model_dir = model_dir
        self.verbose = verbose
        self.random_state = int(random_state)

        self.architecture_y = tuple(int(h) for h in architecture_y)
        self.architecture_z = tuple(int(h) for h in architecture_z)
        if not self.architecture_z or any(h <= 0 for h in self.architecture_z):
            raise ValueError("architecture_z must contain positive layer widths.")
        if not self.architecture_y or any(h <= 0 for h in self.architecture_y):
            raise ValueError("architecture_y must contain positive layer widths.")
        if self.architecture_y[-1] != int(outcome_dim):
            raise ValueError(
                f"The final architecture_y width must equal outcome_dim={outcome_dim}."
            )
        self.outcome_dim = outcome_dim
        self.dropout = dropout
        self.bn = bn
        self.include_Si_in_head = include_Si_in_head
        self.include_Si_in_rep = include_Si_in_rep
        self.include_C_in_head = include_C_in_head
        self.include_C_in_rep = include_C_in_rep
        self.device_spec = device
        self.multimodal = bool(multimodal)
        self.text_input_dim = int(text_input_dim)
        self.text_hidden_dims = tuple(int(h) for h in text_hidden_dims)
        self.text_out_dim = int(text_out_dim)
        self.video_in_channels = int(video_in_channels)
        self.video_channels = tuple(int(c) for c in video_channels)
        self.video_out_dim = int(video_out_dim)

        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)
        self.model = self._build_model()

        self.optim = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, eps=1e-8)

        self.scheduler = None
        if self.step_size is not None:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optim, mode="min", factor=0.5, patience=int(self.step_size)
            )

        self.train_dataloader = None
        self.valid_dataloader = None
        self.valid_loss = None
        self.best_valid_loss = None
        self._has_c = False

        if self.model_dir is not None and self.model_dir != "" and (not os.path.exists(self.model_dir)):
            raise ValueError(f"model_dir does not exist: {self.model_dir}")

    def _build_model(self) -> DynamicTarNetBase:
        return DynamicTarNetBase(
            sizes_z=self.architecture_z,
            sizes_y=self.architecture_y,
            outcome_dim=self.outcome_dim,
            dropout=self.dropout,
            bn=self.bn,
            include_Si_in_head=self.include_Si_in_head,
            include_Si_in_rep=self.include_Si_in_rep,
            include_C_in_head=self.include_C_in_head,
            include_C_in_rep=self.include_C_in_rep,
            multimodal=self.multimodal,
            text_input_dim=self.text_input_dim,
            text_hidden_dims=self.text_hidden_dims,
            text_out_dim=self.text_out_dim,
            video_in_channels=self.video_in_channels,
            video_channels=self.video_channels,
            video_out_dim=self.video_out_dim,
        ).to(self.device)

    def reset_model(self):
        self.model = self._build_model()
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, eps=1e-8)
        self.scheduler = None
        if self.step_size is not None:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optim, mode="min", factor=0.5, patience=int(self.step_size)
            )

    def _to_tensor(self, x: Union[np.ndarray, torch.Tensor], name: str) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            return torch.from_numpy(np.ascontiguousarray(x))
        if isinstance(x, torch.Tensor):
            return x
        raise ValueError(f"{name} must be np.ndarray or torch.Tensor")

    def _ensure_shapes(
        self,
        R: torch.Tensor,
        W: torch.Tensor,
        Y: Optional[torch.Tensor],
        mask: torch.Tensor,
        C: Optional[torch.Tensor] = None,
        R_video: Optional[torch.Tensor] = None,
    ):
        if R.ndim != 3:
            raise ValueError("R must be a 3D tensor.")
        if W.ndim != 2 or mask.ndim != 2:
            raise ValueError("W and mask must be 2D tensors [N, T].")

        N, T = W.shape
        if mask.shape != (N, T):
            raise ValueError(f"mask must have shape [N,T]={tuple(W.shape)}; got {tuple(mask.shape)}")

        if self.multimodal:
            if R_video is None:
                raise ValueError("R_video is required when multimodal=True.")
            if R.shape[:2] == (N, T):
                pass
            elif R.shape[1:] == (N, T):
                R = R.permute(1, 2, 0).contiguous()
            else:
                raise ValueError(
                    f"Multimodal R must be [N,T,F] or [F,N,T] with N={N}, T={T}; "
                    f"got {tuple(R.shape)}"
                )
            if R.shape[2] != self.text_input_dim:
                raise ValueError(
                    f"R last dimension must equal text_input_dim={self.text_input_dim}; "
                    f"got {R.shape[2]}"
                )

            if R_video.ndim == 5:
                R_video = R_video.unsqueeze(2)
            if R_video.ndim != 6:
                raise ValueError(
                    "R_video must be [N,T,C_video,D,H,W] or [N,T,D,H,W]; "
                    f"got {tuple(R_video.shape)}"
                )
            if R_video.shape[:2] != (N, T):
                raise ValueError(
                    f"R_video must agree with W on [N,T]=({N},{T}); "
                    f"got {tuple(R_video.shape[:2])}"
                )
            if R_video.shape[2] != self.video_in_channels:
                raise ValueError(
                    f"R_video has {R_video.shape[2]} channels, but "
                    f"video_in_channels={self.video_in_channels}."
                )
        else:
            if R_video is not None:
                raise ValueError("R_video requires a model created with multimodal=True.")
            if R.shape[:2] == (N, T):
                R = R.permute(2, 0, 1).contiguous()
            if R.shape[1:] != (N, T):
                raise ValueError(
                    f"R must be [F,N,T] or [N,T,F] with N={N}, T={T}; got {tuple(R.shape)}"
                )

        if Y is not None:
            if Y.ndim == 2 and Y.shape[1] == 1:
                Y = Y.view(-1)
            if Y.ndim != 1 or Y.shape[0] != N:
                raise ValueError(f"Y must be [N] (or [N,1]); got {Y.shape}")

        if C is not None:
            if C.ndim == 3:
                if C.shape[0] == N and C.shape[1] == T:
                    pass
                elif C.shape[1] == N and C.shape[2] == T:
                    C = C.permute(1, 2, 0).contiguous()
                else:
                    raise ValueError(
                        f"C 3D must be [N,T,C] or [C,N,T]; got {tuple(C.shape)} with N={N},T={T}"
                    )
                C = C * mask.to(C.dtype).unsqueeze(-1)

            elif C.ndim == 2:
                if C.shape == (N, T):
                    C = C * mask.to(C.dtype)
                elif C.shape[0] == N:
                    pass
                else:
                    raise ValueError(
                        f"C 2D must be [N,T] (scalar segment-varying) or [N,C] (static); got {tuple(C.shape)}"
                    )

            elif C.ndim == 1:
                if C.shape[0] == N:
                    C = C.view(N, 1)
                else:
                    raise ValueError(f"C 1D must be length N={N}; got {tuple(C.shape)}")
            else:
                raise ValueError("C must be 1D, 2D, or 3D.")

        mask_values = torch.unique(mask.detach().cpu())
        if not all(float(v) in (0.0, 1.0) for v in mask_values):
            raise ValueError("mask must contain only 0/1 (or boolean) values.")
        mask_bool = mask != 0
        if T > 1 and torch.any(mask_bool[:, 1:] & ~mask_bool[:, :-1]):
            raise ValueError("mask must be left-aligned: valid steps cannot follow padding.")
        w_observed = W[mask_bool]
        if torch.any(~torch.isfinite(w_observed)) or torch.any(
            (w_observed != 0) & (w_observed != 1)
        ):
            raise ValueError("W must be finite and binary at every valid segment.")

        R = R.float()
        R_video = R_video.float() if R_video is not None else None
        W = W.masked_fill(~mask_bool, 0).float()
        Y = Y.float() if Y is not None else None
        mask = mask.float()
        C = C.float() if C is not None else None
        return R, R_video, W, Y, mask, C

    def create_dataloaders(
        self,
        r_train: Union[np.ndarray, torch.Tensor],
        r_test: Union[np.ndarray, torch.Tensor],
        w_train: Union[np.ndarray, torch.Tensor],
        w_test: Union[np.ndarray, torch.Tensor],
        y_train: Union[np.ndarray, torch.Tensor],
        y_test: Union[np.ndarray, torch.Tensor],
        mask_train: Union[np.ndarray, torch.Tensor],
        mask_test: Union[np.ndarray, torch.Tensor],
        c_train: Optional[Union[np.ndarray, torch.Tensor]] = None,
        c_test: Optional[Union[np.ndarray, torch.Tensor]] = None,
        *,
        r_video_train: Optional[Union[np.ndarray, torch.Tensor]] = None,
        r_video_test: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ):
        Rtr = self._to_tensor(r_train, "r_train")
        Rte = self._to_tensor(r_test, "r_test")
        Wtr = self._to_tensor(w_train, "w_train")
        Wte = self._to_tensor(w_test, "w_test")
        Ytr = self._to_tensor(y_train, "y_train")
        Yte = self._to_tensor(y_test, "y_test")
        Mtr = self._to_tensor(mask_train, "mask_train")
        Mte = self._to_tensor(mask_test, "mask_test")

        Ctr = self._to_tensor(c_train, "c_train") if (c_train is not None) else None
        Cte = self._to_tensor(c_test, "c_test") if (c_test is not None) else None
        if (Ctr is None) != (Cte is None):
            raise ValueError("c_train and c_test must either both be provided or both be None.")
        Vtr = self._to_tensor(r_video_train, "r_video_train") if r_video_train is not None else None
        Vte = self._to_tensor(r_video_test, "r_video_test") if r_video_test is not None else None

        Rtr, Vtr, Wtr, Ytr, Mtr, Ctr = self._ensure_shapes(Rtr, Wtr, Ytr, Mtr, Ctr, Vtr)
        Rte, Vte, Wte, Yte, Mte, Cte = self._ensure_shapes(Rte, Wte, Yte, Mte, Cte, Vte)

        if not self.multimodal:
            Rtr = Rtr.permute(1, 0, 2).contiguous()
            Rte = Rte.permute(1, 0, 2).contiguous()

        if Ctr is None:
            self._has_c = False
            if self.multimodal:
                train_dataset = TensorDataset(Rtr, Vtr, Wtr, Mtr, Ytr)
                valid_dataset = TensorDataset(Rte, Vte, Wte, Mte, Yte)
            else:
                train_dataset = TensorDataset(Rtr, Wtr, Mtr, Ytr)
                valid_dataset = TensorDataset(Rte, Wte, Mte, Yte)
        else:
            self._has_c = True
            if self.multimodal:
                train_dataset = TensorDataset(Rtr, Vtr, Wtr, Ctr, Mtr, Ytr)
                valid_dataset = TensorDataset(Rte, Vte, Wte, Cte, Mte, Yte)
            else:
                train_dataset = TensorDataset(Rtr, Wtr, Ctr, Mtr, Ytr)
                valid_dataset = TensorDataset(Rte, Wte, Cte, Mte, Yte)

        train_batch_size = min(int(self.batch_size), len(train_dataset))
        drop_last = False
        if self.bn:
            if train_batch_size < 2:
                raise ValueError(
                    "bn=True requires at least two training observations and "
                    "batch_size >= 2."
                )
            drop_last = len(train_dataset) % train_batch_size == 1

        generator = torch.Generator().manual_seed(self.random_state)
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            sampler=RandomSampler(train_dataset, generator=generator),
            drop_last=drop_last,
        )
        self.valid_dataloader = DataLoader(
            valid_dataset, batch_size=self.batch_size, sampler=SequentialSampler(valid_dataset)
        )

    def fit(
        self,
        R: Union[np.ndarray, torch.Tensor],
        Y: Union[np.ndarray, torch.Tensor],
        W: Union[np.ndarray, torch.Tensor],
        mask: Union[np.ndarray, torch.Tensor],
        C: Optional[Union[np.ndarray, torch.Tensor]] = None,
        valid_perc: float = 0.2,
        plot_loss: bool = True,
        *,
        R_video: Optional[Union[np.ndarray, torch.Tensor]] = None,
        epoch_callback: Optional[Callable[[int, float], None]] = None,
    ):
        R = self._to_tensor(R, "R")
        Y = self._to_tensor(Y, "Y")
        W = self._to_tensor(W, "W")
        mask = self._to_tensor(mask, "mask")
        C = self._to_tensor(C, "C") if (C is not None) else None
        R_video = self._to_tensor(R_video, "R_video") if R_video is not None else None

        R, R_video, W, Y, mask, C = self._ensure_shapes(R, W, Y, mask, C, R_video)

        N = W.shape[0]
        idx = np.arange(N)
        idx_train, idx_test = train_test_split(
            idx, test_size=valid_perc, random_state=self.random_state
        )

        if self.multimodal:
            R_train = R[idx_train, :, :]
            R_test = R[idx_test, :, :]
            V_train = R_video[idx_train, ...]
            V_test = R_video[idx_test, ...]
        else:
            R_train = R[:, idx_train, :]
            R_test = R[:, idx_test, :]
            V_train = None
            V_test = None
        W_train = W[idx_train, :]
        W_test = W[idx_test, :]
        Y_train = Y[idx_train]
        Y_test = Y[idx_test]
        M_train = mask[idx_train, :]
        M_test = mask[idx_test, :]

        if C is None:
            C_train = None
            C_test = None
        else:
            if C.ndim == 3:
                C_train = C[idx_train, :, :]
                C_test = C[idx_test, :, :]
            else:
                C_train = C[idx_train, :]
                C_test = C[idx_test, :]

        self.create_dataloaders(
            R_train,
            R_test,
            W_train,
            W_test,
            Y_train,
            Y_test,
            M_train,
            M_test,
            C_train,
            C_test,
            r_video_train=V_train,
            r_video_test=V_test,
        )

        all_training_loss = []
        all_valid_loss = []
        best_loss = float("inf")
        best_state = None
        epochs_no_improve = 0

        for epoch in range(self.epochs):
            self.model.train()
            training_loss_sum = 0.0
            training_count = 0

            pbar = tqdm(total=len(self.train_dataloader), desc=f"Training (Epoch {epoch})") if self.verbose else None

            for batch in self.train_dataloader:
                if pbar is not None:
                    pbar.update(1)

                if self.multimodal and len(batch) == 5:
                    r_b, v_b, w_b, m_b, y_b = batch
                    c_b = None
                elif self.multimodal:
                    r_b, v_b, w_b, c_b, m_b, y_b = batch
                elif len(batch) == 4:
                    r_b, w_b, m_b, y_b = batch
                    v_b = None
                    c_b = None
                else:
                    r_b, w_b, c_b, m_b, y_b = batch
                    v_b = None

                r_b = r_b.to(self.device)
                if not self.multimodal:
                    r_b = r_b.permute(1, 0, 2).contiguous()
                if v_b is not None:
                    v_b = v_b.to(self.device)
                w_b = w_b.to(self.device)
                m_b = m_b.to(self.device)
                y_b = y_b.to(self.device)
                if c_b is not None:
                    c_b = c_b.to(self.device)

                self.optim.zero_grad()
                y_pred = self.model(r_b, w_b, m_b, c=c_b, r_video=v_b).view(-1)
                loss = mse_loss(y_b, y_pred)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optim.step()
                training_loss_sum += loss.item() * len(y_b)
                training_count += len(y_b)

            if pbar is not None:
                pbar.close()

            valid_loss = float(self.validate_step())
            train_loss = training_loss_sum / max(training_count, 1)

            all_training_loss.append(train_loss)
            all_valid_loss.append(valid_loss)

            if self.verbose:
                print(
                    f"epoch: {epoch} --- train_loss: {train_loss:.6f} "
                    f"--- valid_loss: {valid_loss:.6f}"
                )

            if self.scheduler is not None:
                self.scheduler.step(valid_loss)

            if valid_loss + self.min_delta < best_loss:
                best_loss = valid_loss
                best_state = copy.deepcopy(self.model.state_dict())
                epochs_no_improve = 0
                if self.model_dir is not None and self.model_dir != "":
                    torch.save(self.model.state_dict(), os.path.join(self.model_dir, "best_DynamicTarNet.pth"))
            else:
                epochs_no_improve += 1

            if epoch_callback is not None:
                epoch_callback(epoch, valid_loss)

            if epoch >= 5 and epochs_no_improve >= self.patience:
                if self.verbose:
                    print(f"Early stopping at epoch {epoch}.")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.best_valid_loss = best_loss
        self.valid_loss = torch.tensor(best_loss, dtype=torch.float32)

        if plot_loss:
            import matplotlib.pyplot as plt

            plt.plot(all_training_loss, label="Training Loss")
            plt.plot(all_valid_loss, label="Validation Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.show()

        return float(best_loss)

    def validate_step(self) -> torch.Tensor:
        was_training = self.model.training
        self.model.eval()
        loss_sum = 0.0
        count = 0
        with torch.no_grad():
            pbar = tqdm(total=len(self.valid_dataloader), desc="Validating") if self.verbose else None
            for batch in self.valid_dataloader:
                if pbar is not None:
                    pbar.update(1)

                if self.multimodal and len(batch) == 5:
                    r_b, v_b, w_b, m_b, y_b = batch
                    c_b = None
                elif self.multimodal:
                    r_b, v_b, w_b, c_b, m_b, y_b = batch
                elif len(batch) == 4:
                    r_b, w_b, m_b, y_b = batch
                    v_b = None
                    c_b = None
                else:
                    r_b, w_b, c_b, m_b, y_b = batch
                    v_b = None

                r_b = r_b.to(self.device)
                if not self.multimodal:
                    r_b = r_b.permute(1, 0, 2).contiguous()
                if v_b is not None:
                    v_b = v_b.to(self.device)
                w_b = w_b.to(self.device)
                m_b = m_b.to(self.device)
                y_b = y_b.to(self.device)
                if c_b is not None:
                    c_b = c_b.to(self.device)

                y_pred = self.model(r_b, w_b, m_b, c=c_b, r_video=v_b).view(-1)
                loss = mse_loss(y_b, y_pred)
                loss_sum += loss.item() * len(y_b)
                count += len(y_b)

            if pbar is not None:
                pbar.close()

        if was_training:
            self.model.train()
        self.valid_loss = torch.tensor(
            loss_sum / max(count, 1), dtype=torch.float32
        )
        return self.valid_loss

    def predict(
        self,
        R: Union[np.ndarray, torch.Tensor],
        W: Union[np.ndarray, torch.Tensor],
        mask: Union[np.ndarray, torch.Tensor],
        C: Optional[Union[np.ndarray, torch.Tensor]] = None,
        return_rep: bool = False,
        *,
        R_video: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ):
        R = self._to_tensor(R, "R")
        W = self._to_tensor(W, "W")
        mask = self._to_tensor(mask, "mask")
        C = self._to_tensor(C, "C") if (C is not None) else None
        R_video = self._to_tensor(R_video, "R_video") if R_video is not None else None

        R, R_video, W, _, mask, C = self._ensure_shapes(R, W, None, mask, C, R_video)

        R_ds = R if self.multimodal else R.permute(1, 0, 2).contiguous()

        if C is None:
            dataset = (
                TensorDataset(R_ds, R_video, W, mask)
                if self.multimodal
                else TensorDataset(R_ds, W, mask)
            )
        else:
            dataset = (
                TensorDataset(R_ds, R_video, W, C, mask)
                if self.multimodal
                else TensorDataset(R_ds, W, C, mask)
            )

        dataloader = DataLoader(dataset, batch_size=self.batch_size)

        self.model.eval()
        preds = []
        reps = []
        reps_flat = []

        with torch.no_grad():
            iterator = tqdm(dataloader, total=len(dataloader), desc="Predicting", disable=not self.verbose)
            for batch in iterator:
                if self.multimodal and len(batch) == 4:
                    r_b, v_b, w_b, m_b = batch
                    c_b = None
                elif self.multimodal:
                    r_b, v_b, w_b, c_b, m_b = batch
                elif len(batch) == 3:
                    r_b, w_b, m_b = batch
                    v_b = None
                    c_b = None
                else:
                    r_b, w_b, c_b, m_b = batch
                    v_b = None

                r_b_model = r_b.to(self.device)
                if not self.multimodal:
                    r_b_model = r_b_model.permute(1, 0, 2).contiguous()
                if v_b is not None:
                    v_b = v_b.to(self.device)
                w_b = w_b.to(self.device)
                m_b = m_b.to(self.device)
                if c_b is not None:
                    c_b = c_b.to(self.device)

                if return_rep:
                    y_pred, h, h_flat = self.model(
                        r_b_model,
                        w_b,
                        m_b,
                        c=c_b,
                        return_rep=True,
                        r_video=v_b,
                    )
                    reps.append(h.cpu())
                    reps_flat.append(h_flat.cpu())
                else:
                    y_pred = self.model(r_b_model, w_b, m_b, c=c_b, r_video=v_b)

                preds.append(y_pred.cpu())

        y = torch.cat(preds, dim=0)
        if return_rep:
            h = torch.cat(reps, dim=0)
            h_flat = torch.cat(reps_flat, dim=0)
            return y, h, h_flat
        return y


def _predict_proba_1(clf, X: np.ndarray) -> np.ndarray:
    classes = getattr(clf, "classes_", None)
    if classes is None or len(classes) == 0:
        raise RuntimeError("Classifier appears to be unfitted (classes_ missing).")

    proba = clf.predict_proba(X)

    if 1 in classes:
        j = int(np.where(classes == 1)[0][0])
        return proba[:, j].astype(np.float32)

    return np.zeros(X.shape[0], dtype=np.float32)


class _ConstantRegressor:
    def __init__(self, value: float):
        self.value = float(value)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        return np.full(X.shape[0], self.value, dtype=np.float32)


def _mps_is_available() -> bool:
    return bool(
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_built()
        and torch.backends.mps.is_available()
    )


def _resolve_torch_device(device: str | torch.device | None = "auto") -> torch.device:
    if device is None or str(device).lower() == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _mps_is_available():
            return torch.device("mps")
        return torch.device("cpu")
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA, but torch.cuda.is_available() is False.")
    if dev.type == "mps" and not _mps_is_available():
        built = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_built())
        available = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
        raise RuntimeError(
            "Requested MPS, but torch.backends.mps is not available "
            f"(is_built={built}, is_available={available})."
        )
    return dev


def _predict_torch_batches(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    outs = []
    model.eval()
    with torch.no_grad():
        for lo in range(0, X.shape[0], batch_size):
            xb = torch.from_numpy(X[lo : lo + batch_size]).to(device)
            outs.append(model(xb).squeeze(-1).cpu().numpy())
    return np.concatenate(outs).astype(np.float32)


class _TorchMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_layer_sizes: tuple[int, ...] | list[int],
        out_features: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        layers = []
        prev = int(in_features)
        for h in hidden_layer_sizes:
            h = int(h)
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if float(dropout) > 0:
                layers.append(nn.Dropout(float(dropout)))
            prev = h
        layers.append(nn.Linear(prev, int(out_features)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _normalize_nn_lr_scheduler(scheduler: str | None) -> str:
    value = "none" if scheduler is None else str(scheduler).strip().lower()
    aliases = {
        "": "none",
        "none": "none",
        "off": "none",
        "false": "none",
        "constant": "none",
        "adaptive": "adaptive",
        "plateau": "adaptive",
        "reduce_on_plateau": "adaptive",
    }
    if value not in aliases:
        raise ValueError(
            "nn_lr_scheduler must be one of "
            "{'none', 'constant', 'adaptive', 'reduce_on_plateau'}."
        )
    return aliases[value]


def _make_nn_lr_scheduler(
    opt: torch.optim.Optimizer,
    scheduler: str | None,
    factor: float,
    patience: int,
    min_lr: float,
):
    policy = _normalize_nn_lr_scheduler(scheduler)
    if policy == "none":
        return None
    factor = float(factor)
    if not (0.0 < factor < 1.0):
        raise ValueError("nn_lr_scheduler_factor must be in (0, 1).")
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="min",
        factor=factor,
        patience=max(0, int(patience)),
        min_lr=max(0.0, float(min_lr)),
    )


class _TorchBinaryMLPClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        hidden_layer_sizes=(64, 32),
        alpha=1e-4,
        learning_rate_init=1e-3,
        learning_rate_scheduler="none",
        learning_rate_scheduler_factor=0.5,
        learning_rate_scheduler_patience=2,
        learning_rate_scheduler_min_lr=1e-6,
        max_iter=300,
        batch_size="auto",
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=42,
        dropout=0.0,
        predict_batch_size=8192,
        device="auto",
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.learning_rate_scheduler = learning_rate_scheduler
        self.learning_rate_scheduler_factor = learning_rate_scheduler_factor
        self.learning_rate_scheduler_patience = learning_rate_scheduler_patience
        self.learning_rate_scheduler_min_lr = learning_rate_scheduler_min_lr
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.random_state = random_state
        self.dropout = dropout
        self.predict_batch_size = predict_batch_size
        self.device = device

    def _predict_p1(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError("X must be 2D.")
        if getattr(self, "constant_p1_", None) is not None:
            return np.full(X.shape[0], self.constant_p1_, dtype=np.float32)
        bs = max(1, min(int(self.predict_batch_size), max(1, X.shape[0])))
        logits = _predict_torch_batches(self.model_, X, self.device_, bs)
        return torch.sigmoid(torch.from_numpy(logits)).numpy().astype(np.float32)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).reshape(-1).astype(np.int64)

        if X.ndim != 2:
            raise ValueError("X must be 2D.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows.")
        if X.shape[0] == 0:
            raise ValueError("Cannot fit on empty data.")
        if np.any((y != 0) & (y != 1)):
            raise ValueError("Only binary labels {0,1} are supported.")

        self.classes_ = np.array([0, 1], dtype=np.int64)
        self.n_features_in_ = int(X.shape[1])
        self.device_ = _resolve_torch_device(self.device)

        if X.shape[1] == 0 or np.unique(y).size < 2:
            self.model_ = None
            self.constant_p1_ = float(np.mean(y))
            return self

        torch.manual_seed(int(self.random_state))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(self.random_state))

        X_train = X
        y_train = y.astype(np.float32)
        X_val = None
        y_val = None

        if bool(self.early_stopping):
            n_val = max(1, int(round(X.shape[0] * float(self.validation_fraction))))
            if n_val >= 2 and (X.shape[0] - n_val) >= 1:
                X_train, X_val, y_train, y_val = train_test_split(
                    X,
                    y.astype(np.float32),
                    test_size=float(self.validation_fraction),
                    random_state=int(self.random_state),
                    stratify=y,
                )

        bs = min(200, X_train.shape[0]) if self.batch_size == "auto" else int(self.batch_size)
        bs = max(1, bs)

        train_ds = TensorDataset(
            torch.from_numpy(X_train),
            torch.from_numpy(y_train.reshape(-1, 1)),
        )
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)

        self.model_ = _TorchMLP(
            in_features=X.shape[1],
            hidden_layer_sizes=tuple(int(h) for h in self.hidden_layer_sizes),
            out_features=1,
            dropout=float(self.dropout),
        ).to(self.device_)

        opt = torch.optim.Adam(
            self.model_.parameters(),
            lr=float(self.learning_rate_init),
            weight_decay=float(self.alpha),
        )
        scheduler = _make_nn_lr_scheduler(
            opt,
            self.learning_rate_scheduler,
            self.learning_rate_scheduler_factor,
            self.learning_rate_scheduler_patience,
            self.learning_rate_scheduler_min_lr,
        )
        loss_fn = nn.BCEWithLogitsLoss()

        best_state = None
        best_val = np.inf
        no_improve = 0

        for _ in range(int(self.max_iter)):
            self.model_.train()
            train_loss_sum = 0.0
            train_loss_n = 0
            for xb, yb in train_loader:
                xb = xb.to(self.device_)
                yb = yb.to(self.device_)
                opt.zero_grad(set_to_none=True)
                logits = self.model_(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()
                batch_n = int(xb.shape[0])
                train_loss_sum += float(loss.detach().cpu()) * batch_n
                train_loss_n += batch_n

            scheduler_metric = train_loss_sum / max(1, train_loss_n)
            stop_training = False
            if X_val is not None:
                p1 = np.clip(self._predict_p1(X_val).astype(np.float64), 1e-6, 1 - 1e-6)
                val_loss = float(-np.mean(y_val * np.log(p1) + (1.0 - y_val) * np.log(1.0 - p1)))
                scheduler_metric = val_loss
                if val_loss + 1e-8 < best_val:
                    best_val = val_loss
                    best_state = copy.deepcopy(self.model_.state_dict())
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= int(self.n_iter_no_change):
                        stop_training = True

            if scheduler is not None:
                scheduler.step(float(scheduler_metric))
            if stop_training:
                break

        if best_state is not None:
            self.model_.load_state_dict(best_state)

        self.constant_p1_ = None
        return self

    def predict_proba(self, X):
        p1 = self._predict_p1(X)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1]).astype(np.float32)

    def predict(self, X):
        return (self._predict_p1(X) >= 0.5).astype(np.int64)


class _TorchMLPRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        hidden_layer_sizes=(64, 32),
        alpha=1e-4,
        learning_rate_init=1e-3,
        learning_rate_scheduler="none",
        learning_rate_scheduler_factor=0.5,
        learning_rate_scheduler_patience=2,
        learning_rate_scheduler_min_lr=1e-6,
        max_iter=300,
        batch_size="auto",
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=42,
        dropout=0.0,
        predict_batch_size=8192,
        device="auto",
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.learning_rate_scheduler = learning_rate_scheduler
        self.learning_rate_scheduler_factor = learning_rate_scheduler_factor
        self.learning_rate_scheduler_patience = learning_rate_scheduler_patience
        self.learning_rate_scheduler_min_lr = learning_rate_scheduler_min_lr
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.random_state = random_state
        self.dropout = dropout
        self.predict_batch_size = predict_batch_size
        self.device = device

    def _predict_raw(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError("X must be 2D.")
        if getattr(self, "constant_", None) is not None:
            return np.full(X.shape[0], self.constant_, dtype=np.float32)
        bs = max(1, min(int(self.predict_batch_size), max(1, X.shape[0])))
        return _predict_torch_batches(self.model_, X, self.device_, bs)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1)

        if X.ndim != 2:
            raise ValueError("X must be 2D.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows.")
        if X.shape[0] == 0:
            raise ValueError("Cannot fit on empty data.")

        self.n_features_in_ = int(X.shape[1])
        self.device_ = _resolve_torch_device(self.device)

        if X.shape[1] == 0 or float(np.nanmax(y) - np.nanmin(y)) < 1e-12:
            self.model_ = None
            self.constant_ = float(np.mean(y))
            return self

        torch.manual_seed(int(self.random_state))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(self.random_state))

        X_train = X
        y_train = y
        X_val = None
        y_val = None

        if bool(self.early_stopping):
            n_val = max(1, int(round(X.shape[0] * float(self.validation_fraction))))
            if (X.shape[0] - n_val) >= 1:
                X_train, X_val, y_train, y_val = train_test_split(
                    X,
                    y,
                    test_size=float(self.validation_fraction),
                    random_state=int(self.random_state),
                )

        bs = min(200, X_train.shape[0]) if self.batch_size == "auto" else int(self.batch_size)
        bs = max(1, bs)

        train_ds = TensorDataset(
            torch.from_numpy(X_train),
            torch.from_numpy(y_train.reshape(-1, 1)),
        )
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)

        self.model_ = _TorchMLP(
            in_features=X.shape[1],
            hidden_layer_sizes=tuple(int(h) for h in self.hidden_layer_sizes),
            out_features=1,
            dropout=float(self.dropout),
        ).to(self.device_)

        opt = torch.optim.Adam(
            self.model_.parameters(),
            lr=float(self.learning_rate_init),
            weight_decay=float(self.alpha),
        )
        scheduler = _make_nn_lr_scheduler(
            opt,
            self.learning_rate_scheduler,
            self.learning_rate_scheduler_factor,
            self.learning_rate_scheduler_patience,
            self.learning_rate_scheduler_min_lr,
        )
        loss_fn = nn.MSELoss()

        best_state = None
        best_val = np.inf
        no_improve = 0

        for _ in range(int(self.max_iter)):
            self.model_.train()
            train_loss_sum = 0.0
            train_loss_n = 0
            for xb, yb in train_loader:
                xb = xb.to(self.device_)
                yb = yb.to(self.device_)
                opt.zero_grad(set_to_none=True)
                pred = self.model_(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()
                batch_n = int(xb.shape[0])
                train_loss_sum += float(loss.detach().cpu()) * batch_n
                train_loss_n += batch_n

            scheduler_metric = train_loss_sum / max(1, train_loss_n)
            stop_training = False
            if X_val is not None:
                val_pred = self._predict_raw(X_val).astype(np.float64)
                val_loss = float(np.mean((y_val.astype(np.float64) - val_pred) ** 2))
                scheduler_metric = val_loss
                if val_loss + 1e-8 < best_val:
                    best_val = val_loss
                    best_state = copy.deepcopy(self.model_.state_dict())
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= int(self.n_iter_no_change):
                        stop_training = True

            if scheduler is not None:
                scheduler.step(float(scheduler_metric))
            if stop_training:
                break

        if best_state is not None:
            self.model_.load_state_dict(best_state)

        self.constant_ = None
        return self

    def predict(self, X):
        return self._predict_raw(X)


def _as_hidden_tuple(hidden: tuple[int, ...] | list[int]) -> tuple[int, ...]:
    hidden_t = tuple(int(h) for h in hidden)
    if len(hidden_t) == 0:
        raise ValueError("nn_hidden must be non-empty.")
    return hidden_t


def _fit_downstream_classifier(
    X: np.ndarray,
    y: np.ndarray,
    *,
    nn_hidden: tuple[int, ...] | list[int] = (64, 32),
    nn_alpha: float = 1e-4,
    nn_lr: float = 1e-3,
    nn_lr_scheduler: str = "none",
    nn_lr_scheduler_factor: float = 0.5,
    nn_lr_scheduler_patience: int = 2,
    nn_lr_scheduler_min_lr: float = 1e-6,
    nn_max_iter: int = 300,
    nn_patience: int = 10,
    nn_batch_size: int | str = "auto",
    nn_dropout: float = 0.0,
    random_state: int = 42,
    device: str | torch.device | None = "auto",
):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64).reshape(-1)

    hidden = _as_hidden_tuple(nn_hidden)
    use_es = bool(
        X.shape[0] >= 40
        and np.unique(y).size >= 2
        and np.min(np.bincount(y)) >= 5
    )

    clf = make_pipeline(
        StandardScaler(),
        _TorchBinaryMLPClassifier(
            hidden_layer_sizes=hidden,
            alpha=float(nn_alpha),
            learning_rate_init=float(nn_lr),
            learning_rate_scheduler=nn_lr_scheduler,
            learning_rate_scheduler_factor=float(nn_lr_scheduler_factor),
            learning_rate_scheduler_patience=int(nn_lr_scheduler_patience),
            learning_rate_scheduler_min_lr=float(nn_lr_scheduler_min_lr),
            max_iter=int(nn_max_iter),
            batch_size=nn_batch_size,
            early_stopping=use_es,
            validation_fraction=0.1,
            n_iter_no_change=int(nn_patience),
            random_state=int(random_state),
            dropout=float(nn_dropout),
            device=device,
        ),
    )
    clf.fit(X, y)
    return clf


def _fit_downstream_regressor(
    X: np.ndarray,
    y: np.ndarray,
    *,
    nn_hidden: tuple[int, ...] | list[int] = (64, 32),
    nn_alpha: float = 1e-4,
    nn_lr: float = 1e-3,
    nn_lr_scheduler: str = "none",
    nn_lr_scheduler_factor: float = 0.5,
    nn_lr_scheduler_patience: int = 2,
    nn_lr_scheduler_min_lr: float = 1e-6,
    nn_max_iter: int = 300,
    nn_patience: int = 10,
    nn_batch_size: int | str = "auto",
    nn_dropout: float = 0.0,
    random_state: int = 42,
    device: str | torch.device | None = "auto",
):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32).reshape(-1)

    if y.size == 0:
        raise ValueError("Cannot fit regressor on empty data.")

    if float(np.nanmax(y) - np.nanmin(y)) < 1e-12:
        return _ConstantRegressor(float(y[0]))

    hidden = _as_hidden_tuple(nn_hidden)
    use_es = bool(X.shape[0] >= 40)

    base = make_pipeline(
        StandardScaler(),
        _TorchMLPRegressor(
            hidden_layer_sizes=hidden,
            alpha=float(nn_alpha),
            learning_rate_init=float(nn_lr),
            learning_rate_scheduler=nn_lr_scheduler,
            learning_rate_scheduler_factor=float(nn_lr_scheduler_factor),
            learning_rate_scheduler_patience=int(nn_lr_scheduler_patience),
            learning_rate_scheduler_min_lr=float(nn_lr_scheduler_min_lr),
            max_iter=int(nn_max_iter),
            batch_size=nn_batch_size,
            early_stopping=use_es,
            validation_fraction=0.1,
            n_iter_no_change=int(nn_patience),
            random_state=int(random_state),
            dropout=float(nn_dropout),
            device=device,
        ),
    )
    reg = TransformedTargetRegressor(
        regressor=base,
        transformer=StandardScaler(),
        check_inverse=False,
    )
    reg.fit(X, y)
    return reg


def _history_code(W: np.ndarray, t: int) -> np.ndarray:
    """Encode binary history W[:, :t] as an integer code in [0, 2**t)."""
    if t <= 0:
        return np.zeros(W.shape[0], dtype=np.int64)
    Wb = (W[:, :t] > 0.5).astype(np.int64)
    powers = (1 << np.arange(t, dtype=np.int64)).reshape(1, -1)
    return (Wb * powers).sum(axis=1).astype(np.int64)


def _fit_saturated_prob(
    y: np.ndarray,
    codes: np.ndarray,
    obs: np.ndarray,
    n_codes: int,
) -> np.ndarray:
    """Saturated (cell-mean) estimate of P(y=1 | code), using only obs rows."""
    y_int = y.astype(np.int64)
    sum_y = np.zeros(n_codes, dtype=np.float64)
    cnt = np.zeros(n_codes, dtype=np.float64)
    if obs.sum() > 0:
        np.add.at(sum_y, codes[obs], y_int[obs])
        np.add.at(cnt, codes[obs], 1.0)
    overall = float(sum_y.sum() / max(cnt.sum(), 1.0))
    out = np.full(n_codes, overall, dtype=np.float64)
    np.divide(sum_y, cnt, out=out, where=(cnt > 0))
    return out.astype(np.float32)


def _fit_saturated_mean(
    v: np.ndarray,
    codes: np.ndarray,
    obs: np.ndarray,
    n_codes: int,
) -> np.ndarray:
    """Saturated (cell-mean) estimate of E[v | code], using only obs rows."""
    sum_v = np.zeros(n_codes, dtype=np.float64)
    cnt = np.zeros(n_codes, dtype=np.float64)
    if obs.sum() > 0:
        np.add.at(sum_v, codes[obs], v[obs].astype(np.float64))
        np.add.at(cnt, codes[obs], 1.0)
    overall = float(sum_v.sum() / max(cnt.sum(), 1.0))
    out = np.full(n_codes, overall, dtype=np.float64)
    np.divide(sum_v, cnt, out=out, where=(cnt > 0))
    return out.astype(np.float32)


def _infer_mask_from_w(W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Infer a left-aligned sequence mask and zero-fill treatment padding."""
    W_arr = np.asarray(W, dtype=np.float32)
    if W_arr.ndim != 2:
        raise ValueError(f"W must be [N,T], got {W_arr.shape}")

    mask_arr = (~np.isnan(W_arr)).astype(np.float32)
    if W_arr.shape[1] > 1 and np.any(
        (mask_arr[:, 1:] > 0) & (mask_arr[:, :-1] == 0)
    ):
        raise ValueError(
            "NaN padding in W must be trailing: observed treatment values "
            "cannot follow padding."
        )
    if np.any(mask_arr.sum(axis=1) == 0):
        raise ValueError("Each row of W must contain at least one observed treatment.")

    observed = mask_arr > 0
    if np.any(~np.isfinite(W_arr[observed])) or np.any(
        ~np.isin(W_arr[observed], (0.0, 1.0))
    ):
        raise ValueError("W must be finite and binary at every valid segment.")
    return np.where(observed, W_arr, 0.0).astype(np.float32), mask_arr


class DynamicGPIHyperparameterTuner:
    """Tune the DynamicTarNet outcome network with Optuna.

    The objective is the best held-out factual-outcome MSE reached during each
    trial. Returned ``best_params_`` can be passed directly to
    :func:`estimate_k_ipsi`. Supplying ``R_video`` automatically tunes the
    multimodal architecture.
    """

    def __init__(
        self,
        R: np.ndarray,
        W: np.ndarray,
        Y: np.ndarray,
        C: Optional[np.ndarray] = None,
        *,
        R_video: Optional[np.ndarray] = None,
        nepoch: Any = (100, 200),
        batch_size: Any = (16, 32),
        lr: Any = (1e-5, 1e-3),
        dropout: Any = (0.0, 0.3),
        architecture_y: Any = ((16, 1), (32, 16, 1)),
        architecture_z: Any = ((64, 32), (128, 64)),
        text_hidden_dims: Any = ((1024, 256),),
        text_out_dim: Any = (128,),
        video_channels: Any = ((8, 16, 32),),
        video_out_dim: Any = (128,),
        valid_perc: float = 0.2,
        random_state: int = 42,
        device: str | torch.device | None = "auto",
        model_dir: str | None = None,
        verbose: bool = False,
    ):
        self.W, self.mask = _infer_mask_from_w(W)
        self.Y = np.asarray(Y, dtype=np.float32).reshape(-1)
        n, t_max = self.W.shape
        if len(self.Y) != n:
            raise ValueError(f"Y must have length N={n}; got {self.Y.shape}")
        if not 0 < float(valid_perc) < 1:
            raise ValueError("valid_perc must be between 0 and 1.")

        observed = self.mask > 0
        R_arr = np.asarray(R, dtype=np.float32)
        if R_arr.ndim != 3:
            raise ValueError(f"R must be 3D, got {R_arr.shape}")

        self.multimodal = R_video is not None
        self.R_video = None
        if self.multimodal:
            if R_arr.shape[:2] == (n, t_max):
                pass
            elif R_arr.shape[1:] == (n, t_max):
                R_arr = np.transpose(R_arr, (1, 2, 0)).copy()
            else:
                raise ValueError(
                    f"Multimodal R must be [N,T,F] or [F,N,T]; got {R_arr.shape}"
                )
            if np.any(~np.isfinite(R_arr[observed])):
                raise ValueError("R must be finite at every observed segment.")
            self.R = np.where(observed[:, :, None], R_arr, 0.0).astype(
                np.float32
            )
            self.text_input_dim = int(self.R.shape[2])

            video = np.asarray(R_video, dtype=np.float32)
            if video.ndim == 5:
                video = video[:, :, None, :, :, :]
            if video.ndim != 6 or video.shape[:2] != (n, t_max):
                raise ValueError(
                    "R_video must be [N,T,C_video,D,H,W] or [N,T,D,H,W] "
                    f"aligned with W; got {video.shape}"
                )
            if np.any(~np.isfinite(video[observed])):
                raise ValueError(
                    "R_video must be finite at every observed segment."
                )
            self.R_video = np.where(
                observed[:, :, None, None, None, None], video, 0.0
            ).astype(np.float32)
            self.video_in_channels = int(self.R_video.shape[2])
        else:
            if R_arr.shape[:2] == (n, t_max):
                R_arr = np.transpose(R_arr, (2, 0, 1)).copy()
            if R_arr.shape[1:] != (n, t_max):
                raise ValueError(
                    f"R must be [F,N,T] or [N,T,F] aligned with W; got {R_arr.shape}"
                )
            if np.any(~np.isfinite(R_arr[:, observed])):
                raise ValueError("R must be finite at every observed segment.")
            self.R = np.where(observed[None, :, :], R_arr, 0.0).astype(
                np.float32
            )
            self.text_input_dim = int(self.R.shape[0])
            self.video_in_channels = 1

        self.C = None
        if C is not None:
            C_arr = np.asarray(C, dtype=np.float32)
            if C_arr.ndim == 3:
                if C_arr.shape[:2] == (n, t_max):
                    pass
                elif C_arr.shape[1:] == (n, t_max):
                    C_arr = np.transpose(C_arr, (1, 2, 0)).copy()
                else:
                    raise ValueError(
                        f"C must be [N,T,C] or [C,N,T]; got {C_arr.shape}"
                    )
                if np.any(~np.isfinite(C_arr[observed])):
                    raise ValueError(
                        "C must be finite at every observed segment."
                    )
                self.C = np.where(observed[:, :, None], C_arr, 0.0).astype(
                    np.float32
                )
            elif C_arr.ndim == 2:
                if C_arr.shape == (n, t_max):
                    if np.any(~np.isfinite(C_arr[observed])):
                        raise ValueError(
                            "C must be finite at every observed segment."
                        )
                    self.C = np.where(observed, C_arr, 0.0).astype(np.float32)
                elif C_arr.shape[0] == n:
                    if np.any(~np.isfinite(C_arr)):
                        raise ValueError("Static C must contain only finite values.")
                    self.C = C_arr
                else:
                    raise ValueError(f"C must be [N,T] or [N,C]; got {C_arr.shape}")
            elif C_arr.ndim == 1:
                if C_arr.shape[0] != n:
                    raise ValueError(
                        f"C 1D must have length N={n}; got {C_arr.shape}"
                    )
                if np.any(~np.isfinite(C_arr)):
                    raise ValueError("Static C must contain only finite values.")
                self.C = C_arr.reshape(n, 1)
            else:
                raise ValueError("C must be 1D, 2D, or 3D.")

        self.nepoch_options = normalize_options(
            nepoch, name="nepoch", cast=int
        )
        self.batch_size_options = normalize_options(
            batch_size, name="batch_size", cast=int
        )
        self.lr_options = normalize_options(
            lr, name="lr", cast=float
        )
        self.dropout_options = normalize_options(
            dropout, name="dropout", cast=float
        )
        self.architecture_y_options = normalize_architectures(
            architecture_y, name="architecture_y"
        )
        self.architecture_z_options = normalize_architectures(
            architecture_z, name="architecture_z"
        )
        if any(values[-1] != 1 for values in self.architecture_y_options):
            raise ValueError(
                "Each architecture_y candidate must end in 1 for scalar Y."
            )
        self.text_hidden_dims_options = normalize_architectures(
            text_hidden_dims, name="text_hidden_dims"
        )
        self.text_out_dim_options = normalize_options(
            text_out_dim, name="text_out_dim", cast=int
        )
        self.video_channels_options = normalize_architectures(
            video_channels, name="video_channels"
        )
        self.video_out_dim_options = normalize_options(
            video_out_dim, name="video_out_dim", cast=int
        )

        if self.multimodal:
            min_spatial_size = min(self.R_video.shape[-3:])
            for channels in self.video_channels_options:
                required_size = 2 ** max(0, len(channels) - 1)
                if min_spatial_size < required_size:
                    raise ValueError(
                        f"video_channels={list(channels)} requires each spatial "
                        f"dimension of R_video to be at least {required_size}; "
                        f"got {self.R_video.shape[-3:]}"
                    )

        if any(
            value <= 0
            for value in self.nepoch_options
            + self.batch_size_options
            + self.lr_options
            + self.text_out_dim_options
            + self.video_out_dim_options
        ):
            raise ValueError("Epoch, size, and learning-rate values must be positive.")
        if any(value < 0 or value >= 1 for value in self.dropout_options):
            raise ValueError("dropout values must lie in [0, 1).")

        self.valid_perc = float(valid_perc)
        self.random_state = int(random_state)
        self.device = device
        self.model_dir = model_dir
        self.verbose = bool(verbose)

        self.study_ = None
        self.best_trial_ = None
        self.best_score_ = None
        self.best_params_ = None
        self.best_model_ = None
        self._resolved_params_by_trial: dict[int, dict[str, Any]] = {}

    def _sample_params(self, trial) -> dict[str, Any]:
        params = {
            "architecture_y": suggest_architecture(
                trial, "architecture_y", self.architecture_y_options
            ),
            "architecture_z": suggest_architecture(
                trial, "architecture_z", self.architecture_z_options
            ),
            "nepoch": int(
                suggest_categorical(trial, "nepoch", self.nepoch_options)
            ),
            "batch_size": int(
                suggest_categorical(
                    trial, "batch_size", self.batch_size_options
                )
            ),
            "lr": suggest_float(
                trial, "lr", self.lr_options, log=True
            ),
            "dropout": suggest_float(
                trial, "dropout", self.dropout_options
            ),
        }
        if self.multimodal:
            params.update(
                {
                    "text_hidden_dims": suggest_architecture(
                        trial,
                        "text_hidden_dims",
                        self.text_hidden_dims_options,
                    ),
                    "text_out_dim": int(
                        suggest_categorical(
                            trial, "text_out_dim", self.text_out_dim_options
                        )
                    ),
                    "video_channels": suggest_architecture(
                        trial, "video_channels", self.video_channels_options
                    ),
                    "video_out_dim": int(
                        suggest_categorical(
                            trial, "video_out_dim", self.video_out_dim_options
                        )
                    ),
                    # Fixed input dimensions are included so best_params_ can
                    # be passed directly to estimate_k_ipsi, including for
                    # multi-channel video.
                    "text_input_dim": self.text_input_dim,
                    "video_in_channels": self.video_in_channels,
                }
            )
        return params

    def _build_model(
        self,
        params: dict[str, Any],
        *,
        random_state: int,
        model_dir: str | None,
    ) -> DynamicTarNet:
        model_params = copy.deepcopy(params)
        model_params["epochs"] = model_params.pop("nepoch")
        model_params["learning_rate"] = model_params.pop("lr")
        return DynamicTarNet(
            **model_params,
            outcome_dim=1,
            bn=False,
            patience=5,
            min_delta=0,
            model_dir=model_dir,
            verbose=self.verbose,
            random_state=random_state,
            include_Si_in_head=True,
            include_Si_in_rep=False,
            include_C_in_head=self.C is not None,
            include_C_in_rep=False,
            device=self.device,
            multimodal=self.multimodal,
        )

    def objective(self, trial) -> float:
        trial_number = int(getattr(trial, "number", 0))
        params = self._sample_params(trial)
        self._resolved_params_by_trial[trial_number] = copy.deepcopy(params)
        record_resolved_params(trial, params)

        # Hold the initialization and validation split fixed so trials are
        # compared on the same data rather than on trial-specific holdouts.
        trial_seed = self.random_state
        model = self._build_model(params, random_state=trial_seed, model_dir=None)
        score = model.fit(
            self.R,
            self.Y,
            self.W,
            self.mask,
            C=self.C,
            valid_perc=self.valid_perc,
            plot_loss=False,
            R_video=self.R_video,
            epoch_callback=lambda epoch, loss: report_and_prune(
                trial, loss, epoch
            ),
        )
        if score is None:
            score = model.best_valid_loss
        score = float(score)
        if not np.isfinite(score):
            raise ValueError("DynamicTarNet produced a non-finite validation loss.")
        return score

    def _set_best_from_study(self, study) -> None:
        validate_minimize_study(study)
        best_trial = study.best_trial
        params = getattr(best_trial, "user_attrs", {}).get("resolved_params")
        if params is None:
            params = self._resolved_params_by_trial.get(int(best_trial.number))
        if params is None:
            raise ValueError("The best trial does not contain resolved parameters.")
        self.study_ = study
        self.best_trial_ = best_trial
        self.best_score_ = float(best_trial.value)
        self.best_params_ = copy.deepcopy(params)

    def tune(
        self,
        n_trials: int = 50,
        *,
        timeout: float | None = None,
        study=None,
        sampler=None,
        pruner=None,
        n_jobs: int = 1,
        refit: bool = False,
        **optimize_kwargs,
    ):
        optuna = require_optuna()
        if study is None:
            sampler = sampler or optuna.samplers.TPESampler(seed=self.random_state)
            pruner = pruner or optuna.pruners.MedianPruner(n_startup_trials=5)
            study = optuna.create_study(
                direction="minimize", sampler=sampler, pruner=pruner
            )
        validate_minimize_study(study)
        study.optimize(
            self.objective,
            n_trials=int(n_trials),
            timeout=timeout,
            n_jobs=validate_n_jobs(n_jobs),
            **optimize_kwargs,
        )
        self._set_best_from_study(study)
        if refit:
            self.fit_best()
        return study

    def fit_best(self, study=None) -> DynamicTarNet:
        if study is not None:
            self._set_best_from_study(study)
        if self.best_params_ is None:
            raise RuntimeError("Run tune() or provide a completed study first.")
        self.best_model_ = self._build_model(
            self.best_params_,
            random_state=self.random_state,
            model_dir=self.model_dir,
        )
        self.best_model_.fit(
            self.R,
            self.Y,
            self.W,
            self.mask,
            C=self.C,
            valid_perc=self.valid_perc,
            plot_loss=False,
            R_video=self.R_video,
        )
        return self.best_model_


DynamicTarNetHyperparameterTuner = DynamicGPIHyperparameterTuner


def estimate_k_ipsi(
    R: np.ndarray,
    W: np.ndarray,
    Y: np.ndarray,
    delta_seq: np.ndarray,
    K: int = 5,
    sample_split_only: bool = False,
    sample_split_fold: int = 1,
    architecture_y: Sequence[int] = (16, 1),
    architecture_z: Sequence[int] = (64, 32),
    nepoch: int = 200,
    batch_size: int = 32,
    lr: float = 2e-5,
    dropout: float = 0.3,
    valid_perc: float = 0.2,
    step_size: int | None = None,
    bn: bool = False,
    patience: int = 5,
    min_delta: float = 0,
    verbose: bool = True,
    random_state: int = 42,
    eps_prob: float | None = 1e-6,
    nn_hidden: tuple[int, ...] | list[int] = (64, 32),
    nn_alpha: float = 1e-4,
    nn_lr: float = 1e-3,
    nn_lr_scheduler: str = "none",
    nn_lr_scheduler_factor: float = 0.5,
    nn_lr_scheduler_patience: int = 2,
    nn_lr_scheduler_min_lr: float = 1e-6,
    nn_max_iter: int = 300,
    nn_patience: int = 5,
    nn_batch_size: int | str = "auto",
    nn_dropout: float = 0.0,
    H: Optional[np.ndarray] = None,
    C: Optional[np.ndarray] = None,
    model_dir: str | None = None,
    device: str | torch.device | None = "auto",
    *,
    n_boot: int = 0,
    R_video: Optional[np.ndarray] = None,
    text_input_dim: int | None = None,
    text_hidden_dims: Sequence[int] = (1024, 256),
    text_out_dim: int = 128,
    video_in_channels: int = 1,
    video_channels: Sequence[int] = (8, 16, 32),
    video_out_dim: int = 128,
) -> Dict[str, Any]:
    """Estimate a longitudinal incremental-intervention curve by cross-fitting.

    Parameters
    ----------
    R : np.ndarray
        Per-segment vector representations. Vector-only estimation accepts
        ``[F,N,T]`` or ``[N,T,F]``. With ``R_video``, this is the aligned
        text/vector modality ``[N,T,F]`` (``[F,N,T]`` is also accepted).
    W : np.ndarray
        Binary treatment history ``[N,T]``. Finite 0/1 entries are observed;
        trailing ``NaN`` entries are padding and determine the sequence mask.
    Y : np.ndarray
        One scalar outcome per unit, shape ``[N]`` or ``[N,1]``.
    delta_seq : array-like
        Positive intervention odds multipliers. Supply a scalar or ``[J]`` to
        apply each multiplier at every observed segment, or ``[J,T]`` for
        segment-specific intervention schedules.
    K : int, default=5
        Number of cross-fitting folds.
    sample_split_only : bool, default=False
        If true, estimate only the fold selected by ``sample_split_fold``.
    sample_split_fold : int, default=1
        One-indexed held-out fold used when ``sample_split_only=True``.
    architecture_y : sequence of int, default=(16, 1)
        Outcome-network layer widths. As in :class:`TarNet`, the final width is
        the output dimension and must be 1 for the supported scalar outcome.
    architecture_z : sequence of int, default=(64, 32)
        Shared/deconfounder representation layer widths. The final width is the
        learned representation dimension for each segment.
    nepoch : int, default=200
        Maximum DynamicTarNet training epochs per fold.
    batch_size : int, default=32
        DynamicTarNet training batch size.
    lr : float, default=2e-5
        DynamicTarNet learning rate.
    dropout : float, default=0.3
        Dropout probability in the representation and outcome networks.
    valid_perc : float, default=0.2
        Fraction of each training fold used for neural-network validation.
    step_size : int, optional
        Patience of the reduce-on-plateau learning-rate scheduler. ``None``
        disables that scheduler.
    bn : bool, default=False
        Whether to use batch normalization in DynamicTarNet.
    patience : int, default=5
        Early-stopping patience in epochs.
    min_delta : float, default=0
        Minimum validation-loss improvement used by early stopping.
    verbose : bool, default=True
        Whether to print fold and training progress.
    random_state : int, default=42
        Seed used for fold assignment and neural-network training.
    eps_prob : float or None, default=1e-6
        Optional lower/upper clipping tolerance for estimated probabilities.
    nn_hidden : sequence of int, default=(64, 32)
        Hidden widths for the downstream propensity and regression MLPs.
    nn_alpha : float, default=1e-4
        Weight-decay strength for downstream MLPs.
    nn_lr : float, default=1e-3
        Learning rate for downstream MLPs.
    nn_lr_scheduler : str, default="none"
        Downstream learning-rate policy: ``"none"`` or ``"adaptive"``
        (aliases ``"plateau"`` and ``"reduce_on_plateau"`` are accepted).
    nn_lr_scheduler_factor : float, default=0.5
        Multiplicative factor for the downstream adaptive scheduler.
    nn_lr_scheduler_patience : int, default=2
        Plateau patience for the downstream adaptive scheduler.
    nn_lr_scheduler_min_lr : float, default=1e-6
        Minimum downstream learning rate.
    nn_max_iter : int, default=300
        Maximum downstream MLP training epochs.
    nn_patience : int, default=5
        Early-stopping patience for downstream MLPs.
    nn_batch_size : int or "auto", default="auto"
        Batch size for downstream MLPs.
    nn_dropout : float, default=0
        Dropout probability for downstream MLPs.
    H : np.ndarray, optional
        Precomputed longitudinal representations ``[N,T,d_z]`` or
        ``[d_z,N,T]``. When supplied, DynamicTarNet representation learning is
        skipped.
    C : np.ndarray, optional
        Static confounders ``[N,P]``/``[N]`` or segment-varying confounders
        ``[N,T,P]``/``[P,N,T]``/``[N,T]``.
    model_dir : str, optional
        Existing directory under which fold-specific checkpoints are stored.
    device : str, torch.device, or None, default="auto"
        Device for DynamicTarNet and downstream MLPs.
    n_boot : int, default=0
        Number of Rademacher multiplier-bootstrap draws. A positive value adds
        simultaneous 95% bands ``ll2`` and ``ul2``; both are ``None`` at zero.
    R_video : np.ndarray, optional
        Aligned video representations ``[N,T,D,H,W]`` or
        ``[N,T,C_video,D,H,W]``. Here ``T`` is the common padded number of
        segment positions, while unit ``i`` has ``S_i`` observed segments. For
        an unpooled Cosmos representation, ``D`` is retained latent time within
        each segment. Its presence enables multimodal mode.
    text_input_dim : int, optional
        Text/vector feature width. Inferred from ``R`` when omitted.
    text_hidden_dims : sequence of int, default=(1024, 256)
        Hidden widths of the per-segment text encoder.
    text_out_dim : int, default=128
        Encoded text feature width.
    video_in_channels : int, default=1
        ``C_video`` for six-dimensional ``R_video``; use 1 for five-dimensional
        ``R_video``.
    video_channels : sequence of int, default=(8, 16, 32)
        Channel widths of the 3D video encoder.
    video_out_dim : int, default=128
        Encoded video feature width.

    Returns
    -------
    dict
        ``est`` and ``se`` contain the estimated curve and standard errors;
        ``ll1``/``ul1`` are pointwise 95% intervals; ``ll2``/``ul2`` are
        simultaneous 95% bands when requested; ``ifvals`` is the ``[N,J]``
        influence-function matrix. The dictionary also contains ``delta``,
        expanded ``delta_paths``, ``sigma``, and ``n_eff``.

    Notes
    -----
    Dimension notation is: ``N`` units, ``T`` padded segment positions,
    ``S_i`` observed segments for unit ``i``, ``F`` vector/text features,
    ``C_video`` video channels, ``D`` video depth, ``H``/``W`` latent spatial
    dimensions, ``P`` covariates, ``d_z`` learned representation features,
    ``J`` interventions, and ``K`` cross-fitting folds. For unpooled Cosmos
    features, ``D`` equals latent time within a segment and is distinct from
    the outer segment axis ``T``.
    A finite zero in ``W`` is an observed control treatment, not padding.
    This function supports one scalar outcome per unit; it does not implement
    inference for repeated outcomes shaped ``[N,T]``.
    """

    architecture_y = tuple(int(width) for width in architecture_y)
    architecture_z = tuple(int(width) for width in architecture_z)
    if not architecture_y or any(width <= 0 for width in architecture_y):
        raise ValueError("architecture_y must contain positive layer widths.")
    if architecture_y[-1] != 1:
        raise ValueError("architecture_y must end in 1 for scalar Y.")
    if not architecture_z or any(width <= 0 for width in architecture_z):
        raise ValueError("architecture_z must contain positive layer widths.")
    if int(nepoch) <= 0 or int(batch_size) <= 0:
        raise ValueError("nepoch and batch_size must be positive.")
    if float(lr) <= 0:
        raise ValueError("lr must be positive.")
    if not 0 <= float(dropout) < 1:
        raise ValueError("dropout must lie in [0, 1).")
    if not 0 < float(valid_perc) < 1:
        raise ValueError("valid_perc must lie strictly between 0 and 1.")
    if step_size is not None and int(step_size) <= 0:
        raise ValueError("step_size must be positive or None.")
    if int(patience) <= 0:
        raise ValueError("patience must be positive.")
    if float(min_delta) < 0:
        raise ValueError("min_delta must be nonnegative.")

    if (
        isinstance(n_boot, (bool, np.bool_))
        or not isinstance(n_boot, (int, np.integer))
        or n_boot < 0
    ):
        raise ValueError("n_boot must be a nonnegative integer.")

    use_multimodal = R_video is not None

    if model_dir is not None and model_dir != "" and not os.path.exists(model_dir):
        raise ValueError(f"model_dir does not exist: {model_dir}")

    def _train_tarnet(
        R_sub: np.ndarray,
        W_sub: np.ndarray,
        Y_sub: np.ndarray,
        mask_sub: np.ndarray,
        C_sub_nn: Optional[np.ndarray],
        R_video_sub: Optional[np.ndarray],
        *,
        fold_label: str,
        save_dir: Optional[str] = None,
    ) -> "DynamicTarNet":
        if verbose:
            print(f"  [NN train] {fold_label}: n={W_sub.shape[0]}")

        if save_dir is not None and save_dir != "" and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        model = DynamicTarNet(
            architecture_y=architecture_y,
            architecture_z=architecture_z,
            outcome_dim=1,
            epochs=nepoch,
            batch_size=batch_size,
            learning_rate=lr,
            dropout=dropout,
            bn=bn,
            step_size=step_size,
            patience=patience,
            min_delta=min_delta,
            model_dir=save_dir,
            verbose=verbose,
            random_state=random_state,
            include_Si_in_head=True,
            include_Si_in_rep=False,
            include_C_in_head=(C is not None),
            include_C_in_rep=False,
            device=device,
            multimodal=use_multimodal,
            text_input_dim=int(text_input_dim_resolved),
            text_hidden_dims=text_hidden_dims,
            text_out_dim=text_out_dim,
            video_in_channels=video_in_channels,
            video_channels=video_channels,
            video_out_dim=video_out_dim,
        )

        n_sub = int(W_sub.shape[0])
        valid_perc_nn = valid_perc
        if n_sub < 2:
            raise ValueError(f"Cannot train NN on n={n_sub} samples.")
        if n_sub == 2:
            valid_perc_nn = 0.5

        model.fit(
            R=R_sub,
            Y=Y_sub,
            W=W_sub,
            C=C_sub_nn,
            mask=mask_sub,
            valid_perc=valid_perc_nn,
            plot_loss=False,
            R_video=R_video_sub,
        )
        return model

    def _predict_mu_H(
        model: "DynamicTarNet",
        R_sub: np.ndarray,
        W_sub: np.ndarray,
        mask_sub: np.ndarray,
        C_sub_nn: Optional[np.ndarray],
        R_video_sub: Optional[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        y_hat, H_hat, _ = model.predict(
            R_sub,
            W_sub,
            mask_sub,
            C=C_sub_nn,
            return_rep=True,
            R_video=R_video_sub,
        )
        mu = y_hat.numpy().astype(np.float32).ravel()
        Hn = H_hat.numpy().astype(np.float32)
        return mu, Hn

    def _c_upto(
        C_time: Optional[np.ndarray],
        C_static: Optional[np.ndarray],
        t: int,
        n_rows: int,
    ) -> np.ndarray:
        if (C_time is None) and (C_static is None):
            return np.zeros((n_rows, 0), dtype=np.float32)

        parts = []
        if C_static is not None:
            parts.append(C_static.astype(np.float32, copy=False))

        if C_time is not None:
            if C_time.ndim == 3:
                parts.append(C_time[:, : t + 1, :].reshape(n_rows, -1))
            else:
                parts.append(C_time[:, : t + 1])

        return np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]

    def _fit_pi_p_on_train_predict_test(
        W_tr: np.ndarray,
        mask_tr: np.ndarray,
        H_tr: np.ndarray,
        S_tr: np.ndarray,
        C_tr_time: Optional[np.ndarray],
        C_tr_static: Optional[np.ndarray],
        W_te: np.ndarray,
        mask_te: np.ndarray,
        H_te: np.ndarray,
        S_te: np.ndarray,
        C_te_time: Optional[np.ndarray],
        C_te_static: Optional[np.ndarray],
    ) -> dict:
        n_tr, T_max_local = W_tr.shape
        n_te = W_te.shape[0]

        pi_tr = np.full((n_tr, T_max_local), np.nan, dtype=np.float32)
        pi_te = np.full((n_te, T_max_local), np.nan, dtype=np.float32)
        p_tr = np.full((n_tr, T_max_local), np.nan, dtype=np.float32)
        p_te = np.full((n_te, T_max_local), np.nan, dtype=np.float32)

        codes_tr_all = [_history_code(W_tr, t) for t in range(T_max_local)]
        codes_te_all = [_history_code(W_te, t) for t in range(T_max_local)]

        for t in range(T_max_local):
            train_obs = mask_tr[:, t] > 0
            test_obs = mask_te[:, t] > 0

            if train_obs.sum() == 0:
                continue

            Cpi_tr = _c_upto(C_tr_time, C_tr_static, t, n_tr)
            Cpi_te = _c_upto(C_te_time, C_te_static, t, n_te)

            if t == 0:
                Xpi_tr = np.concatenate([S_tr, H_tr[:, 0, :], Cpi_tr], axis=1)
                Xpi_te = np.concatenate([S_te, H_te[:, 0, :], Cpi_te], axis=1)
            else:
                W_past_tr = W_tr[:, :t]
                H_upto_tr = H_tr[:, : t + 1, :].reshape(n_tr, -1)
                Xpi_tr = np.concatenate([S_tr, W_past_tr, H_upto_tr, Cpi_tr], axis=1)

                W_past_te = W_te[:, :t]
                H_upto_te = H_te[:, : t + 1, :].reshape(n_te, -1)
                Xpi_te = np.concatenate([S_te, W_past_te, H_upto_te, Cpi_te], axis=1)

            y_tr = W_tr[:, t].astype(int)
            y_tr_obs = y_tr[train_obs]

            if np.unique(y_tr_obs).size < 2:
                const_p = float(y_tr_obs[0])
                pi_tr[train_obs, t] = const_p
                if test_obs.sum() > 0:
                    pi_te[test_obs, t] = const_p
            else:
                clf_pi = _fit_downstream_classifier(
                    X=Xpi_tr[train_obs],
                    y=y_tr_obs,
                    nn_hidden=nn_hidden,
                    nn_alpha=nn_alpha,
                    nn_lr=nn_lr,
                    nn_lr_scheduler=nn_lr_scheduler,
                    nn_lr_scheduler_factor=nn_lr_scheduler_factor,
                    nn_lr_scheduler_patience=nn_lr_scheduler_patience,
                    nn_lr_scheduler_min_lr=nn_lr_scheduler_min_lr,
                    nn_max_iter=nn_max_iter,
                    nn_patience=nn_patience,
                    nn_batch_size=nn_batch_size,
                    nn_dropout=nn_dropout,
                    random_state=int(random_state + 5000 * t + 1),
                    device=device,
                )
                pi_tr[train_obs, t] = _predict_proba_1(clf_pi, Xpi_tr[train_obs])
                if test_obs.sum() > 0:
                    pi_te[test_obs, t] = _predict_proba_1(clf_pi, Xpi_te[test_obs])

            if t == 0:
                p0 = float(np.mean(y_tr_obs))
                p_tr[train_obs, t] = p0
                if test_obs.sum() > 0:
                    p_te[test_obs, t] = p0
            else:
                if np.unique(y_tr_obs).size < 2:
                    const_p = float(y_tr_obs[0])
                    p_tr[train_obs, t] = const_p
                    if test_obs.sum() > 0:
                        p_te[test_obs, t] = const_p
                else:
                    n_codes = 2**t
                    p_map = _fit_saturated_prob(y_tr, codes_tr_all[t], train_obs, n_codes)
                    p_tr[train_obs, t] = p_map[codes_tr_all[t][train_obs]]
                    if test_obs.sum() > 0:
                        p_te[test_obs, t] = p_map[codes_te_all[t][test_obs]]

        return {
            "pi_tr": pi_tr,
            "pi_te": pi_te,
            "p_tr": p_tr,
            "p_te": p_te,
            "codes_tr_all": codes_tr_all,
            "codes_te_all": codes_te_all,
        }

    def _fit_m_and_predict_test(
        delta_vec: np.ndarray,
        *,
        W_tr: np.ndarray,
        mask_tr: np.ndarray,
        H_tr: np.ndarray,
        mu_tr: np.ndarray,
        S_tr: np.ndarray,
        C_tr_time: Optional[np.ndarray],
        C_tr_static: Optional[np.ndarray],
        pi_tr: np.ndarray,
        p_tr: np.ndarray,
        codes_tr_all: list[np.ndarray],
        W_te: np.ndarray,
        mask_te: np.ndarray,
        H_te: np.ndarray,
        S_te: np.ndarray,
        C_te_time: Optional[np.ndarray],
        C_te_static: Optional[np.ndarray],
        p_te: np.ndarray,
        codes_te_all: list[np.ndarray],
    ) -> dict:
        n_tr, T_max_local = W_tr.shape
        n_te = W_te.shape[0]

        n_segments_tr = mask_tr.sum(axis=1).astype(int)

        R_tr_rec = np.full((n_tr, T_max_local + 1), np.nan, dtype=np.float32)
        for i in range(n_tr):
            Si = int(n_segments_tr[i])
            if Si > 0:
                R_tr_rec[i, Si] = float(mu_tr[i])

        m1_te_all = np.full((n_te, T_max_local), np.nan, dtype=np.float32)
        m0_te_all = np.full((n_te, T_max_local), np.nan, dtype=np.float32)
        mtilde1_te_all = np.full((n_te, T_max_local), np.nan, dtype=np.float32)
        mtilde0_te_all = np.full((n_te, T_max_local), np.nan, dtype=np.float32)

        omega_prev_tr = np.ones((n_tr, T_max_local), dtype=np.float32)
        prod_omega_tr = np.ones(n_tr, dtype=np.float64)
        clip_eps = float(eps_prob) if (eps_prob is not None and eps_prob > 0) else None

        for s in range(T_max_local):
            omega_prev_tr[:, s] = prod_omega_tr.astype(np.float32)

            obs_s = mask_tr[:, s] > 0
            if obs_s.sum() == 0:
                continue

            W_s = W_tr[obs_s, s].astype(np.float64, copy=False)
            p_s = p_tr[obs_s, s].astype(np.float64, copy=False)
            pi_raw = pi_tr[obs_s, s].astype(np.float64, copy=False)

            if np.any(~np.isfinite(p_s)) or np.any(~np.isfinite(pi_raw)):
                raise RuntimeError(
                    f"Non-finite p/pi for training unit at segment s={s}."
                )

            delta_s = float(delta_vec[s])
            denom_s = delta_s * p_s + 1.0 - p_s

            pi_w1 = pi_raw.copy()
            pi_w0 = 1.0 - pi_raw
            if clip_eps is not None:
                pi_w1 = np.clip(pi_w1, clip_eps, 1.0 - clip_eps)
                pi_w0 = np.clip(pi_w0, clip_eps, 1.0 - clip_eps)
            else:
                pi_w1 = np.where(pi_w1 <= 0.0, 1e-12, pi_w1)
                pi_w0 = np.where(pi_w0 <= 0.0, 1e-12, pi_w0)

            omega_s = np.empty_like(p_s, dtype=np.float64)
            treated = W_s >= 0.5
            omega_s[treated] = (
                delta_s * p_s[treated] / pi_w1[treated]
            ) / denom_s[treated]
            omega_s[~treated] = (
                (1.0 - p_s[~treated]) / pi_w0[~treated]
            ) / denom_s[~treated]

            prod_obs = prod_omega_tr[obs_s]
            prod_obs *= omega_s
            prod_omega_tr[obs_s] = prod_obs

        def _mean_or_fallback(v: np.ndarray, obs: np.ndarray, fallback_obs: np.ndarray) -> float:
            if np.any(obs):
                return float(np.mean(v[obs]))
            if np.any(fallback_obs):
                return float(np.mean(v[fallback_obs]))
            return 0.0

        def _fit_mtilde_saturated(
            y_target: np.ndarray,
            codes: np.ndarray,
            obs: np.ndarray,
            fallback_obs: np.ndarray,
            n_codes: int,
        ) -> np.ndarray:
            fit_obs = obs if np.any(obs) else fallback_obs
            return _fit_saturated_mean(y_target, codes, fit_obs, n_codes)

        for s in range(T_max_local - 1, -1, -1):
            train_obs = mask_tr[:, s] > 0
            test_obs = mask_te[:, s] > 0

            if train_obs.sum() == 0:
                continue

            y_out_tr = R_tr_rec[:, s + 1]
            if not np.all(np.isfinite(y_out_tr[train_obs])):
                raise RuntimeError(
                    f"Non-finite training pseudo-outcomes at segment s={s} "
                    "(segment-wise cross-fit)."
                )

            if s == 0:
                H_upto_tr = H_tr[:, 0, :]
                W_past_tr = np.zeros((n_tr, 0), dtype=np.float32)
                H_upto_te = H_te[:, 0, :]
                W_past_te = np.zeros((n_te, 0), dtype=np.float32)
            else:
                H_upto_tr = H_tr[:, : s + 1, :].reshape(n_tr, -1)
                W_past_tr = W_tr[:, :s]
                H_upto_te = H_te[:, : s + 1, :].reshape(n_te, -1)
                W_past_te = W_te[:, :s]

            Cout_tr = _c_upto(C_tr_time, C_tr_static, s, n_tr)
            Cout_te = _c_upto(C_te_time, C_te_static, s, n_te)

            Wt_tr = W_tr[:, s : s + 1]
            X_tr_out = np.concatenate([S_tr, W_past_tr, H_upto_tr, Cout_tr, Wt_tr], axis=1)

            Wt1_tr = np.ones((n_tr, 1), dtype=np.float32)
            Wt0_tr = np.zeros((n_tr, 1), dtype=np.float32)
            X_tr_1 = np.concatenate([S_tr, W_past_tr, H_upto_tr, Cout_tr, Wt1_tr], axis=1)
            X_tr_0 = np.concatenate([S_tr, W_past_tr, H_upto_tr, Cout_tr, Wt0_tr], axis=1)

            if test_obs.sum() > 0:
                Wt1_te = np.ones((n_te, 1), dtype=np.float32)
                Wt0_te = np.zeros((n_te, 1), dtype=np.float32)
                X_te_1 = np.concatenate([S_te, W_past_te, H_upto_te, Cout_te, Wt1_te], axis=1)
                X_te_0 = np.concatenate([S_te, W_past_te, H_upto_te, Cout_te, Wt0_te], axis=1)

            reg = _fit_downstream_regressor(
                X=X_tr_out[train_obs],
                y=y_out_tr[train_obs],
                nn_hidden=nn_hidden,
                nn_alpha=nn_alpha,
                nn_lr=nn_lr,
                nn_lr_scheduler=nn_lr_scheduler,
                nn_lr_scheduler_factor=nn_lr_scheduler_factor,
                nn_lr_scheduler_patience=nn_lr_scheduler_patience,
                nn_lr_scheduler_min_lr=nn_lr_scheduler_min_lr,
                nn_max_iter=nn_max_iter,
                nn_patience=nn_patience,
                nn_batch_size=nn_batch_size,
                nn_dropout=nn_dropout,
                random_state=int(random_state + 10000 * s),
                device=device,
            )

            m1_tr = reg.predict(X_tr_1).astype(np.float32)
            m0_tr = reg.predict(X_tr_0).astype(np.float32)

            if test_obs.sum() > 0:
                m1_te = reg.predict(X_te_1).astype(np.float32)
                m0_te = reg.predict(X_te_0).astype(np.float32)
                m1_te_all[test_obs, s] = m1_te[test_obs]
                m0_te_all[test_obs, s] = m0_te[test_obs]

            mt1_target_tr = (omega_prev_tr[:, s] * m1_tr).astype(np.float32)
            mt0_target_tr = (omega_prev_tr[:, s] * m0_tr).astype(np.float32)

            if s == 0:
                mt1 = _mean_or_fallback(mt1_target_tr, train_obs, train_obs)
                mt0 = _mean_or_fallback(mt0_target_tr, train_obs, train_obs)
                if test_obs.sum() > 0:
                    mtilde1_te_all[test_obs, s] = mt1
                    mtilde0_te_all[test_obs, s] = mt0
            else:
                n_codes = 2 ** s
                mt1_map = _fit_mtilde_saturated(
                    mt1_target_tr,
                    codes_tr_all[s],
                    train_obs,
                    train_obs,
                    n_codes,
                )
                mt0_map = _fit_mtilde_saturated(
                    mt0_target_tr,
                    codes_tr_all[s],
                    train_obs,
                    train_obs,
                    n_codes,
                )
                if test_obs.sum() > 0:
                    mtilde1_te_all[test_obs, s] = mt1_map[codes_te_all[s][test_obs]]
                    mtilde0_te_all[test_obs, s] = mt0_map[codes_te_all[s][test_obs]]

            p_s_tr = p_tr[:, s]
            delta_s = float(delta_vec[s])
            denom_tr = delta_s * p_s_tr + 1.0 - p_s_tr

            R_tr_rec[train_obs, s] = (
                (delta_s * p_s_tr[train_obs] * m1_tr[train_obs] + (1.0 - p_s_tr[train_obs]) * m0_tr[train_obs])
                / denom_tr[train_obs]
            )

        out = {
            "m1_te": m1_te_all,
            "m0_te": m0_te_all,
            "mt1_te": mtilde1_te_all,
            "mt0_te": mtilde0_te_all,
        }
        return out

    def _compute_psi_test(
        delta_vec: np.ndarray,
        *,
        W_te: np.ndarray,
        Y_te: np.ndarray,
        mask_te: np.ndarray,
        p_te: np.ndarray,
        pi_te: np.ndarray,
        m1_te: np.ndarray,
        m0_te: np.ndarray,
        mt1_te: np.ndarray,
        mt0_te: np.ndarray,
    ) -> np.ndarray:
        n_te, _ = W_te.shape
        n_segments_te = mask_te.sum(axis=1).astype(int)

        psi_vals = np.full(n_te, np.nan, dtype=np.float32)

        for i in range(n_te):
            Si = int(n_segments_te[i])
            if Si <= 0:
                continue

            prod_omega = 1.0
            acc = 0.0

            for s in range(Si):
                Ws = float(W_te[i, s])

                p_s = float(p_te[i, s])
                pi_raw = float(pi_te[i, s])

                if not np.isfinite(p_s) or not np.isfinite(pi_raw):
                    raise RuntimeError(
                        f"Non-finite p/pi for test unit (i={i}) at segment s={s}."
                    )

                delta_s = float(delta_vec[s])
                denom = delta_s * p_s + 1.0 - p_s

                m1 = float(m1_te[i, s])
                m0 = float(m0_te[i, s])
                mt1 = float(mt1_te[i, s])
                mt0 = float(mt0_te[i, s])

                if not (np.isfinite(m1) and np.isfinite(m0) and np.isfinite(mt1) and np.isfinite(mt0)):
                    raise RuntimeError(
                        f"Non-finite m/mtilde for test unit (i={i}) at segment s={s}."
                    )

                clip_eps = float(eps_prob) if (eps_prob is not None and eps_prob > 0) else None
                if Ws >= 0.5:
                    pi_denom = pi_raw
                    if clip_eps is not None:
                        pi_denom = min(max(pi_denom, clip_eps), 1.0 - clip_eps)
                    elif pi_denom <= 0.0:
                        pi_denom = 1e-12
                    A_num = delta_s * p_s * m1 * (1.0 - 1.0 / pi_denom) + (1.0 - p_s) * m0
                    omega = (delta_s * p_s / pi_denom) / denom
                else:
                    one_minus = 1.0 - pi_raw
                    if clip_eps is not None:
                        one_minus = min(max(one_minus, clip_eps), 1.0 - clip_eps)
                    elif one_minus <= 0.0:
                        one_minus = 1e-12
                    A_num = delta_s * p_s * m1 + (1.0 - p_s) * m0 * (1.0 - 1.0 / one_minus)
                    omega = ((1.0 - p_s) / one_minus) / denom

                A = A_num / denom
                B = delta_s * (Ws - p_s) * (mt1 - mt0) / (denom ** 2)

                acc += prod_omega * A + B
                prod_omega *= omega

            acc += prod_omega * float(Y_te[i])
            psi_vals[i] = float(acc)

        return psi_vals

    W_arr, mask_arr = _infer_mask_from_w(W)
    Y_arr = np.asarray(Y, dtype=np.float32).reshape(-1)

    n, T_max = W_arr.shape

    if Y_arr.shape[0] != n:
        raise ValueError(f"Y must have length N={n}; got {Y_arr.shape}")

    observed = mask_arr > 0

    R_arr = np.asarray(R, dtype=np.float32)
    if R_arr.ndim != 3:
        raise ValueError(f"R must be 3D, got {R_arr.shape}")

    R_video_arr: Optional[np.ndarray] = None
    if use_multimodal:
        if R_arr.shape[:2] == (n, T_max):
            pass
        elif R_arr.shape[1:] == (n, T_max):
            R_arr = np.transpose(R_arr, (1, 2, 0)).copy()
        else:
            raise ValueError(
                f"Multimodal R must be [N,T,F] or [F,N,T] with N={n}, T={T_max}; "
                f"got {R_arr.shape}"
            )

        text_input_dim_resolved = R_arr.shape[2] if text_input_dim is None else int(text_input_dim)
        if R_arr.shape[2] != text_input_dim_resolved:
            raise ValueError(
                f"R last dimension must equal text_input_dim={text_input_dim_resolved}; "
                f"got {R_arr.shape[2]}"
            )
        if np.any(~np.isfinite(R_arr[observed])):
            raise ValueError("R must be finite at every observed segment.")
        R_arr = np.where(observed[:, :, None], R_arr, 0.0).astype(np.float32)

        R_video_arr = np.asarray(R_video, dtype=np.float32)
        if R_video_arr.ndim == 5:
            R_video_arr = R_video_arr[:, :, None, :, :, :]
        if R_video_arr.ndim != 6:
            raise ValueError(
                "R_video must be [N,T,C_video,D,H,W] or [N,T,D,H,W]; "
                f"got {R_video_arr.shape}"
            )
        if R_video_arr.shape[:2] != (n, T_max):
            raise ValueError(
                f"R_video must agree with W on [N,T]=({n},{T_max}); "
                f"got {R_video_arr.shape[:2]}"
            )
        if R_video_arr.shape[2] != int(video_in_channels):
            raise ValueError(
                f"R_video has {R_video_arr.shape[2]} channels, but "
                f"video_in_channels={video_in_channels}."
            )
        if np.any(~np.isfinite(R_video_arr[observed])):
            raise ValueError("R_video must be finite at every observed segment.")
        R_video_arr = np.where(
            observed[:, :, None, None, None, None], R_video_arr, 0.0
        ).astype(np.float32)
    else:
        if R_arr.shape[:2] == (n, T_max):
            R_arr = np.transpose(R_arr, (2, 0, 1)).copy()
        F, N_check, T_check = R_arr.shape
        if N_check != n or T_check != T_max:
            raise ValueError(
                f"R must align with W: got [F,N,T]={R_arr.shape}, "
                f"expected N={n}, T={T_max}"
            )
        if np.any(~np.isfinite(R_arr[:, observed])):
            raise ValueError("R must be finite at every observed segment.")
        R_arr = np.where(observed[None, :, :], R_arr, 0.0).astype(np.float32)
        text_input_dim_resolved = int(text_input_dim) if text_input_dim is not None else int(F)

    delta_arr = np.asarray(delta_seq, dtype=float)
    if delta_arr.ndim == 0:
        delta_arr = delta_arr.reshape(1)

    if delta_arr.ndim == 1:
        if np.any(~np.isfinite(delta_arr)) or np.any(delta_arr <= 0):
            raise ValueError("delta_seq must be finite and >0.")
        delta_paths = np.tile(delta_arr.reshape(-1, 1), (1, T_max)).astype(float)
        delta_out = delta_arr.astype(float)
        delta_is_schedule = False
    elif delta_arr.ndim == 2:
        if delta_arr.shape[1] != T_max:
            raise ValueError(f"delta_seq 2D must be [K,T_max={T_max}], got {delta_arr.shape}")
        if np.any(~np.isfinite(delta_arr)) or np.any(delta_arr <= 0):
            raise ValueError("delta_seq must be finite and >0.")
        delta_paths = delta_arr.astype(float)
        delta_out = delta_paths
        delta_is_schedule = True
    else:
        raise ValueError("delta_seq must be scalar, 1D (J,), or 2D (J,T).")

    k = int(delta_paths.shape[0])
    ifvals = np.full((n, k), np.nan, dtype=float)
    est_eff = np.full(k, np.nan, dtype=float)

    K = int(K)
    if K < 2:
        raise ValueError("K must be at least 2.")
    if K > n:
        raise ValueError(f"K={K} cannot exceed the number of observations n={n}.")
    sample_split_fold = int(sample_split_fold)
    if sample_split_fold < 1 or sample_split_fold > K:
        raise ValueError(f"sample_split_fold must be in [1, K={K}].")

    rng = np.random.default_rng(random_state)
    s = np.tile(np.arange(1, K + 1), int(np.ceil(n / K)))[:n]
    rng.shuffle(s)
    split_iter = [sample_split_fold] if sample_split_only else range(1, K + 1)

    S_full = mask_arr.sum(axis=1).astype(np.float32)

    H_full: Optional[np.ndarray] = None
    if H is not None:
        H_arr = np.asarray(H, dtype=np.float32)
        if H_arr.ndim != 3:
            raise ValueError(f"H must be 3D, got {H_arr.shape}")
        if H_arr.shape[0] == n and H_arr.shape[1] == T_max:
            H_full = H_arr
        elif H_arr.shape[1] == n and H_arr.shape[2] == T_max:
            H_full = np.transpose(H_arr, (1, 2, 0)).copy()
        else:
            raise ValueError(f"H must be [N,T,rep] or [rep,N,T], got {H_arr.shape}")

        if np.any(~np.isfinite(H_full[observed])):
            raise ValueError("H must be finite at every observed segment.")
        H_full = np.where(observed[:, :, None], H_full, 0.0).astype(np.float32)

    C_time_full: Optional[np.ndarray] = None
    C_static_full: Optional[np.ndarray] = None

    if C is not None:
        C_arr = np.asarray(C, dtype=np.float32)

        if C_arr.ndim == 3:
            if C_arr.shape[0] == n and C_arr.shape[1] == T_max:
                C_time_full = C_arr
            elif C_arr.shape[1] == n and C_arr.shape[2] == T_max:
                C_time_full = np.transpose(C_arr, (1, 2, 0)).copy()
            else:
                raise ValueError(f"C must be [N,T,C] or [C,N,T]; got {C_arr.shape}")
            if np.any(~np.isfinite(C_time_full[observed])):
                raise ValueError("C must be finite at every observed segment.")
            C_time_full = np.where(
                observed[:, :, None], C_time_full, 0.0
            ).astype(np.float32)

        elif C_arr.ndim == 2:
            if C_arr.shape == (n, T_max):
                if np.any(~np.isfinite(C_arr[observed])):
                    raise ValueError("C must be finite at every observed segment.")
                C_time_full = np.where(observed, C_arr, 0.0).astype(np.float32)
            elif C_arr.shape[0] == n:
                C_static_full = C_arr
            else:
                raise ValueError(f"C must be [N,T] or [N,C]; got {C_arr.shape}")

        elif C_arr.ndim == 1:
            if C_arr.shape[0] != n:
                raise ValueError(f"C 1D must have length N={n}; got {C_arr.shape}")
            C_static_full = C_arr.reshape(n, 1)

        else:
            raise ValueError("C must be 1D, 2D, or 3D.")

    for split in split_iter:
        if verbose:
            if sample_split_only:
                print(f"\n{'='*60}\nSample-split held-out fold {split}/{K}\n{'='*60}")
            else:
                print(f"\n{'='*60}\nFold {split}/{K}\n{'='*60}")

        train_idx = np.where(s != split)[0]
        test_idx = np.where(s == split)[0]

        if use_multimodal:
            R_test = R_arr[test_idx, :, :]
            R_video_test = R_video_arr[test_idx, ...]
        else:
            R_test = R_arr[:, test_idx, :]
            R_video_test = None
        W_test = W_arr[test_idx, :]
        Y_test = Y_arr[test_idx]
        mask_test = mask_arr[test_idx, :]

        S_te = S_full[test_idx].reshape(-1, 1)

        C_te_time = C_time_full[test_idx, ...] if C_time_full is not None else None
        C_te_static = C_static_full[test_idx, :] if C_static_full is not None else None
        C_test_nn = C_te_time if C_te_time is not None else C_te_static

        pipelines = []

        if H_full is not None:
            if verbose:
                print("Step 1-2: using externally provided H (skipping DynamicTarNet)")

            H_tr_full = H_full[train_idx, :, :].astype(np.float32, copy=False)
            H_te_full = H_full[test_idx, :, :].astype(np.float32, copy=False)

            mu_tr_full = Y_arr[train_idx].astype(np.float32, copy=False)
            mu_te_full = Y_test.astype(np.float32, copy=False)

            W_tr_full = W_arr[train_idx, :]
            mask_tr_full = mask_arr[train_idx, :]
            S_tr_full = S_full[train_idx].reshape(-1, 1)

            C_tr_time_full = C_time_full[train_idx, ...] if C_time_full is not None else None
            C_tr_static_full = C_static_full[train_idx, :] if C_static_full is not None else None

            pipelines.append(
                {
                    "label": "externalH_fullTrain",
                    "R_tr": R_arr[train_idx, :, :] if use_multimodal else R_arr[:, train_idx, :],
                    "W_tr": W_tr_full,
                    "Y_tr": Y_arr[train_idx],
                    "mask_tr": mask_tr_full,
                    "S_tr": S_tr_full,
                    "H_tr": H_tr_full,
                    "mu_tr": mu_tr_full,
                    "C_tr_time": C_tr_time_full,
                    "C_tr_static": C_tr_static_full,
                    "H_te": H_te_full,
                    "mu_te": mu_te_full,
                }
            )

        else:
            if verbose:
                print("Step 1: training a single NN on full training fold")

            if use_multimodal:
                R_train = R_arr[train_idx, :, :]
                R_video_train = R_video_arr[train_idx, ...]
            else:
                R_train = R_arr[:, train_idx, :]
                R_video_train = None
            W_train = W_arr[train_idx, :]
            Y_train = Y_arr[train_idx]
            mask_train = mask_arr[train_idx, :]

            C_tr_time = C_time_full[train_idx, ...] if C_time_full is not None else None
            C_tr_static = C_static_full[train_idx, :] if C_static_full is not None else None
            C_train_nn = C_tr_time if C_tr_time is not None else C_tr_static

            if verbose:
                print(f"  [NN train] full train fold: n={len(train_idx)}")

            nn_full = _train_tarnet(
                R_train,
                W_train,
                Y_train,
                mask_train,
                C_train_nn,
                R_video_train,
                fold_label="NN_full",
                save_dir=os.path.join(model_dir, f"fold{split}_full") if model_dir else None,
            )

            if verbose:
                print("Step 2: extracting representations/outcomes on train + test (in-sample train)", flush=True)

            t_step2 = time.time()
            mu_tr_full, H_tr_full = _predict_mu_H(
                nn_full, R_train, W_train, mask_train, C_train_nn, R_video_train
            )
            mu_te_full, H_te_full = _predict_mu_H(
                nn_full, R_test, W_test, mask_test, C_test_nn, R_video_test
            )
            if verbose:
                print(f"  Step 2 done in {time.time() - t_step2:.1f}s", flush=True)

            pipelines.append(
                {
                    "label": "NN_fullTrain",
                    "R_tr": R_train,
                    "W_tr": W_train,
                    "Y_tr": Y_train,
                    "mask_tr": mask_train,
                    "S_tr": S_full[train_idx].reshape(-1, 1),
                    "H_tr": H_tr_full,
                    "mu_tr": mu_tr_full,
                    "C_tr_time": C_tr_time,
                    "C_tr_static": C_tr_static,
                    "H_te": H_te_full,
                    "mu_te": mu_te_full,
                }
            )

        if verbose:
            print("Step 3: fit pi_s and p_s within each pipeline; predict on test; average nuisances", flush=True)

        t_step3 = time.time()
        for pp in pipelines:
            pi_p = _fit_pi_p_on_train_predict_test(
                W_tr=pp["W_tr"],
                mask_tr=pp["mask_tr"],
                H_tr=pp["H_tr"],
                S_tr=pp["S_tr"],
                C_tr_time=pp["C_tr_time"],
                C_tr_static=pp["C_tr_static"],
                W_te=W_test,
                mask_te=mask_test,
                H_te=pp["H_te"],
                S_te=S_te,
                C_te_time=C_te_time,
                C_te_static=C_te_static,
            )
            pp["pi_te"] = pi_p["pi_te"]
            pp["pi_tr"] = pi_p["pi_tr"]
            pp["p_tr"] = pi_p["p_tr"]
            pp["p_te"] = pi_p["p_te"]
            pp["codes_tr_all"] = pi_p["codes_tr_all"]
            pp["codes_te_all"] = pi_p["codes_te_all"]

        if verbose:
            print(f"  Step 3 done in {time.time() - t_step3:.1f}s", flush=True)

        pi_te_avg = pipelines[0]["pi_te"]
        p_te_avg = pipelines[0]["p_te"]

        for j in range(k):
            delta_vec = delta_paths[j, :].astype(float, copy=False)

            if verbose:
                if np.allclose(delta_vec, delta_vec[0]):
                    print(
                        f"Step 4-5: backward outcome regression + EIF for delta={float(delta_vec[0]):.6f}",
                        flush=True,
                    )
                else:
                    print(
                        "Step 4-5: backward outcome regression + EIF for "
                        f"delta schedule j={j} (min={float(delta_vec.min()):.6f}, max={float(delta_vec.max()):.6f})",
                        flush=True,
                    )

            t_step45 = time.time()
            pp = pipelines[0]
            md = _fit_m_and_predict_test(
                delta_vec=delta_vec,
                W_tr=pp["W_tr"],
                mask_tr=pp["mask_tr"],
                H_tr=pp["H_tr"],
                mu_tr=pp["mu_tr"],
                S_tr=pp["S_tr"],
                C_tr_time=pp["C_tr_time"],
                C_tr_static=pp["C_tr_static"],
                pi_tr=pp["pi_tr"],
                p_tr=pp["p_tr"],
                codes_tr_all=pp["codes_tr_all"],
                W_te=W_test,
                mask_te=mask_test,
                H_te=pp["H_te"],
                S_te=S_te,
                C_te_time=C_te_time,
                C_te_static=C_te_static,
                p_te=pp["p_te"],
                codes_te_all=pp["codes_te_all"],
            )

            m1_te_avg = md["m1_te"]
            m0_te_avg = md["m0_te"]
            mt1_te_avg = md["mt1_te"]
            mt0_te_avg = md["mt0_te"]

            psi_vals = _compute_psi_test(
                delta_vec=delta_vec,
                W_te=W_test,
                Y_te=Y_test,
                mask_te=mask_test,
                p_te=p_te_avg,
                pi_te=pi_te_avg,
                m1_te=m1_te_avg,
                m0_te=m0_te_avg,
                mt1_te=mt1_te_avg,
                mt0_te=mt0_te_avg,
            )

            ifvals[test_idx, j] = psi_vals.astype(float)

            if verbose:
                print(f"  Step 4-5 done in {time.time() - t_step45:.1f}s", flush=True)

    for j in range(k):
        est_eff[j] = np.nanmean(ifvals[:, j])

    estimation_units = (S_full > 0) & np.any(np.isfinite(ifvals), axis=1)
    n_eff = int(np.sum(estimation_units))
    if n_eff == 0:
        raise RuntimeError("No estimated units with positive length S_i; check sample split and masks.")

    sigma2 = np.nanmean((ifvals - est_eff.reshape(1, -1)) ** 2, axis=0)
    sigma = np.sqrt(np.maximum(sigma2, 0.0))
    se = sigma / np.sqrt(n_eff)

    ll1 = est_eff - 1.96 * se
    ul1 = est_eff + 1.96 * se

    ll2 = None
    ul2 = None
    if n_boot > 0:
        scores = ifvals[estimation_units, :]
        if not np.all(np.isfinite(scores)):
            raise RuntimeError(
                "Uniform confidence bands require finite influence-function "
                "values for every estimated unit and delta."
            )

        centered_scores = scores - est_eff.reshape(1, -1)
        active = sigma > 0
        if np.any(active):
            bootstrap_rng = np.random.default_rng(random_state)
            maxima = np.empty(n_boot, dtype=float)
            boot_batch_size = max(
                1, min(n_boot, 1_000_000 // max(n_eff, 1))
            )
            scale = np.sqrt(n_eff) * sigma[active]
            for start in range(0, n_boot, boot_batch_size):
                stop = min(start + boot_batch_size, n_boot)
                multipliers = bootstrap_rng.choice(
                    (-1.0, 1.0), size=(stop - start, n_eff)
                )
                process = multipliers @ centered_scores[:, active]
                process /= scale
                maxima[start:stop] = np.max(np.abs(process), axis=1)
            critical_value = float(np.quantile(maxima, 0.95))
        else:
            critical_value = 0.0

        ll2 = est_eff - critical_value * se
        ul2 = est_eff + critical_value * se

    if verbose:
        print("\n" + "=" * 60)
        print("Final summary")
        print("=" * 60)
        if delta_is_schedule:
            print("delta_paths shape:", delta_paths.shape)
        else:
            print("delta:", delta_out)
        print("est  :", est_eff)
        print("sigma:", sigma)
        print("se   :", se)
        print("pointwise 95% CI:", np.vstack([ll1, ul1]))
        if ll2 is not None:
            print("uniform   95% CI:", np.vstack([ll2, ul2]))
        print("n_eff:", n_eff)

    out = {
        "delta": np.array(delta_out, dtype=float),
        "delta_paths": np.array(delta_paths, dtype=float),
        "est": est_eff,
        "sigma": sigma,
        "se": se,
        "ll1": ll1,
        "ul1": ul1,
        "ll2": ll2,
        "ul2": ul2,
        "ifvals": ifvals,
        "n_eff": n_eff,
    }
    return out
