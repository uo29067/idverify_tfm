"""
model.py — Arquitectura, pérdidas y métricas del modelo DocVerify en PyTorch.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


# ============================================================
# PÉRDIDAS
# ============================================================

def dice_loss(y_pred: torch.Tensor, y_true: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """
    Dice Loss por imagen sobre probabilidades (aplica sigmoid internamente).
    Devuelve un tensor escalar (media del batch).
    """
    y_true  = y_true.float()
    y_pred  = torch.sigmoid(y_pred.float())  # logits → probabilidades

    y_true_f = y_true.view(y_true.size(0), -1)
    y_pred_f = y_pred.view(y_pred.size(0), -1)

    inter = (y_true_f * y_pred_f).sum(dim=1)
    denom = y_true_f.sum(dim=1) + y_pred_f.sum(dim=1)

    dice_per_img = 1.0 - (2.0 * inter + smooth) / (denom + smooth)
    return dice_per_img.mean()


def bce_dice_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    Pérdida combinada para segmentación (trabaja con logits, compatible con AMP):
      BCEWithLogits(y_pred, y_true) + DiceLoss(sigmoid(y_pred), y_true)
    """
    bce = F.binary_cross_entropy_with_logits(y_pred, y_true.float())
    return bce + dice_loss(y_pred, y_true)


# ============================================================
# BLOQUES DE LA ARQUITECTURA
# ============================================================

class ConvLeakyBN(nn.Module):
    """Conv2D + LeakyReLU + BatchNorm."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, alpha: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=kernel // 2, bias=False),
            nn.LeakyReLU(negative_slope=alpha, inplace=True),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecBlock(nn.Module):
    """Bloque decoder: UpSampling × 2 + Concatenate skip + 2× Conv."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, alpha: float = 0.2):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=alpha, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=alpha, inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ============================================================
# ARQUITECTURA PRINCIPAL
# ============================================================

class DocVerifyModel(nn.Module):
    """
    Patel CNN Encoder + U-Net Decoder para clasificación y segmentación multi-tarea.

    Entradas:
      x — (B, 3, H, W) float32 [0, 1]

    Salidas (dict):
      "cls"  — (B, 1) sigmoid: probabilidad de ataque
      "mask" — (B, 1, H, W) sigmoid: máscara de regiones alteradas
    """

    def __init__(
        self,
        in_ch:        int   = 3,
        alpha:        float = 0.2,
        dropout_rate: float = 0.5,
        dec_ch:       int   = 128,
    ):
        super().__init__()
        self.alpha = alpha

        # ── Encoder ──────────────────────────────────────────

        # Bloque 1: 8 filtros
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, 8, 3, padding=1, bias=False),
            nn.LeakyReLU(alpha, inplace=True),
            nn.BatchNorm2d(8),
        )

        # Bloque 2: 16 filtros → skip s224
        self.enc2 = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1, bias=False),
            nn.LeakyReLU(alpha, inplace=True),
            nn.Conv2d(16, 16, 3, padding=1, bias=False),
            nn.LeakyReLU(alpha, inplace=True),
            nn.BatchNorm2d(16),
        )
        self.pool1 = nn.AvgPool2d(2)  # → 112×112

        # Bloque 3: 32 filtros × 3 → skip s112
        self.enc3 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1, bias=False), nn.LeakyReLU(alpha, inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False), nn.LeakyReLU(alpha, inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False), nn.LeakyReLU(alpha, inplace=True),
            nn.BatchNorm2d(32),
        )
        self.pool2 = nn.AvgPool2d(2)  # → 56×56

        # Bloque 4: 64 filtros × 4 → skip s56
        self.enc4 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, bias=False), nn.LeakyReLU(alpha, inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False), nn.LeakyReLU(alpha, inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False), nn.LeakyReLU(alpha, inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False), nn.LeakyReLU(alpha, inplace=True),
            nn.BatchNorm2d(64),
        )
        self.pool3 = nn.AvgPool2d(2)  # → 28×28

        # Bloque 5: 128 filtros → skip s28
        self.enc5 = nn.Sequential(
            nn.Conv2d(64, 128, 5, padding=2, bias=False),
            nn.LeakyReLU(alpha, inplace=True),
            nn.BatchNorm2d(128),
        )
        self.pool4 = nn.MaxPool2d(2)  # → 14×14

        # Bloque 6: 256 filtros → skip s14 → bottleneck 7×7
        self.enc6 = nn.Sequential(
            nn.Conv2d(128, 256, 5, padding=2, bias=False),
            nn.LeakyReLU(alpha, inplace=True),
            nn.BatchNorm2d(256),
        )
        self.pool5 = nn.MaxPool2d(2)  # → 7×7

        # ── Cabeza de clasificación ───────────────────────────
        self.cls_gap = nn.AdaptiveAvgPool2d(1)
        self.cls_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(256, 32), nn.LeakyReLU(alpha, inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 16),  nn.LeakyReLU(alpha, inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(16, 16),  nn.LeakyReLU(alpha, inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(16, 1),
        )

        # ── Cabeza de segmentación (decoder U-Net) ────────────
        self.mask_proj = nn.Sequential(
            nn.Conv2d(256, dec_ch, 1, bias=False),
            nn.LeakyReLU(alpha, inplace=True),
        )

        self.dec14  = DecBlock(dec_ch,       256, dec_ch,        alpha)   # 7  → 14
        self.dec28  = DecBlock(dec_ch,       128, dec_ch // 2,   alpha)   # 14 → 28
        self.dec56  = DecBlock(dec_ch // 2,  64,  dec_ch // 4,   alpha)   # 28 → 56
        self.dec112 = DecBlock(dec_ch // 4,  32,  dec_ch // 8,   alpha)   # 56 → 112
        self.dec224 = DecBlock(dec_ch // 8,  16,  dec_ch // 16,  alpha)   # 112 → 224

        self.mask_out = nn.Conv2d(dec_ch // 16, 1, 1)

    def forward(self, x: torch.Tensor) -> dict:
        H, W = x.shape[2], x.shape[3]

        # Encoder
        e1 = self.enc1(x)
        s224 = self.enc2(e1)
        e2 = self.pool1(s224)

        s112 = self.enc3(e2)
        e3 = self.pool2(s112)

        s56 = self.enc4(e3)
        e4 = self.pool3(s56)

        s28 = self.enc5(e4)
        e5 = self.pool4(s28)

        s14 = self.enc6(e5)
        bottleneck = self.pool5(s14)

        # Clasificación
        c = self.cls_gap(bottleneck).flatten(1)
        cls_out = self.cls_head(c)  # logits (B, 1) — sigmoid aplicado en la pérdida

        # Segmentación
        m = self.mask_proj(bottleneck)
        m = self.dec14(m,  s14)
        m = self.dec28(m,  s28)
        m = self.dec56(m,  s56)
        m = self.dec112(m, s112)
        m = self.dec224(m, s224)

        # Asegurar que la salida tiene el mismo tamaño que la entrada
        m = F.interpolate(m, size=(H, W), mode="bilinear", align_corners=False)
        mask_out = self.mask_out(m)  # logits (B, 1, H, W) — sigmoid aplicado en la pérdida

        return {"cls": cls_out, "mask": mask_out}


# ============================================================
# FACTORY: construir modelo + optimizador
# ============================================================

def build_model(params: dict, device: torch.device) -> nn.Module:
    """Construye el modelo con los hiperparámetros dados y lo mueve al device."""

    model = DocVerifyModel(
        alpha        = float(params.get("alpha", config.LEAKY_RELU_ALPHA)),
        dropout_rate = float(params["dropout_rate"]),
        dec_ch       = int(params["dec_ch"]),
    )

    return model.to(device)


def build_optimizer(model: nn.Module, params: dict) -> torch.optim.Optimizer:
    """Construye AdamW con los hiperparámetros dados."""
    return torch.optim.AdamW(
        model.parameters(),
        lr           = float(params["lr"]),
        weight_decay = float(params["weight_decay"]),
    )
