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
# ENCODER ALTERNATIVO: EfficientNet-B4 (drop-in replacement)
# ============================================================

class DocVerifyEfficientNet(nn.Module):
    """
    EfficientNet-B4 Encoder (pretrained ImageNet) + U-Net Decoder.
    Drop-in replacement del encoder Patel en DocVerify.

    Diferencias respecto a DocVerifyModel (Patel):
      - Encoder: EfficientNet-B4 (torchvision, pesos ImageNet)
      - Skip channels: [24, 32, 56, 160, 448] vs Patel [16, 32, 64, 128, 256]
      - Sin skip a 224x224 (EfficientNet hace stride=2 desde la primera capa)
      - Normalización ImageNet aplicada internamente (inputs siguen siendo [0,1])
      - Bottleneck: 448ch en vez de 256ch
    """

    def __init__(
        self,
        alpha:        float = 0.2,
        dropout_rate: float = 0.5,
        dec_ch:       int   = 128,
        pretrained:   bool  = True,
    ):
        super().__init__()

        # ── Normalización ImageNet (aplicada en forward) ──────
        self.register_buffer(
            "norm_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "norm_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

        # ── Encoder EfficientNet-B4 ───────────────────────────
        from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
        weights = EfficientNet_B4_Weights.DEFAULT if pretrained else None
        _eff  = efficientnet_b4(weights=weights)
        feats = _eff.features  # nn.Sequential de 9 bloques

        # stem + MBConv stage1  → 24ch, 112x112 (stride=2 en stem)
        self.enc_112 = nn.Sequential(feats[0], feats[1])
        # MBConv stage2         → 32ch,  56x56 (stride=2)
        self.enc_56  = feats[2]
        # MBConv stage3         → 56ch,  28x28 (stride=2)
        self.enc_28  = feats[3]
        # MBConv stages 4+5     → 160ch, 14x14 (stride=2 en stage4, stride=1 en stage5)
        self.enc_14  = nn.Sequential(feats[4], feats[5])
        # MBConv stages 6+7     → 448ch,  7x7 (stride=2 en stage6, stride=1 en stage7)
        self.enc_7   = nn.Sequential(feats[6], feats[7])
        # feats[8] (head 1792ch) no se usa

        # ── Cabeza de clasificación ───────────────────────────
        self.cls_gap = nn.AdaptiveAvgPool2d(1)
        self.cls_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(448, 32), nn.LeakyReLU(alpha, inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 16),  nn.LeakyReLU(alpha, inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(16, 16),  nn.LeakyReLU(alpha, inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(16, 1),
        )

        # ── Cabeza de segmentación (decoder U-Net) ────────────
        self.mask_proj = nn.Sequential(
            nn.Conv2d(448, dec_ch, 1, bias=False),
            nn.LeakyReLU(alpha, inplace=True),
        )

        self.dec14  = DecBlock(dec_ch,       160, dec_ch,       alpha)  # 7  → 14, skip 160ch
        self.dec28  = DecBlock(dec_ch,        56, dec_ch // 2,  alpha)  # 14 → 28, skip  56ch
        self.dec56  = DecBlock(dec_ch // 2,   32, dec_ch // 4,  alpha)  # 28 → 56, skip  32ch
        self.dec112 = DecBlock(dec_ch // 4,   24, dec_ch // 8,  alpha)  # 56 → 112, skip 24ch

        # Sin skip a 224x224: EfficientNet no tiene features a esa escala
        self.dec224 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(dec_ch // 8, dec_ch // 16, 3, padding=1, bias=False),
            nn.LeakyReLU(alpha, inplace=True),
            nn.Conv2d(dec_ch // 16, dec_ch // 16, 3, padding=1, bias=False),
            nn.LeakyReLU(alpha, inplace=True),
        )

        self.mask_out = nn.Conv2d(dec_ch // 16, 1, 1)

    def forward(self, x: torch.Tensor) -> dict:
        H, W = x.shape[2], x.shape[3]

        # Normalización ImageNet (inputs en [0,1])
        x = (x - self.norm_mean) / self.norm_std

        # Encoder con skip connections
        s112       = self.enc_112(x)    # 24ch, 112x112
        s56        = self.enc_56(s112)  # 32ch,  56x56
        s28        = self.enc_28(s56)   # 56ch,  28x28
        s14        = self.enc_14(s28)   # 160ch, 14x14
        bottleneck = self.enc_7(s14)    # 448ch,  7x7

        # Clasificación
        c = self.cls_gap(bottleneck).flatten(1)
        cls_out = self.cls_head(c)

        # Segmentación
        m = self.mask_proj(bottleneck)  # dec_ch,      7x7
        m = self.dec14(m,  s14)         # dec_ch,     14x14
        m = self.dec28(m,  s28)         # dec_ch//2,  28x28
        m = self.dec56(m,  s56)         # dec_ch//4,  56x56
        m = self.dec112(m, s112)        # dec_ch//8, 112x112
        m = self.dec224(m)              # dec_ch//16,224x224

        m = F.interpolate(m, size=(H, W), mode="bilinear", align_corners=False)
        mask_out = self.mask_out(m)

        return {"cls": cls_out, "mask": mask_out}


# ============================================================
# FACTORY: construir modelo + optimizador
# ============================================================

def build_model(params: dict, device: torch.device) -> nn.Module:
    """Construye el modelo con los hiperparámetros dados y lo mueve al device."""

    # ── [Patel] encoder original ──────────────────────────────
    # model = DocVerifyModel(
    #     alpha        = float(params.get("alpha", config.LEAKY_RELU_ALPHA)),
    #     dropout_rate = float(params["dropout_rate"]),
    #     dec_ch       = int(params["dec_ch"]),
    # )

    # ── [EfficientNet-B4] encoder drop-in replacement ─────────
    model = DocVerifyEfficientNet(
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
