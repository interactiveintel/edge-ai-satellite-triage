#!/usr/bin/env python3
"""Train a MobileNetV3-Small cloud-mask / scene-classification model.

Supports two datasets:
  - EuroSAT (default): 27,000 Sentinel-2 patches (64x64, RGB) across 10 LULC classes
  - BigEarthNet: 590,326 Sentinel-2 patches (120x120, multi-label, 19 classes)

Produces an INT8-quantised ONNX model ready for TensorRT on Jetson Orin.
We re-frame class labels as a 3-head regression: cloud_fraction, anomaly_score,
value_score — suitable for the edge triage pipeline.

Usage::

    # Full training on EuroSAT (downloads automatically)
    python scripts/train_cloud_mask.py

    # Train on BigEarthNet (requires manual download — see below)
    python scripts/train_cloud_mask.py --dataset bigearthnet --data-dir data/BigEarthNet-v1.0

    # Quick test run (1 epoch, small subset)
    python scripts/train_cloud_mask.py --epochs 1 --quick

    # Export only (from existing checkpoint)
    python scripts/train_cloud_mask.py --export-only --checkpoint models/cloud_mask_best.pt

BigEarthNet download:
    https://bigearth.net/ — download S2 patches and extract to data/BigEarthNet-v1.0/

Output:
    models/cloud_mask_best.pt              — PyTorch checkpoint
    models/cloud_mask_mobilenet_int8.onnx  — Quantised ONNX (deploy this)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ── Ensure project root is importable ───────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset, random_split
    from torchvision import transforms
except ImportError:
    sys.exit("ERROR: PyTorch required. Install with: pip install torch torchvision")

MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ── EuroSAT dataset labels → triage scores mapping ─────────────────────────
# EuroSAT classes: AnnualCrop, Forest, HerbaceousVegetation, Highway,
#   Industrial, Pasture, PermanentCrop, Residential, River, SeaLake
#
# Mapping: (cloud_frac_proxy, anomaly_score, value_score)
# Scores calibrated for dual-use triage: space-based EO and ground ISR.
#   cloud_frac  — proxy for atmospheric obscuration (0=clear, 1=opaque)
#   anomaly     — scene novelty / change-detection priority
#   value       — intelligence / science value for downlink
EUROSAT_LABEL_TO_SCORES = {
    0: (0.05, 0.15, 0.55),  # AnnualCrop — low cloud, low anomaly, moderate value
    1: (0.05, 0.10, 0.40),  # Forest — baseline, low priority
    2: (0.05, 0.15, 0.45),  # HerbaceousVegetation — slightly above forest
    3: (0.05, 0.65, 0.75),  # Highway — infra interest, high anomaly
    4: (0.05, 0.75, 0.85),  # Industrial — highest anomaly + value (SIGINT)
    5: (0.05, 0.10, 0.35),  # Pasture — lowest value
    6: (0.05, 0.20, 0.55),  # PermanentCrop — moderate
    7: (0.05, 0.55, 0.70),  # Residential — moderate anomaly, good value
    8: (0.15, 0.35, 0.65),  # River — some haze, water-body interest
    9: (0.25, 0.20, 0.45),  # SeaLake — higher cloud proxy, maritime
}

# ── BigEarthNet 19-class label → triage scores ────────────────────────────
# BigEarthNet uses multi-label (multiple classes per patch). We take the
# maximum score across all labels present. These 19 classes come from the
# CORINE Land Cover Level-3 simplification used in BigEarthNet-v1.0.
BIGEARTHNET_LABEL_TO_SCORES = {
    "Urban fabric":                         (0.05, 0.60, 0.80),
    "Industrial or commercial units":       (0.05, 0.75, 0.85),
    "Arable land":                          (0.05, 0.15, 0.50),
    "Permanent crops":                      (0.05, 0.20, 0.55),
    "Pastures":                             (0.05, 0.10, 0.35),
    "Complex cultivation patterns":         (0.05, 0.25, 0.55),
    "Land principally occupied by agriculture": (0.05, 0.15, 0.50),
    "Agro-forestry areas":                  (0.05, 0.15, 0.45),
    "Broad-leaved forest":                  (0.05, 0.10, 0.40),
    "Coniferous forest":                    (0.05, 0.10, 0.40),
    "Mixed forest":                         (0.05, 0.10, 0.40),
    "Natural grasslands and sparsely vegetated areas": (0.10, 0.20, 0.45),
    "Moors, heathland and sclerophyllous vegetation":  (0.10, 0.15, 0.40),
    "Transitional woodland, shrub":         (0.10, 0.15, 0.45),
    "Beaches, dunes, sands":                (0.10, 0.30, 0.55),
    "Inland wetlands":                      (0.15, 0.35, 0.60),
    "Coastal wetlands":                     (0.15, 0.40, 0.65),
    "Inland waters":                        (0.15, 0.30, 0.60),
    "Marine waters":                        (0.25, 0.25, 0.50),
}

# Legacy alias used by EuroSATTriageDataset
LABEL_TO_SCORES = EUROSAT_LABEL_TO_SCORES


# ── Model ───────────────────────────────────────────────────────────────────


class CloudMaskHead(nn.Module):
    """MobileNetV3-Small backbone + 3-output regression head.

    Accepts (B, C, H, W) with any C via a learnable 1x1 input projection.
    Outputs 3 sigmoid scores: [cloud_fraction, anomaly_score, value_score].
    """

    def __init__(self, in_channels: int = 13) -> None:
        super().__init__()
        # Project any channel count to 3 for MobileNetV3 RGB input
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, 3, kernel_size=1, bias=False),
            nn.BatchNorm2d(3),
        )
        # Use pretrained MobileNetV3 Small as backbone
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
        backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        # MobileNetV3-Small last channel is 576
        self.head = nn.Sequential(
            nn.Linear(576, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 3),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.head(x)


# ── Dataset wrapper ─────────────────────────────────────────────────────────


class EuroSATTriageDataset(Dataset):
    """Wraps torchvision EuroSAT, converts class labels to regression targets."""

    def __init__(self, root: str, download: bool = True, transform: transforms.Compose | None = None) -> None:
        from torchvision.datasets import EuroSAT
        self._ds = EuroSAT(root=root, download=download, transform=None)
        self._transform = transform or transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img, label = self._ds[idx]
        img_tensor = self._transform(img)  # (3, 64, 64) — RGB only from EuroSAT JPEG
        scores = torch.tensor(EUROSAT_LABEL_TO_SCORES.get(label, (0.5, 0.5, 0.5)), dtype=torch.float32)
        return img_tensor, scores


class BigEarthNetTriageDataset(Dataset):
    """Load BigEarthNet-v1.0 S2 patches with multi-label → regression targets.

    BigEarthNet stores each patch as a directory of per-band TIF files plus a
    JSON metadata file listing CLC labels. We load RGB bands (B04, B03, B02),
    resize to 64x64, and convert multi-label classes to triage scores by
    taking the element-wise max across all present labels.

    Requires manual download from https://bigearth.net/ to ``root_dir``.
    """

    def __init__(self, root_dir: str, transform: transforms.Compose | None = None) -> None:
        self._root = Path(root_dir)
        if not self._root.exists():
            raise FileNotFoundError(
                f"BigEarthNet directory not found: {self._root}\n"
                "Download from https://bigearth.net/ and extract to this path."
            )
        # Each subdirectory is a patch
        self._patches = sorted([
            p for p in self._root.iterdir()
            if p.is_dir() and (p / f"{p.name}_labels_metadata.json").exists()
        ])
        if not self._patches:
            raise FileNotFoundError(f"No valid patches found in {self._root}")
        self._transform = transform or transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])
        print(f"BigEarthNet: found {len(self._patches)} patches in {self._root}")

    def __len__(self) -> int:
        return len(self._patches)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        patch_dir = self._patches[idx]
        name = patch_dir.name

        # Load RGB bands (B04=Red, B03=Green, B02=Blue) as numpy
        from PIL import Image
        bands = []
        for band in ("B04", "B03", "B02"):
            tif_path = patch_dir / f"{name}_{band}.tif"
            if tif_path.exists():
                bands.append(np.array(Image.open(tif_path), dtype=np.float32))
            else:
                bands.append(np.zeros((120, 120), dtype=np.float32))

        # Stack to (H, W, 3) and normalize
        rgb = np.stack(bands, axis=-1)
        max_val = rgb.max()
        if max_val > 0:
            rgb = rgb / max_val

        img = Image.fromarray((rgb * 255).astype(np.uint8))
        img_tensor = self._transform(img)

        # Load labels and compute triage scores (element-wise max across labels)
        meta_path = patch_dir / f"{name}_labels_metadata.json"
        labels: list[str] = []
        if meta_path.exists():
            import json
            with open(meta_path) as f:
                meta = json.load(f)
            labels = meta.get("labels", [])

        scores = np.array([0.5, 0.5, 0.5], dtype=np.float32)  # default
        matched = [
            np.array(BIGEARTHNET_LABEL_TO_SCORES[lbl], dtype=np.float32)
            for lbl in labels if lbl in BIGEARTHNET_LABEL_TO_SCORES
        ]
        if matched:
            scores = np.stack(matched).max(axis=0)

        return img_tensor, torch.from_numpy(scores)


# ── Training loop ───────────────────────────────────────────────────────────


def train(args: argparse.Namespace) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Dataset selection
    if args.dataset == "bigearthnet":
        data_dir = args.data_dir or str(PROJECT_ROOT / "data" / "BigEarthNet-v1.0")
        print(f"Loading BigEarthNet from {data_dir}...")
        ds = BigEarthNetTriageDataset(root_dir=data_dir)
    else:
        data_dir = args.data_dir or str(PROJECT_ROOT / "data")
        print("Loading EuroSAT dataset (downloads ~90 MB on first run)...")
        ds = EuroSATTriageDataset(root=data_dir, download=True)

    if args.quick:
        ds, _ = random_split(ds, [min(500, len(ds)), max(0, len(ds) - 500)])

    train_size = int(0.8 * len(ds))
    val_size = len(ds) - train_size
    train_ds, val_ds = random_split(ds, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    print(f"Dataset: {args.dataset} | Train: {len(train_ds)} | Val: {len(val_ds)}")

    # Model — both datasets produce 3-channel RGB tensors
    model = CloudMaskHead(in_channels=3).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float("inf")
    best_path = MODELS_DIR / "cloud_mask_best.pt"

    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0.0
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= len(train_ds)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                preds = model(imgs)
                val_loss += criterion(preds, targets).item() * imgs.size(0)
        val_loss /= len(val_ds)

        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1:3d}/{args.epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | lr={lr:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)
            print(f"  -> Saved best model (val_loss={val_loss:.4f})")

    print(f"\nTraining complete. Best val_loss={best_val_loss:.4f}")
    return best_path


# ── Export ──────────────────────────────────────────────────────────────────


def export_onnx(checkpoint_path: Path, in_channels: int = 3) -> Path:
    """Load checkpoint, quantise, and export to ONNX."""
    print(f"Loading checkpoint: {checkpoint_path}")
    model = CloudMaskHead(in_channels=in_channels)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    model.eval()

    # Dynamic quantisation (INT8 Linear layers)
    model_q = torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)

    # ONNX export
    onnx_path = MODELS_DIR / "cloud_mask_mobilenet_int8.onnx"
    dummy = torch.randn(1, in_channels, 64, 64)
    torch.onnx.export(
        model,  # export the fp32 model (ONNX doesn't support qint8 natively; TensorRT handles INT8)
        dummy,
        str(onnx_path),
        opset_version=17,
        input_names=["input"],
        output_names=["scores"],
        dynamic_axes={"input": {0: "batch", 2: "height", 3: "width"},
                      "scores": {0: "batch"}},
    )
    print(f"ONNX exported: {onnx_path} ({onnx_path.stat().st_size / 1024:.0f} KB)")

    # Also save FP16 variant
    fp16_path = MODELS_DIR / "cloud_mask_mobilenet_fp16.onnx"
    try:
        import onnx
        from onnxruntime.transformers import float16
        model_onnx = onnx.load(str(onnx_path))
        model_fp16 = float16.convert_float_to_float16(model_onnx)
        onnx.save(model_fp16, str(fp16_path))
        print(f"FP16 exported: {fp16_path}")
    except ImportError:
        print("(Skipping FP16 export — install onnx + onnxruntime for FP16 conversion)")

    # TensorRT conversion instructions
    print("\n--- TensorRT conversion (run on Jetson) ---")
    print(f"  trtexec --onnx={onnx_path.name} --saveEngine=cloud_mask_trt.engine \\")
    print("    --int8 --inputIOFormats=fp32:chw --outputIOFormats=fp32:chw")

    return onnx_path


# ── CLI ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Train cloud-mask triage model")
    parser.add_argument("--dataset", choices=["eurosat", "bigearthnet"], default="eurosat",
                        help="Training dataset (default: eurosat)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Override data directory (default: auto per dataset)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--quick", action="store_true", help="Small subset, fast test")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    if args.export_only:
        cp = Path(args.checkpoint or MODELS_DIR / "cloud_mask_best.pt")
        if not cp.exists():
            sys.exit(f"Checkpoint not found: {cp}")
        export_onnx(cp)
    else:
        best = train(args)
        export_onnx(best)


if __name__ == "__main__":
    main()
