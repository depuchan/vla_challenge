"""
ACT-VLA Training Script — Optimized for Google Colab T4 GPU
============================================================
Key optimizations applied:
  - Mixed-precision (AMP) training via torch.cuda.amp  →  ~1.5–2x throughput
  - torch.compile() (PyTorch 2.x) for graph-level fusion      →  ~10–30% speedup
  - Pin memory + persistent workers for faster host→GPU transfers
  - GradScaler for numerically-stable fp16 backward pass
  - Learned task embedding (single nn.Parameter) — no external text encoder needed
  - Gradient clipping to stabilise training with lower precision
  - torch.backends.cudnn.benchmark = True for cuDNN auto-tuner
  - Increased batch size (64) to saturate T4 VRAM (~16 GB)
  - Reduced DataLoader workers to 2 (Colab limit) with prefetch_factor=2
  - tqdm progress bars for readable Colab cell output
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler          # ← AMP
import h5py
import numpy as np
import torchvision.transforms as T
import matplotlib.pyplot as plt
from tqdm.auto import tqdm                               # ← nice Colab bars

# ---------------------------------------------------------------------------
# 0. Global perf knobs
# ---------------------------------------------------------------------------
torch.backends.cudnn.benchmark = True   # auto-tune cuDNN kernels for fixed input sizes


# ---------------------------------------------------------------------------
# 1. ACT MODEL
# ---------------------------------------------------------------------------
class ACTPolicy(nn.Module):
    def __init__(self, action_dim: int = 7, chunk_size: int = 10):
        super().__init__()
        self.chunk_size = chunk_size

        # ResNet-18 backbone — keep pretrained weights
        import torchvision.models as tvm
        resnet = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])   # → (B, 512, 1, 1)

        self.input_proj = nn.Linear(512, 512)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=512, nhead=8, batch_first=True,
            dropout=0.1,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=3)

        # Single learned task embedding replaces the CLIP text encoder.
        # Since the instruction never changes, a trainable vector is sufficient.
        self.task_emb     = nn.Parameter(torch.zeros(1, 512))

        self.query_embed  = nn.Embedding(chunk_size, 512)
        self.action_head  = nn.Linear(512, action_dim)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        features = self.backbone(img).flatten(1)                        # (B, 512)
        features = self.input_proj(features)                            # (B, 512)
        z        = features + self.task_emb.expand(img.size(0), -1)    # (B, 512)
        queries  = self.query_embed.weight.unsqueeze(0).expand(z.size(0), -1, -1)  # (B, T, 512)
        out      = self.transformer(queries, z.unsqueeze(1))                        # (B, T, 512)
        return self.action_head(out)                    # (B, T, action_dim)


# ---------------------------------------------------------------------------
# 2. DATASET
# ---------------------------------------------------------------------------
class ACTDataset(Dataset):
    """Episode-based HDF5 dataset with optional augmentation."""

    def __init__(self, file_path: str, episode_indices, chunk_size: int = 10, augment: bool = False):
        self.augment    = augment
        self.chunk_size = chunk_size
        self.data: list = []

        # Augmentation pipeline (applied in __getitem__ on tensor)
        self.transform = T.Compose([
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.RandomResizedCrop(84, scale=(0.9, 1.0)),
        ])
        # ImageNet normalisation constants (pre-built as buffers)
        self._mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self._std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        with h5py.File(file_path, "r") as f:
            for idx in episode_indices:
                demo_id = f"demo_{idx}"
                if demo_id not in f["data"]:
                    continue
                images  = f["data"][demo_id]["image"][:]    # load entire episode
                actions = f["data"][demo_id]["action"][:]
                for t in range(len(actions) - chunk_size):
                    self.data.append((images[t], actions[t : t + chunk_size]))

    # -- fast pre-processing kept in __getitem__ so workers parallelise it --
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        img_raw, action_chunk = self.data[idx]

        # Vertical flip (domain-specific correction) + CHW float tensor
        img = torch.from_numpy(np.flipud(img_raw).copy()).float().permute(2, 0, 1) / 255.0

        if self.augment:
            img = self.transform(img)

        img = (img - self._mean) / self._std
        return img, torch.from_numpy(action_chunk).float()


# ---------------------------------------------------------------------------
# 3. TRAINING LOOP — T4-OPTIMISED
# ---------------------------------------------------------------------------
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ---- splits -----------------------------------------------------------
    all_indices   = np.arange(200)
    np.random.shuffle(all_indices)
    train_indices = all_indices[:180]
    val_indices   = all_indices[180:]

    # ---- datasets ---------------------------------------------------------
    train_dataset = ACTDataset("demo_data.h5", train_indices, augment=True)
    val_dataset   = ACTDataset("demo_data.h5", val_indices,   augment=False)

    # Optimised DataLoader settings for Colab/T4
    #   - pin_memory      → faster CPU→GPU copies via page-locked memory
    #   - num_workers=2   → Colab typically allows 2 workers safely
    #   - prefetch_factor → keep 2 batches pre-staged in host RAM
    #   - persistent_workers → avoids process respawn overhead per epoch
    loader_kwargs = dict(
        batch_size=64,          # T4 has 16 GB VRAM — double the original 32
        pin_memory=True,
        num_workers=2,
        prefetch_factor=2,
        persistent_workers=True,
    )
    train_loader = DataLoader(train_dataset, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_dataset,   shuffle=False, **loader_kwargs)

    # ---- model ------------------------------------------------------------
    model = ACTPolicy().to(device)

    # torch.compile() (PyTorch ≥ 2.0) — fuses ops & generates optimised kernels
    # Falls back gracefully on older PyTorch versions
    if hasattr(torch, "compile"):
        print("Applying torch.compile() …")
        model = torch.compile(model)

    # ---- optimiser & scheduler --------------------------------------------
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=3, factor=0.5)
    criterion = nn.MSELoss()

    # ---- AMP GradScaler ---------------------------------------------------
    # Keeps fp16 numerics stable by dynamically scaling gradients
    scaler = GradScaler(enabled=(device.type == "cuda"))

    # ---- training ---------------------------------------------------------
    train_losses, val_losses = [], []
    EPOCHS = 25
    print(f"\nStarting {EPOCHS}-epoch training sprint …\n")

    for epoch in range(EPOCHS):

        # -- TRAIN --
        model.train()
        epoch_t_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{EPOCHS} [train]", leave=False)
        for imgs, chunks in pbar:
            imgs   = imgs.to(device, non_blocking=True)    # non_blocking pairs with pin_memory
            chunks = chunks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)          # faster than zero_grad()

            # --- Mixed-precision forward pass ---
            with autocast(device_type="cuda", enabled=(device.type == "cuda")):
                preds = model(imgs)
                loss  = criterion(preds, chunks)

            # --- Scaled backward pass ---
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)   # stability
            scaler.step(optimizer)
            scaler.update()

            epoch_t_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.5f}")

        avg_t_loss = epoch_t_loss / len(train_loader)
        train_losses.append(avg_t_loss)

        # -- VALIDATE --
        model.eval()
        epoch_v_loss = 0.0

        with torch.no_grad():
            for imgs, chunks in tqdm(val_loader, desc=f"Epoch {epoch+1:02d}/{EPOCHS} [val] ", leave=False):
                imgs   = imgs.to(device, non_blocking=True)
                chunks = chunks.to(device, non_blocking=True)
                with autocast(device_type="cuda", enabled=(device.type == "cuda")):
                    preds = model(imgs)
                    epoch_v_loss += criterion(preds, chunks).item()

        avg_v_loss = epoch_v_loss / len(val_loader)
        val_losses.append(avg_v_loss)

        scheduler.step(avg_v_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1:02d}/{EPOCHS} | "
            f"Train: {avg_t_loss:.6f} | "
            f"Val: {avg_v_loss:.6f} | "
            f"LR: {current_lr:.2e}"
        )

        # Save checkpoint every epoch.
        # Use _orig_mod when model is torch.compiled to avoid key prefix issues
        # when loading into a plain (uncompiled) ACTPolicy at eval time.
        save_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        torch.save(save_model.state_dict(), "act_vla_model.pth")

    # ---- learning curve ---------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss (Augmented)")
    plt.plot(val_losses,   label="Validation Loss")
    plt.yscale("log")
    plt.title("ACT-VLA Training — T4 Optimised (180/20 Split)")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss (Log Scale)")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig("learning_curve.png", dpi=150)
    print("\nTraining complete. Plot saved → learning_curve.png")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    train()