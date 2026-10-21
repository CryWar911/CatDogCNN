"""Entrena una CNN desde cero para clasificar gatos vs perros y la exporta a ONNX
para usarla en el navegador (docs/ -> GitHub Pages).

Uso:
    python train/prepare_data.py
    python train/train.py --epochs 30
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(__file__)
CACHE = os.path.join(HERE, "cache")
DOCS = os.path.join(HERE, "..", "docs")
MODEL_DIR = os.path.join(DOCS, "model")
CLASSES = ["gato", "perro"]
IMG_SIZE = 128


def block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class CatDogCNN(nn.Module):
    """4 bloques convolucionales (32-64-128-256) + Global Average Pooling + capa densa."""

    def __init__(self):
        super().__init__()
        self.b1 = block(3, 32)     # 128 -> 64
        self.b2 = block(32, 64)    # 64  -> 32
        self.b3 = block(64, 128)   # 32  -> 16
        self.b4 = block(128, 256)  # 16  -> 8
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(256, 2)

    def features(self, x):
        a1 = self.b1(x)
        a2 = self.b2(a1)
        a3 = self.b3(a2)
        a4 = self.b4(a3)
        return a1, a2, a3, a4

    def forward(self, x):
        a4 = self.features(x)[-1]
        return self.fc(self.dropout(a4.mean((2, 3))))


class ExportWrapper(nn.Module):
    """Modelo para el navegador: entrada RGB en [0,1] (N,3,128,128).

    Salidas: probabilidades, activaciones de cada bloque y el Class Activation Map
    (CAM) de cada clase, que muestra en que zonas de la imagen se fija la red.
    """

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = (x - 0.5) / 0.5
        a1, a2, a3, a4 = self.net.features(x)
        logits = self.net.fc(a4.mean((2, 3)))
        probs = torch.softmax(logits, dim=1)
        cam = torch.einsum("kc,nchw->nkhw", self.net.fc.weight, a4)
        return probs, a1, a2, a3, a4, cam


def augment(x):
    """Aumentacion de datos en GPU. x: float (N,3,H,W) en [0,1]."""
    n = x.shape[0]
    dev = x.device
    # Transformacion afin aleatoria: rotacion, escala, traslacion y espejo horizontal
    ang = (torch.rand(n, device=dev) - 0.5) * math.radians(30)
    scale = 1 / (0.8 + torch.rand(n, device=dev) * 0.35)
    flip = torch.where(torch.rand(n, device=dev) < 0.5, -1.0, 1.0)
    tx, ty = [(torch.rand(n, device=dev) - 0.5) * 0.2 for _ in range(2)]
    cos, sin = torch.cos(ang) * scale, torch.sin(ang) * scale
    theta = torch.stack([
        torch.stack([cos * flip, -sin, tx], 1),
        torch.stack([sin * flip, cos, ty], 1),
    ], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, padding_mode="reflection", align_corners=False)
    # Brillo, contraste y saturacion
    b = 1 + (torch.rand(n, 1, 1, 1, device=dev) - 0.5) * 0.4
    c = 1 + (torch.rand(n, 1, 1, 1, device=dev) - 0.5) * 0.4
    s = 1 + (torch.rand(n, 1, 1, 1, device=dev) - 0.5) * 0.4
    mean = x.mean((1, 2, 3), keepdim=True)
    x = (x - mean) * c + mean
    gray = x.mean(1, keepdim=True)
    x = (x - gray) * s + gray
    return (x * b).clamp(0, 1)


def to_input(xb):
    return xb.permute(0, 3, 1, 2).float().div_(255)


@torch.no_grad()
def evaluate(model, x, y, bs=512):
    model.eval()
    preds = []
    for i in range(0, len(x), bs):
        xb = (to_input(x[i:i + bs]) - 0.5) / 0.5
        with torch.autocast("cuda", dtype=torch.bfloat16):
            preds.append(model(xb).argmax(1))
    preds = torch.cat(preds)
    acc = (preds == y).float().mean().item()
    cm = torch.zeros(2, 2, dtype=torch.long)
    for t, p in zip(y.tolist(), preds.tolist()):
        cm[t, p] += 1
    return acc, cm.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    args = ap.parse_args()

    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.from_numpy(np.load(os.path.join(CACHE, "x.npy")))
    y = torch.from_numpy(np.load(os.path.join(CACHE, "y.npy")))

    # Particion 80% entrenamiento / 10% validacion / 10% prueba
    g = torch.Generator().manual_seed(42)
    perm = torch.randperm(len(x), generator=g)
    n_val = n_test = len(x) // 10
    idx_test, idx_val, idx_train = perm[:n_test], perm[n_test:n_test + n_val], perm[n_test + n_val:]
    xtr, ytr = x[idx_train].to(dev), y[idx_train].to(dev)
    xva, yva = x[idx_val].to(dev), y[idx_val].to(dev)
    xte, yte = x[idx_test].to(dev), y[idx_test].to(dev)
    print(f"train {len(xtr)}  val {len(xva)}  test {len(xte)}  device {dev}")

    model = CatDogCNN().to(dev).to(memory_format=torch.channels_last)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parametros: {n_params:,}")

    steps = math.ceil(len(xtr) / args.bs)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * steps, pct_start=0.2)
    history = []
    best_acc, best_state = 0.0, None

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        order = torch.randperm(len(xtr), device=dev)
        tot_loss = tot_ok = 0
        for i in range(steps):
            idx = order[i * args.bs:(i + 1) * args.bs]
            xb = augment(to_input(xtr[idx]))
            xb = ((xb - 0.5) / 0.5).contiguous(memory_format=torch.channels_last)
            yb = ytr[idx]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(xb)
                loss = F.cross_entropy(out, yb, label_smoothing=0.05)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot_loss += loss.item() * len(idx)
            tot_ok += (out.argmax(1) == yb).sum().item()
        train_loss, train_acc = tot_loss / len(xtr), tot_ok / len(xtr)
        val_acc, _ = evaluate(model, xva, yva)
        history.append({"epoch": epoch, "train_loss": round(train_loss, 4),
                        "train_acc": round(train_acc, 4), "val_acc": round(val_acc, 4)})
        print(f"epoca {epoch:2d}  loss {train_loss:.4f}  train {train_acc:.4f}  val {val_acc:.4f}  ({time.time() - t0:.1f}s)")
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    test_acc, cm = evaluate(model, xte, yte)
    print(f"Mejor val {best_acc:.4f}  ->  test {test_acc:.4f}")
    print("Matriz de confusion (filas=real, cols=pred):", cm)

    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(best_state, os.path.join(HERE, "catdog_cnn.pt"))

    # Exportar a ONNX para onnxruntime-web
    model = model.float().to("cpu").to(memory_format=torch.contiguous_format).eval()
    wrapper = ExportWrapper(model).eval()
    dummy = torch.rand(1, 3, IMG_SIZE, IMG_SIZE)
    torch.onnx.export(
        wrapper, dummy, os.path.join(MODEL_DIR, "catdog_cnn.onnx"),
        input_names=["image"], output_names=["probs", "act1", "act2", "act3", "act4", "cam"],
        opset_version=17, dynamo=False,
    )

    # Filtros de la primera capa convolucional (para visualizarlos en la web)
    w1 = model.b1[0].weight.detach().numpy()  # (32,3,3,3)
    meta = {
        "classes": CLASSES,
        "img_size": IMG_SIZE,
        "params": n_params,
        "epochs": args.epochs,
        "best_val_acc": round(best_acc, 4),
        "test_acc": round(test_acc, 4),
        "confusion_matrix": cm,
        "split": {"train": len(xtr), "val": len(xva), "test": len(xte)},
        "history": history,
        "conv1_filters": np.round(w1, 4).tolist(),
        "fc_weight": np.round(model.fc.weight.detach().numpy(), 4).tolist(),
    }
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)

    # Imagenes de ejemplo del conjunto de prueba
    ex_dir = os.path.join(DOCS, "examples")
    os.makedirs(ex_dir, exist_ok=True)
    from PIL import Image
    xte_cpu, yte_cpu = xte.cpu().numpy(), yte.cpu().numpy()
    names = []
    for cls in (0, 1):
        for j, k in enumerate(np.where(yte_cpu == cls)[0][:6]):
            name = f"{CLASSES[cls]}_{j + 1}.jpg"
            Image.fromarray(xte_cpu[k]).save(os.path.join(ex_dir, name), quality=90)
            names.append(name)
    with open(os.path.join(ex_dir, "index.json"), "w") as f:
        json.dump(names, f)
    print("Exportado a", os.path.abspath(MODEL_DIR))


if __name__ == "__main__":
    main()
