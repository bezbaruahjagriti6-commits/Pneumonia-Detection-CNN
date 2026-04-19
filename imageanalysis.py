import os

import ast
import random
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
# =========================================================
# CONFIG
# =========================================================
TaskType = Literal["binary", "multiclass", "multilabel", "regression"]


@dataclass
class Config:
    # Dataset CSVs (The ones you just created!)
    train_csv: str = "train.csv"
    val_csv: str = "val.csv"

    # CSV columns
    image_col: str = "image_path"
    label_col: str = "label"

    # Task setup
    task_type: TaskType = "binary"       # Switched to binary!
    num_classes: int = 1                 # Binary only needs 1 output (0 or 1)
    in_channels: int = 3

    # Image / loader
    image_size: int = 224
    batch_size: int = 16                 # 16 is a very safe number for your 6GB RTX 4050
    num_workers: int = 4

    # Training
    epochs: int = 10                     # Let's do 10 passes through the data to start
    learning_rate: float = 1e-4          
    weight_decay: float = 1e-4
    dropout: float = 0.4                 
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Checkpoint
    checkpoint_path: str = "best_pneumonia_model.pt" # Name of your saved brain!

    # Inference test image 
    sample_infer_image: str = ""

# =========================================================
# UTILS
# =========================================================
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(model: nn.Module, path: str) -> None:
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(model: nn.Module, path: str, device: str) -> nn.Module:
    model.load_state_dict(torch.load(path, map_location=device))
    return model


def get_loss_fn(task_type: str, pos_weight=None):
    if task_type == "binary":
        # Pass the penalty weight into the binary loss function!
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if task_type == "multiclass":
        return nn.CrossEntropyLoss()
    if task_type == "multilabel":
        return nn.BCEWithLogitsLoss()
    if task_type == "regression":
        return nn.MSELoss()
    raise ValueError(f"Unsupported task_type: {task_type}")

def compute_metric(logits: torch.Tensor, targets: torch.Tensor, task_type: str) -> float:
    if task_type == "binary":
        preds = (torch.sigmoid(logits).squeeze(1) > 0.5).long()
        truth = targets.long()
        return (preds == truth).float().mean().item()

    if task_type == "multiclass":
        preds = torch.argmax(logits, dim=1)
        return (preds == targets).float().mean().item()

    if task_type == "multilabel":
        preds = (torch.sigmoid(logits) > 0.5).float()
        return (preds == targets).float().mean().item()

    if task_type == "regression":
        mse = torch.mean((logits.squeeze() - targets) ** 2).item()
        return mse

    raise ValueError(f"Unsupported task_type: {task_type}")


# =========================================================
# DATASET
# =========================================================
class ImageAnalysisDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        image_col: str,
        label_col: str,
        image_size: int,
        task_type: str,
        train: bool = True,
    ):
        self.df = pd.read_csv(csv_path)
        self.image_col = image_col
        self.label_col = label_col
        self.task_type = task_type

        if train:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

    def __len__(self) -> int:
        return len(self.df)

    def _parse_label(self, label_value):
        if self.task_type == "binary":
            return torch.tensor(float(label_value), dtype=torch.float32)

        if self.task_type == "multiclass":
            return torch.tensor(int(label_value), dtype=torch.long)

        if self.task_type == "regression":
            return torch.tensor(float(label_value), dtype=torch.float32)

        if self.task_type == "multilabel":
            if isinstance(label_value, str):
                if label_value.startswith("["):
                    parsed = ast.literal_eval(label_value)
                else:
                    parsed = [float(x) for x in label_value.split(",")]
            else:
                parsed = label_value
            return torch.tensor(parsed, dtype=torch.float32)

        raise ValueError(f"Unsupported task_type: {self.task_type}")

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_path = row[self.image_col]
        label_value = row[self.label_col]

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        label = self._parse_label(label_value)

        return image, label


# =========================================================
# MODEL
# =========================================================
class ConvBNAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        weights = self.pool(x).view(b, c)
        weights = self.fc(weights).view(b, c, 1, 1)
        return x * weights


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = ConvBNAct(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.se = SEBlock(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.se(out)
        out = out + identity
        out = self.act(out)
        return out


class CNNBackbone(nn.Module):
    def __init__(self, in_channels: int = 3):
        super().__init__()

        self.stem = nn.Sequential(
            ConvBNAct(in_channels, 32, kernel_size=3, stride=2, padding=1),
            ConvBNAct(32, 64, kernel_size=3, stride=1, padding=1),
        )

        self.stage1 = nn.Sequential(
            ResidualBlock(64, 64, stride=1),
            ResidualBlock(64, 64, stride=1),
        )
        self.stage2 = nn.Sequential(
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128, stride=1),
        )
        self.stage3 = nn.Sequential(
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256, stride=1),
        )
        self.stage4 = nn.Sequential(
            ResidualBlock(256, 512, stride=2),
            ResidualBlock(512, 512, stride=1),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feature_dim = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x).flatten(1)
        return x


class UniversalCNN(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, task_type: str = "multiclass", dropout: float = 0.3):
        super().__init__()
        self.task_type = task_type
        self.backbone = CNNBackbone(in_channels=in_channels)

        out_dim = 1 if task_type in ["binary", "regression"] else num_classes

        self.head = nn.Sequential(
            nn.Linear(self.backbone.feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits = self.head(features)
        return logits


# =========================================================
# TRAIN / VALIDATE
# =========================================================
def train_one_epoch(model, loader, optimizer, loss_fn, device, task_type):
    model.train()
    total_loss = 0.0
    total_metric = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)

        if task_type in ["binary", "regression"]:
            loss = loss_fn(logits.squeeze(1), labels)
        else:
            loss = loss_fn(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_metric += compute_metric(logits.detach(), labels.detach(), task_type)

    return total_loss / len(loader), total_metric / len(loader)


@torch.no_grad()
def validate(model, loader, loss_fn, device, task_type):
    model.eval()
    total_loss = 0.0
    total_metric = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)

        if task_type in ["binary", "regression"]:
            loss = loss_fn(logits.squeeze(1), labels)
        else:
            loss = loss_fn(logits, labels)

        total_loss += loss.item()
        total_metric += compute_metric(logits, labels, task_type)

    return total_loss / len(loader), total_metric / len(loader)


def run_training(cfg: Config):
    set_seed(cfg.seed)

    # ==========================================
    # NEW: Calculate Class Weights automatically
    # ==========================================
    train_df = pd.read_csv(cfg.train_csv)
    num_normal = len(train_df[train_df[cfg.label_col] == 0])
    num_pneumonia = len(train_df[train_df[cfg.label_col] == 1])
    
    print(f"\n📊 Training Data: {num_normal} Normal | {num_pneumonia} Pneumonia")
    
    # Formula for PyTorch binary weights: Negative Samples / Positive Samples
    weight_ratio = num_normal / num_pneumonia
    pos_weight_tensor = torch.tensor([weight_ratio], dtype=torch.float32).to(cfg.device)
    print(f"⚖️ Applied Class Weight Penalty: {weight_ratio:.4f}\n")
    # ==========================================

    train_dataset = ImageAnalysisDataset(
        csv_path=cfg.train_csv,
        image_col=cfg.image_col,
        label_col=cfg.label_col,
        image_size=cfg.image_size,
        task_type=cfg.task_type,
        train=True,
    )

    val_dataset = ImageAnalysisDataset(
        csv_path=cfg.val_csv,
        image_col=cfg.image_col,
        label_col=cfg.label_col,
        image_size=cfg.image_size,
        task_type=cfg.task_type,
        train=False,
    )

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    model = UniversalCNN(
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        task_type=cfg.task_type,
        dropout=cfg.dropout,
    ).to(cfg.device)

    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    
    # Pass our newly calculated weights into the loss function!
    loss_fn = get_loss_fn(cfg.task_type, pos_weight=pos_weight_tensor)

    best_score = float("-inf") if cfg.task_type != "regression" else float("inf")

    for epoch in range(cfg.epochs):
        train_loss, train_metric = train_one_epoch(model, train_loader, optimizer, loss_fn, cfg.device, cfg.task_type)
        val_loss, val_metric = validate(model, val_loader, loss_fn, cfg.device, cfg.task_type)

        print(
            f"Epoch [{epoch + 1}/{cfg.epochs}] | "
            f"Train Loss: {train_loss:.4f} | Train Metric: {train_metric:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Metric: {val_metric:.4f}"
        )

        is_better = (val_metric > best_score) if cfg.task_type != "regression" else (val_metric < best_score)
        if is_better:
            best_score = val_metric
            save_checkpoint(model, cfg.checkpoint_path)
            print(f"Saved best model to {cfg.checkpoint_path}")

    return model

# =========================================================
# INFERENCE
# =========================================================
@torch.no_grad()
def predict_image(image_path: str, cfg: Config):
    model = UniversalCNN(
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        task_type=cfg.task_type,
        dropout=cfg.dropout,
    ).to(cfg.device)

    model = load_checkpoint(model, cfg.checkpoint_path, cfg.device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    image = Image.open(image_path).convert("RGB")
    x = transform(image).unsqueeze(0).to(cfg.device)

    logits = model(x)

    if cfg.task_type == "binary":
        prob = torch.sigmoid(logits).item()
        pred = 1 if prob > 0.5 else 0
        return {"prediction": pred, "probability": prob}

    if cfg.task_type == "multiclass":
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        return {
            "prediction": pred,
            "probabilities": probs.squeeze(0).cpu().tolist(),
        }

    if cfg.task_type == "multilabel":
        probs = torch.sigmoid(logits).squeeze(0)
        preds = (probs > 0.5).int().cpu().tolist()
        return {
            "prediction": preds,
            "probabilities": probs.cpu().tolist(),
        }

    if cfg.task_type == "regression":
        value = logits.squeeze().item()
        return {"prediction": value}

    raise ValueError(f"Unsupported task_type: {cfg.task_type}")
@torch.no_grad()
def plot_confusion_matrix(model, loader, device):
    print("\n--- 📊 Generating Confusion Matrix ---")
    model.eval() # Freeze the AI's brain for testing
    
    all_preds = []
    all_labels = []

    # Feed the test images to the AI and record its guesses
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        
        # Convert the raw math probabilities into 0 (Normal) or 1 (Pneumonia)
        preds = (torch.sigmoid(logits).squeeze(1) > 0.5).long()
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # Calculate the exact numbers for the 2x2 grid
    cm = confusion_matrix(all_labels, all_preds)
    
    # Draw the visual graph
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal (0)", "Pneumonia (1)"])
    disp.plot(cmap="Blues", values_format="d")
    
    plt.title("Final Test - Confusion Matrix")
    
    # Save a high-quality picture of the graph for your presentation slide!
    plt.savefig("presentation_confusion_matrix.png")
    print("✅ Saved graphic as 'presentation_confusion_matrix.png'")
    
    # Pop the graph up on your screen
    plt.show()

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    cfg = Config()

    print(f"🚀 Running on device: {cfg.device}")
    print(f"🧠 Task type: {cfg.task_type}")

    # 1. Train the model (This will overwrite the old saved file with new, better versions)
    print("\n--- 📚 Starting Training ---")
    model = run_training(cfg)

    # 2. Evaluate on the Unseen Final Exam (Test Set)
    print("\n--- 🎓 Running Final Evaluation on Test Set ---")
    
    # Load the absolute best saved "brain" from the training phase we just finished
    model = load_checkpoint(model, cfg.checkpoint_path, cfg.device)
    loss_fn = get_loss_fn(cfg.task_type)
    
    # Create the final exam test loader
    test_dataset = ImageAnalysisDataset(
        csv_path="test.csv",
        image_col=cfg.image_col,
        label_col=cfg.label_col,
        image_size=cfg.image_size,
        task_type=cfg.task_type,
        train=False, 
    )
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)
    
    test_loss, test_metric = validate(model, test_loader, loss_fn, cfg.device, cfg.task_type)
    print(f"🎯 Final Test Loss: {test_loss:.4f} | Final Test Accuracy: {test_metric*100:.2f}%")

# Generate the visual Confusion Matrix
    plot_confusion_matrix(model, test_loader, cfg.device)

    # 3. Quick Real-World Test!
    print("\n--- 🔍 Looking at a single random Test Image ---")
    test_df = pd.read_csv("test.csv")
    
    # Grab the very first image in the test set
    sample_img = test_df.iloc[0]['image_path']
    actual_label = test_df.iloc[0]['label']
    
    cfg.sample_infer_image = sample_img
    result = predict_image(cfg.sample_infer_image, cfg)
    
    print(f"X-Ray Image File: {sample_img}")
    print(f"Actual Truth: {'Pneumonia (1)' if actual_label == 1 else 'Normal (0)'}")
    print(f"AI Prediction: {'Pneumonia (1)' if result['prediction'] == 1 else 'Normal (0)'}")
    print(f"AI Confidence: {result['probability']*100:.2f}%")
