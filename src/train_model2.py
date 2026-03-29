"""
train_model2.py - 模型2（冻结大模型对照组）训练脚本
冻结的 Wav2Vec2 + ResNet 后端

用法:
    conda activate llmdevelop
    cd src
    python train_model2.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import (
    TRAIN_AUDIO_DIR, DEV_AUDIO_DIR,
    TRAIN_PROTOCOL, DEV_PROTOCOL,
    SEED, BATCH_SIZE, NUM_EPOCHS, BACKEND_LR, WEIGHT_DECAY,
    NUM_WORKERS, EARLY_STOP_PATIENCE, GRAD_CLIP_NORM,
    CHECKPOINT_DIR, FIGURES_DIR, RESULTS_DIR
)
from utils import set_global_seed, setup_logger, save_checkpoint, EarlyStopping
from dataset import ASVspoofDataset
from models import FrozenSSLModel
from evaluate import evaluate_model, plot_training_curves


def train_model2():
    """
    训练模型2：冻结 Wav2Vec2 + ResNet。
    
    与模型1的唯一区别是特征提取器从 MFCC 变为 Wav2Vec2。
    通过这一控制变量，可以纯粹归因于"自监督大模型特征是否优于手工特征"。
    """
    
    # -------- 初始化 --------
    set_global_seed(SEED)
    logger = setup_logger(
        log_file=os.path.join(RESULTS_DIR, "train_model2.log")
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # -------- 数据加载（无增强） --------
    logger.info("加载数据集...")
    
    train_dataset = ASVspoofDataset(
        audio_dir=TRAIN_AUDIO_DIR,
        protocol_file=TRAIN_PROTOCOL,
        augment_fn=None  # 无数据增强（控制变量）
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True if device.type == "cuda" else False
    )
    
    dev_dataset = ASVspoofDataset(
        audio_dir=DEV_AUDIO_DIR,
        protocol_file=DEV_PROTOCOL,
        augment_fn=None
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=NUM_WORKERS
    )
    
    # -------- 模型构建 --------
    model = FrozenSSLModel().to(device)
    
    # 关键：只优化后端 ResNet 的参数（前端已冻结）
    # 不优化冻结参数可以节省内存和计算量
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logger.info(f"可训练参数量: {sum(p.numel() for p in trainable_params):,}")
    
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=BACKEND_LR,
        weight_decay=WEIGHT_DECAY
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-6
    )
    
    criterion = nn.BCEWithLogitsLoss()
    early_stop = EarlyStopping(patience=EARLY_STOP_PATIENCE)
    
    # -------- 训练记录 --------
    history = {"train_loss": [], "val_eer": []}
    best_val_eer = float("inf")
    
    # -------- 训练循环 --------
    logger.info("=" * 50)
    logger.info("开始训练模型2：冻结 Wav2Vec2 + ResNet")
    logger.info("=" * 50)
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        n_batches  = 0
        
        for batch_idx, (waveforms, labels) in enumerate(train_loader):
            waveforms = waveforms.to(device)
            labels    = labels.to(device)
            
            # 前向传播Wav2Vec2 在 with torch.no_grad() 下运行，节省显存）
            logits = model(waveforms)
            
            # NaN 防护：检查输出是否包含 NaN
            if torch.isnan(logits).any():
                logger.warning(f"  ⚠ Epoch [{epoch+1}/{NUM_EPOCHS}] Batch [{batch_idx+1}] 检测到 NaN logits，跳过该 batch")
                continue
            
            loss   = criterion(logits, labels)
            
            # NaN 防护：检查 loss 是否为 NaN
            if torch.isnan(loss):
                logger.warning(f"  ⚠ Epoch [{epoch+1}/{NUM_EPOCHS}] Batch [{batch_idx+1}] Loss 为 NaN，跳过该 batch")
                continue
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=GRAD_CLIP_NORM)
            optimizer.step()
            
            total_loss += loss.item()
            n_batches  += 1
            
            if (batch_idx + 1) % 100 == 0:
                logger.info(
                    f"  Epoch [{epoch+1}/{NUM_EPOCHS}] "
                    f"Batch [{batch_idx+1}/{len(train_loader)}]  "
                    f"Loss: {loss.item():.4f}"
                )
        
        scheduler.step()
        
        # 防止整个 epoch 中所有 batch 均因 NaN 被跳过
        if n_batches == 0:
            logger.error(f"Epoch [{epoch+1}/{NUM_EPOCHS}] 所有 batch 均为 NaN，请检查模型或数据。")
            break
        
        # 验证集评估
        val_eer, _, _, _ = evaluate_model(model, dev_loader, device, desc="验证中")
        
        avg_loss = total_loss / n_batches
        history["train_loss"].append(avg_loss)
        history["val_eer"].append(val_eer)
        
        logger.info(
            f"Epoch [{epoch+1:2d}/{NUM_EPOCHS}] | "
            f"Loss: {avg_loss:.4f} | "
            f"Val EER: {val_eer*100:.2f}% | "
            f"LR: {scheduler.get_last_lr()[0]:.2e}"
        )
        
        if val_eer < best_val_eer:
            best_val_eer = val_eer
            save_checkpoint(
                model,
                path=os.path.join(CHECKPOINT_DIR, "model2_frozen_ssl.pth"),
                metadata={"epoch": epoch+1, "val_eer": val_eer, "model_name": "Model2_FrozenSSL"}
            )
            logger.info(f"  ★ 新的最佳 EER: {best_val_eer*100:.2f}% — 模型已保存")
        
        if early_stop.step(val_eer):
            logger.info(f"早停触发！")
            break
    
    logger.info(f"\n训练完成！最佳验证 EER: {best_val_eer*100:.2f}%")
    
    plot_training_curves(
        history=history,
        model_name="Model 2 (Frozen Wav2Vec2 + ResNet)",
        save_path=os.path.join(FIGURES_DIR, "training_model2.png")
    )
    
    return history, best_val_eer


if __name__ == "__main__":
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR,    exist_ok=True)
    os.makedirs(RESULTS_DIR,    exist_ok=True)
    train_model2()
