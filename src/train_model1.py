"""
train_model1.py - 模型1（经典基线）训练脚本
传统 MFCC 特征 + 轻量级 CNN

用法:
    conda activate llmdevelop
    cd src
    python train_model1.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import (
    PROJECT_ROOT, TRAIN_AUDIO_DIR, DEV_AUDIO_DIR,
    TRAIN_PROTOCOL, DEV_PROTOCOL,
    SEED, BATCH_SIZE, NUM_EPOCHS, BACKEND_LR, WEIGHT_DECAY,
    NUM_WORKERS, EARLY_STOP_PATIENCE, GRAD_CLIP_NORM,
    CHECKPOINT_DIR, FIGURES_DIR, RESULTS_DIR
)
from utils import set_global_seed, setup_logger, save_checkpoint, EarlyStopping
from dataset import ASVspoofDataset
from models import BaselineModel
from evaluate import evaluate_model, plot_training_curves


def train_model1():
    """训练模型1：MFCC + CNN 基线模型"""
    
    # -------- 初始化 --------
    set_global_seed(SEED)
    logger = setup_logger(
        log_file=os.path.join(RESULTS_DIR, "train_model1.log")
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # -------- 数据加载 --------
    logger.info("加载数据集...")
    
    # 训练集（无数据增强）
    train_dataset = ASVspoofDataset(
        audio_dir=TRAIN_AUDIO_DIR,
        protocol_file=TRAIN_PROTOCOL,
        augment_fn=None  # 基线模型不使用增强
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS,
        pin_memory=True if device.type == "cuda" else False
    )
    
    # 验证集（无增强）
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
    
    logger.info(f"训练集: {len(train_dataset)} 条，验证集: {len(dev_dataset)} 条")
    
    # -------- 模型、优化器、调度器、损失函数 --------
    model = BaselineModel().to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=BACKEND_LR, 
        weight_decay=WEIGHT_DECAY
    )
    
    # 余弦退火学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-6
    )
    
    # 二元交叉熵损失（内置 Sigmoid，数值更稳定）
    criterion = nn.BCEWithLogitsLoss()
    
    # 早停机制
    early_stop = EarlyStopping(patience=EARLY_STOP_PATIENCE)
    
    # -------- 训练记录 --------
    history = {"train_loss": [], "val_eer": []}
    best_val_eer = float("inf")
    
    # -------- 训练循环 --------
    logger.info("=" * 50)
    logger.info("开始训练模型1：MFCC + CNN Baseline")
    logger.info("=" * 50)
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        n_batches  = 0
        
        for batch_idx, (waveforms, labels) in enumerate(train_loader):
            waveforms = waveforms.to(device)
            labels    = labels.to(device)
            
            # 前向传播
            logits = model(waveforms)
            loss   = criterion(logits, labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=GRAD_CLIP_NORM
            )
            
            optimizer.step()
            
            total_loss += loss.item()
            n_batches  += 1
            
            # 每 200 个 batch 打印一次进度
            if (batch_idx + 1) % 200 == 0:
                logger.info(
                    f"  Epoch [{epoch+1}/{NUM_EPOCHS}] "
                    f"Batch [{batch_idx+1}/{len(train_loader)}]  "
                    f"Loss: {loss.item():.4f}"
                )
        
        # 更新学习率
        scheduler.step()
        
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
        
        # 保存最佳模型
        if val_eer < best_val_eer:
            best_val_eer = val_eer
            save_checkpoint(
                model,
                path=os.path.join(CHECKPOINT_DIR, "model1_baseline.pth"),
                metadata={"epoch": epoch+1, "val_eer": val_eer, "model_name": "Model1_Baseline"}
            )
            logger.info(f"  ★ 新的最佳 EER: {best_val_eer*100:.2f}% — 模型已保存")
        
        # 早停检查
        if early_stop.step(val_eer):
            logger.info(f"早停触发！连续 {EARLY_STOP_PATIENCE} 个 Epoch 验证 EER 未降低")
            break
    
    logger.info(f"\n训练完成！最佳验证 EER: {best_val_eer*100:.2f}%")
    
    # 绘制训练曲线
    plot_training_curves(
        history=history,
        model_name="Model 1 (MFCC + CNN Baseline)",
        save_path=os.path.join(FIGURES_DIR, "training_model1.png")
    )
    
    return history, best_val_eer


if __name__ == "__main__":
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR,    exist_ok=True)
    os.makedirs(RESULTS_DIR,    exist_ok=True)
    train_model1()
