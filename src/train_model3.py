"""
train_model3.py - 模型3（终极创新方案）训练脚本
渐进式解冻微调 Wav2Vec2 + ResNet + GPU 实时数据增强

用法:
    conda activate llmdevelop
    cd src
    python train_model3.py
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
    AUG_PROB, SNR_MIN, SNR_MAX,
    UNFREEZE_EPOCH_1, UNFREEZE_EPOCH_2,
    CHECKPOINT_DIR, FIGURES_DIR, RESULTS_DIR
)
from utils import set_global_seed, setup_logger, save_checkpoint, EarlyStopping
from dataset import ASVspoofDataset, OnTheFlyAugmentor
from models import FineTunedSSLModel, progressive_unfreeze
from evaluate import evaluate_model, plot_training_curves

# 模型3 专用配置：解冻后显存要求大，使用小一层的 batch_size + 梯度累积
MODEL3_BATCH_SIZE = max(16, BATCH_SIZE // 4)         # 16（防止解冻后 OOM）
GRAD_ACCUM_STEPS  = BATCH_SIZE // MODEL3_BATCH_SIZE  # 梯度累积步数，等效 batch=64


def train_model3():
    """
    训练模型3：渐进式解冻微调 Wav2Vec2 + ResNet + 数据增强。
    
    与模型2的差异点（确保论证的"控制变量"严谨性）：
        ① 训练数据使用 OnTheFlyAugmentor 进行实时增强
        ② 训练过程中在 Epoch 5 和 8 执行渐进式解冻
    
    核心技巧：
        - 梯度裁剪（解冻后初始梯度可能很大）
        - 分层学习率（前端 1e-5，后端 1e-3，差 100 倍）
        - 早停策略避免过拟合
    """
    
    # -------- 初始化 --------
    set_global_seed(SEED)
    logger = setup_logger(
        log_file=os.path.join(RESULTS_DIR, "train_model3.log")
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # -------- 数据加载（含实时增强） --------
    logger.info("初始化数据增强器...")
    
    augmentor = OnTheFlyAugmentor(
        p=AUG_PROB,
        snr_range=(SNR_MIN, SNR_MAX)
    )
    logger.info(f"增强参数: p={AUG_PROB}, SNR 范围=[{SNR_MIN},{SNR_MAX}] dB")
    
    # 训练集：启用数据增强
    train_dataset = ASVspoofDataset(
        audio_dir=TRAIN_AUDIO_DIR,
        protocol_file=TRAIN_PROTOCOL,
        augment_fn=augmentor  # 启用增强！（与模型2的核心区别之一）
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=MODEL3_BATCH_SIZE,   # 缩小单步 batch，防 OOM
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True if device.type == "cuda" else False
    )
    
    # 验证集：无增强（客观评估）
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
    
    logger.info(f"训练集: {len(train_dataset)} 条（含实时增强）| 验证集: {len(dev_dataset)} 条")
    logger.info(f"模型3 Batch 配置: 单步 batch={MODEL3_BATCH_SIZE}, 梯度累积步数={GRAD_ACCUM_STEPS}, 等效 batch={MODEL3_BATCH_SIZE*GRAD_ACCUM_STEPS}")
    
    # -------- 模型构建 --------
    model = FineTunedSSLModel().to(device)
    
    # 初始只优化后端参数（Wav2Vec2 初始全部冻结）
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=BACKEND_LR,
        weight_decay=WEIGHT_DECAY
    )
    
    # 余弦退火调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-6
    )
    
    criterion = nn.BCEWithLogitsLoss()
    early_stop = EarlyStopping(patience=EARLY_STOP_PATIENCE)
    
    # -------- 训练记录 --------
    history = {"train_loss": [], "val_eer": []}
    best_val_eer = float("inf")
    unfreeze_log = []  # 记录解冻事件（用于在曲线图上标注）
    
    # -------- 训练循环 --------
    logger.info("=" * 60)
    logger.info("开始训练模型3：微调 Wav2Vec2 + ResNet + 数据增强")
    logger.info(f"解冻计划: Epoch {UNFREEZE_EPOCH_1} 解冻最后2层，Epoch {UNFREEZE_EPOCH_2} 解冻最后4层")
    logger.info("=" * 60)
    
    for epoch in range(NUM_EPOCHS):
        
        # ★ 渐进式解冻（与模型2的核心区别之二）
        progressive_unfreeze(model, epoch, optimizer)
        
        model.train()
        total_loss = 0.0
        n_batches  = 0
        
        optimizer.zero_grad()  # 在 epoch 开始时清零（梯度累积模式）
        
        for batch_idx, (waveforms, labels) in enumerate(train_loader):
            waveforms = waveforms.to(device)
            labels    = labels.to(device)
            
            # 前向传播（梯度累积：loss 需要除以累积步数）
            logits = model(waveforms)
            loss   = criterion(logits, labels) / GRAD_ACCUM_STEPS
            
            # 反向传播（累积梯度）
            loss.backward()
            
            total_loss += loss.item() * GRAD_ACCUM_STEPS  # 还原实际 loss 值记录
            n_batches  += 1
            
            # 每 GRAD_ACCUM_STEPS 步执行一次参数更新
            if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
                # 梯度裁剪（解冻后初始梯度可能极大，必须裁剪）
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=GRAD_CLIP_NORM
                )
                optimizer.step()
                optimizer.zero_grad()
            
            if (batch_idx + 1) % 100 == 0:
                logger.info(
                    f"  Epoch [{epoch+1}/{NUM_EPOCHS}] "
                    f"Batch [{batch_idx+1}/{len(train_loader)}]  "
                    f"Loss: {loss.item()*GRAD_ACCUM_STEPS:.4f}"
                )
        
        # 处理最后一个不完整的累积步
        remaining = len(train_loader) % GRAD_ACCUM_STEPS
        if remaining != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
            optimizer.step()
            optimizer.zero_grad()
        
        scheduler.step()
        
        # 验证集评估
        val_eer, _, _, _ = evaluate_model(model, dev_loader, device, desc="验证中")
        
        avg_loss = total_loss / n_batches
        history["train_loss"].append(avg_loss)
        history["val_eer"].append(val_eer)
        
        # 记录解冻 epoch
        if epoch in [UNFREEZE_EPOCH_1, UNFREEZE_EPOCH_2]:
            unfreeze_log.append(epoch)
        
        logger.info(
            f"Epoch [{epoch+1:2d}/{NUM_EPOCHS}] | "
            f"Loss: {avg_loss:.4f} | "
            f"Val EER: {val_eer*100:.2f}% | "
            f"LR(backend): {optimizer.param_groups[0]['lr']:.2e}"
        )
        
        if val_eer < best_val_eer:
            best_val_eer = val_eer
            save_checkpoint(
                model,
                path=os.path.join(CHECKPOINT_DIR, "model3_finetuned_aug.pth"),
                metadata={
                    "epoch": epoch+1,
                    "val_eer": val_eer,
                    "model_name": "Model3_FineTuned",
                    "unfreeze_epochs": unfreeze_log
                }
            )
            logger.info(f"  ★ 新的最佳 EER: {best_val_eer*100:.2f}% — 模型已保存")
        
        if early_stop.step(val_eer):
            logger.info(f"早停触发！")
            break
    
    logger.info(f"\n训练完成！最佳验证 EER: {best_val_eer*100:.2f}%")
    
    plot_training_curves(
        history=history,
        model_name="Model 3 (Fine-tuned Wav2Vec2 + ResNet + Augmentation)",
        save_path=os.path.join(FIGURES_DIR, "training_model3.png")
    )
    
    return history, best_val_eer


if __name__ == "__main__":
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR,    exist_ok=True)
    os.makedirs(RESULTS_DIR,    exist_ok=True)
    train_model3()
