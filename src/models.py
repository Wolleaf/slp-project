"""
models.py - 三个对比模型的定义
包含：
  - BaselineModel: 模型1，传统 MFCC + 轻量级 CNN
  - ResNetBackend: 共享后端分类器（模型2/3使用）
  - FrozenSSLModel: 模型2，冻结 Wav2Vec2 + ResNet
  - FineTunedSSLModel: 模型3，可微调 Wav2Vec2 + ResNet + 渐进式解冻
  - progressive_unfreeze: 渐进式解冻策略函数
"""

import torch
import torch.nn as nn
import torchaudio.transforms as T
from transformers import Wav2Vec2Model, Wav2Vec2Config

from config import (
    SAMPLE_RATE, N_MFCC, N_FFT, HOP_LENGTH, N_MELS,
    SSL_MODEL_NAME, SSL_FEATURE_DIM,
    UNFREEZE_EPOCH_1, UNFREEZE_EPOCH_2,
    UNFREEZE_LAYERS_1, UNFREEZE_LAYERS_2,
    SSL_LR
)


# ============================================================
# 模型 1：传统 MFCC + 轻量级 CNN（Baseline）
# ============================================================

class BaselineModel(nn.Module):
    """
    模型1：传统声学特征 + 轻量级 CNN 分类器。
    
    代表"前代/旧的技术方案"，作为消融实验的最低性能参照。
    
    架构说明：
        前端：MFCC 特征提取（非学习型，手工设计）
        后端：4 层 CNN + 全连接层（参数量约 100K 量级）
    
    使命：证明传统特征方案的局限性（EER 较高），为引入大模型奠定对比基础。
    """
    
    def __init__(self, n_mfcc: int = N_MFCC):
        super().__init__()
        
        # 前端：MFCC 特征提取（模拟人耳听觉感知的频谱包络）
        self.mfcc_transform = T.MFCC(
            sample_rate=SAMPLE_RATE,
            n_mfcc=n_mfcc,
            melkwargs={
                "n_fft": N_FFT,
                "hop_length": HOP_LENGTH,
                "n_mels": N_MELS,
                "f_min": 20.0,
                "f_max": 8000.0
            }
        )
        
        # 后端：轻量级 CNN 分类器
        # 输入维度：(batch, 1, n_mfcc=40, time_frames≈400)
        self.cnn = nn.Sequential(
            # 第1块：提取底层纹理特征
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 降采样 2x
            nn.Dropout2d(0.1),
            
            # 第2块：提取中层特征组合
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 再次降采样 2x
            nn.Dropout2d(0.1),
            
            # 第3块：提取高层语义特征
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 全局平均池化：将任意大小的特征图压缩为 (batch, 128, 1, 1)
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # 全连接分类头：128 → 64 → 1（输出 logit）
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
        
        # 统计并打印模型参数量
        total_params = sum(p.numel() for p in self.parameters())
        print(f"[模型1] BaselineModel 参数量: {total_params:,}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。
        
        参数:
            x: 原始音频张量, shape=(batch, num_samples)
        
        返回:
            logits: 未经激活的分类分数, shape=(batch,)
        """
        # MFCC 特征提取：(batch, num_samples) → (batch, n_mfcc, time_frames)
        mfcc = self.mfcc_transform(x)
        
        # 添加通道维度（CNN 期望 4D 输入）：→ (batch, 1, n_mfcc, time_frames)
        mfcc = mfcc.unsqueeze(1)
        
        # CNN 特征提取：→ (batch, 128, 1, 1)
        features = self.cnn(mfcc)
        
        # 展平：→ (batch, 128)
        features = features.view(features.size(0), -1)
        
        # 分类：→ (batch, 1) → (batch,)
        logits = self.classifier(features)
        return logits.squeeze(-1)


# ============================================================
# 共享后端：ResNet 风格分类器（模型2/3使用）
# ============================================================

class ResNetBackend(nn.Module):
    """
    ResNet 风格的后端分类器，供模型2和模型3共享使用。
    
    接收 Wav2Vec2 输出的高维特征序列 (batch, time_frames, feature_dim)，
    先在时间轴进行均值池化得到全局表示，
    再通过带残差连接的 MLP 输出二分类 logit。
    
    相比 2D CNN 方案的优势：
      - 显存占用大幅减少（避免对 (1, 1024, T) 大特征图做卷积）
      - 训练更稳定（MLP+BN 对 1D 特征更适合）
      - 参数量相当，但利用率更高
    
    输入维度：(batch, time_frames, feature_dim)
    """
    
    def __init__(self, input_dim: int = SSL_FEATURE_DIM):
        super().__init__()
        
        # 时间轴统计特征：均值池化后的特征经过 BN 归一化
        self.input_bn = nn.BatchNorm1d(input_dim)
        
        # 第1个残差块：feature_dim → 512
        self.block1 = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
        )
        self.proj1 = nn.Linear(input_dim, 512)  # 残差投影
        self.act1  = nn.GELU()
        
        # 第2个残差块：512 → 256
        self.block2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
        )
        self.proj2 = nn.Linear(512, 256)  # 残差投影
        self.act2  = nn.GELU()
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: 特征张量, shape=(batch, time_frames, feature_dim)
        
        返回:
            logits: shape=(batch,)
        """
        # 时间轴均值池化：(batch, T, D) → (batch, D)
        x = x.mean(dim=1)
        
        # 输入层归一化
        x = self.input_bn(x)
        
        # 残差块1
        identity = self.proj1(x)
        x = self.act1(self.block1(x) + identity)  # → (batch, 512)
        
        # 残差块2
        identity = self.proj2(x)
        x = self.act2(self.block2(x) + identity)  # → (batch, 256)
        
        # 分类输出
        return self.classifier(x).squeeze(-1)      # → (batch,)


# ============================================================
# 模型 2：冻结 Wav2Vec2 + ResNet（对照组）
# ============================================================

class FrozenSSLModel(nn.Module):
    """
    模型2：冻结的预训练 Wav2Vec2 + ResNet 后端分类器。
    
    代表"直接套用大模型，但不做任何针对性优化"的常规做法。
    
    关键设计：
        - Wav2Vec2 参数全部冻结（requires_grad=False）
        - 前向传播使用 torch.no_grad() 节省约 40-60% 显存
        - 只训练后端 ResNet 的参数
    
    使命：
        在干净数据上表现好（证明大模型本身强大）；
        在退化数据上暴跌（证明不优化的致命缺陷）。
    """
    
    def __init__(self, ssl_model_name: str = SSL_MODEL_NAME):
        super().__init__()
        
        print(f"[模型2] 加载预训练模型: {ssl_model_name} ...")
        
        # 加载 Wav2Vec2 预训练模型（特征提取器）
        self.ssl_model = Wav2Vec2Model.from_pretrained(ssl_model_name)
        
        # 关键：彻底冻结前端大模型的全部参数
        # 这使 Wav2Vec2 成为一个固定的"特征自动售货机"
        for param in self.ssl_model.parameters():
            param.requires_grad = False
        
        # 后端分类器（仅此部分参与训练）
        self.backend = ResNetBackend(input_dim=SSL_FEATURE_DIM)
        
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        print(f"[模型2] FrozenSSLModel | 总参数: {total:,} | 可训练: {trainable:,}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: 原始音频张量, shape=(batch, num_samples)
        
        返回:
            logits: shape=(batch,)
        """
        # torch.no_grad() 禁止梯度计算，大幅节省冻结模型的显存
        with torch.no_grad():
            # Wav2Vec2 输出 shape: (batch, time_frames, feature_dim=1024)
            ssl_output = self.ssl_model(x).last_hidden_state
        
        # ResNetBackend 接收 (batch, time_frames, feature_dim) 直接送入
        return self.backend(ssl_output)


# ============================================================
# 模型 3：可微调 Wav2Vec2 + ResNet（终极方案）
# ============================================================

class FineTunedSSLModel(nn.Module):
    """
    模型3：渐进式解冻微调 Wav2Vec2 + ResNet + 实时数据增强。
    
    代表"本项目提出的创新解决方案"。
    
    关键设计：
        - 初始状态与模型2相同（Wav2Vec2 全冻结）
        - 通过 progressive_unfreeze() 在训练过程中逐步解冻
        - 配合 OnTheFlyAugmentor 的数据增强提升鲁棒性
    
    与模型2的唯一区别：
        ① 数据增强开启
        ② 渐进式解冻开启
    因此两者的 EER 差异可精确归因于这两个创新点。
    """
    
    def __init__(self, ssl_model_name: str = SSL_MODEL_NAME):
        super().__init__()
        
        print(f"[模型3] 加载预训练模型: {ssl_model_name} ...")
        
        # 加载 Wav2Vec2 预训练模型
        self.ssl_model = Wav2Vec2Model.from_pretrained(ssl_model_name)
        
        # 初始状态：全部冻结（与模型2相同起点，保证公平对比）
        for param in self.ssl_model.parameters():
            param.requires_grad = False
        
        # 后端分类器
        self.backend = ResNetBackend(input_dim=SSL_FEATURE_DIM)
        
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        print(f"[模型3] FineTunedSSLModel | 总参数: {total:,} | 初始可训练: {trainable:,}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        注意：当 SSL 参数仍全部冻结时，使用 torch.no_grad() 节省激活值显存。
        一旦任何 SSL 参数被解冻（requires_grad=True），则允许梯度流通。
        
        参数:
            x: 原始音频张量, shape=(batch, num_samples)
        
        返回:
            logits: shape=(batch,)
        """
        # 检查 SSL 是否已有可训练参数（即是否已经解冻过）
        ssl_has_grad = any(p.requires_grad for p in self.ssl_model.parameters())
        
        if ssl_has_grad:
            # 解冻后：正常前向，允许梯度计算（Wav2Vec2 参与反向传播）
            ssl_output = self.ssl_model(x).last_hidden_state
        else:
            # 完全冻结阶段：用 no_grad 节省约 20GB 激活值显存
            with torch.no_grad():
                ssl_output = self.ssl_model(x).last_hidden_state
        
        # ResNetBackend 接收 (batch, time_frames, feature_dim)
        return self.backend(ssl_output)


# ============================================================
# 渐进式解冻策略函数
# ============================================================

def progressive_unfreeze(model: FineTunedSSLModel, 
                          epoch: int, 
                          optimizer: torch.optim.Optimizer) -> None:
    """
    渐进式解冻策略：根据当前训练 epoch 动态调整大模型的冻结状态。
    
    解冻时间表（针对 wav2vec2-base，共12层）：
        Epoch 0~4：全部冻结（后端 ResNet 预热学习基本判别能力）
        Epoch 5：  解冻最后 2 层（indexes: UNFREEZE_LAYERS_1）
        Epoch 8：  解冻最后 4 层（indexes: UNFREEZE_LAYERS_2）
    
    设计原理：
        先冻结让后端 "热身"，再用极低学习率（ssl_lr=1e-5）小心解冻大模型，
        防止灾难性遗忘（Catastrophic Forgetting）。
    
    参数:
        model: 模型3实例（FineTunedSSLModel）
        epoch: 当前训练轮次（从0开始）
        optimizer: 当前优化器（解冻时向其添加新参数组）
    """
    if epoch == UNFREEZE_EPOCH_1:
        print(f"[解冻] Epoch {epoch}: 解冻 Wav2Vec2 最后 {len(UNFREEZE_LAYERS_1)} 层 Transformer")
        
        newly_unfrozen = []
        for name, param in model.ssl_model.named_parameters():
            # 检查是否是目标层的参数
            if any(f"encoder.layers.{i}" in name for i in UNFREEZE_LAYERS_1):
                param.requires_grad = True
                newly_unfrozen.append(param)
        
        if newly_unfrozen:
            # 以极低学习率将新解冻的参数添加到优化器
            optimizer.add_param_group({
                "params": newly_unfrozen,
                "lr": SSL_LR,
                "name": "ssl_unfrozen_stage1"
            })
            print(f"[解冻] 已解冻 {len(newly_unfrozen)} 个参数张量，学习率：{SSL_LR}")
    
    elif epoch == UNFREEZE_EPOCH_2:
        print(f"[解冻] Epoch {epoch}: 解冻 Wav2Vec2 最后 {len(UNFREEZE_LAYERS_2)} 层 Transformer")
        
        newly_unfrozen = []
        for name, param in model.ssl_model.named_parameters():
            if any(f"encoder.layers.{i}" in name for i in UNFREEZE_LAYERS_2):
                if not param.requires_grad:  # 只添加之前未解冻的层
                    param.requires_grad = True
                    newly_unfrozen.append(param)
        
        if newly_unfrozen:
            optimizer.add_param_group({
                "params": newly_unfrozen,
                "lr": SSL_LR,
                "name": "ssl_unfrozen_stage2"
            })
            print(f"[解冻] 新增解冻 {len(newly_unfrozen)} 个参数张量，学习率：{SSL_LR}")
        
        # 打印当前可训练参数总量
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[解冻] 当前可训练参数量: {trainable:,}")
