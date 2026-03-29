"""
dataset.py - 数据集与数据增强模块
包含：
  - ASVspoofDataset: ASVspoof 2019 LA 数据集加载器
  - OnTheFlyAugmentor: GPU 实时动态数据增强（模型3使用）
  - HeavyDegradation: 重度退化处理（考场B使用）

注意：使用 soundfile 加载 .flac 文件（兼容 torchaudio 2.9.x，
该版本的 torchaudio.load 需要额外安装 torchcodec）
"""

import os
import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as aF
from torch.utils.data import Dataset
from typing import Optional, Callable, List, Tuple

from config import (
    SAMPLE_RATE, MAX_LENGTH, LABEL_MAP,
    AUG_PROB, SNR_MIN, SNR_MAX, QUANT_BITS_MIN, QUANT_BITS_MAX,
    DEGRADE_SNR, DEGRADE_HP_FREQ, DEGRADE_LP_FREQ, DEGRADE_QUANT_BITS
)


# ============================================================
# 数据集类
# ============================================================

class ASVspoofDataset(Dataset):
    """
    ASVspoof 2019 LA 数据集加载器。
    
    核心职责：
    1. 解析协议文件，建立（音频文件名 → 标签）的映射
    2. 加载 .flac 文件，统一采样率至 SAMPLE_RATE
    3. 截断/零填充至 MAX_LENGTH，保证批次维度一致
    4. 可选地调用数据增强函数
    
    标签约定：
        0 = bonafide（真实人声）
        1 = spoof（伪造/TTS/VC）
    """
    
    def __init__(self, 
                 audio_dir: str, 
                 protocol_file: str, 
                 augment_fn: Optional[Callable] = None):
        """
        参数:
            audio_dir: 音频 .flac 文件所在目录
            protocol_file: 标签协议文件路径
            augment_fn: 数据增强函数（None = 不增强，用于模型1/2和测试）
        """
        self.audio_dir = audio_dir
        self.augment_fn = augment_fn
        self.samples: List[Tuple[str, int]] = []
        
        # 解析协议文件
        # 格式: speaker_id  filename  -  attack_type  label
        # 示例: LA_0079  LA_T_1138215  -  A10  spoof
        with open(protocol_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue  # 跳过格式异常行
                filename = parts[1]                   # 音频文件名（无后缀）
                label_str = parts[-1]                 # "bonafide" 或 "spoof"
                label = LABEL_MAP.get(label_str, 1)  # 未知标签默认视为伪造
                self.samples.append((filename, label))
        
        print(f"[数据集] 加载完成：{len(self.samples)} 条样本 | 目录: {os.path.basename(audio_dir)}")
        
        # 统计类别分布
        n_bonafide = sum(1 for _, l in self.samples if l == 0)
        n_spoof    = sum(1 for _, l in self.samples if l == 1)
        print(f"[数据集] 真实: {n_bonafide} | 伪造: {n_spoof} | 增强: {'开启' if augment_fn else '关闭'}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        filename, label = self.samples[idx]
        
        # 1. 构建完整文件路径
        filepath = os.path.join(self.audio_dir, f"{filename}.flac")
        
        # 2. 使用 soundfile 加载原始波形（兼容 torchaudio 2.9.x）
        # soundfile 返回 np.ndarray，shape=(num_samples,) 或 (num_samples, channels)
        data, sr = sf.read(filepath, dtype="float32")
        
        # 3. 处理多声道：取第一个声道
        if data.ndim > 1:
            data = data[:, 0]  # → shape: (num_samples,)
        
        # 4. 转换为 PyTorch Tensor
        waveform = torch.from_numpy(data)  # shape: (num_samples,)
        
        # 5. 重采样（如果原始采样率不是目标采样率）
        # ASVspoof 2019 已是 16kHz，一般无需重采样，但保留此逻辑保证通用性
        if sr != SAMPLE_RATE:
            # 使用简单的线性插值重采样（soundfile 场景下的简易替代）
            resample_ratio = SAMPLE_RATE / sr
            new_length = int(len(waveform) * resample_ratio)
            waveform = torch.nn.functional.interpolate(
                waveform.view(1, 1, -1),
                size=new_length,
                mode="linear",
                align_corners=False
            ).view(-1)
        
        # 6. 截断 / 零填充至固定长度 MAX_LENGTH（4秒）
        n = waveform.shape[0]
        if n > MAX_LENGTH:
            # 随机截取（训练时增加多样性；评估时取前段）
            if self.augment_fn is not None:
                start = torch.randint(0, n - MAX_LENGTH + 1, (1,)).item()
                waveform = waveform[start: start + MAX_LENGTH]
            else:
                waveform = waveform[:MAX_LENGTH]
        elif n < MAX_LENGTH:
            # 零填充至 MAX_LENGTH
            pad_len = MAX_LENGTH - n
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))
        
        # 7. 数据增强（仅训练时启用，传入 augment_fn）
        if self.augment_fn is not None:
            waveform = self.augment_fn(waveform)
        
        return waveform, torch.tensor(label, dtype=torch.float32)



# ============================================================
# 数据增强类（模型3：训练阶段）
# ============================================================

class OnTheFlyAugmentor:
    """
    GPU 实时动态数据增强器（模型3专用）。
    
    在每个 batch 送入模型前随机施加声学干扰，相当于无限扩充训练数据。
    模拟真实世界的恶劣通信环境（背景噪声、压缩失真、频带限制）。
    
    核心目的：
        训练阶段见过各种干扰 → 推理时遇到真实噪声不会失效 → 提升跨信道鲁棒性
    """
    
    def __init__(self, 
                 p: float = AUG_PROB, 
                 snr_range: Tuple[float, float] = (SNR_MIN, SNR_MAX)):
        """
        参数:
            p: 每种增强策略的触发概率（0~1）
            snr_range: 信噪比范围（dB），值越小噪声越强
        """
        self.p = p
        self.snr_range = snr_range
    
    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        对单个音频张量施加随机增强。
        
        参数:
            waveform: 音频张量，shape=(num_samples,)
        
        返回:
            增强后的音频张量（相同 shape）
        """
        # --------------- 增强 1：高斯白噪声注入 ---------------
        # 模拟环境底噪、电气噪声
        if torch.rand(1).item() < self.p:
            snr_db = torch.FloatTensor(1).uniform_(*self.snr_range).item()
            signal_power = waveform.norm(p=2)
            noise = torch.randn_like(waveform)
            noise_power = noise.norm(p=2)
            # 根据目标 SNR 算出噪声缩放系数
            if noise_power > 1e-8:  # 避免除以0
                scale = signal_power / (noise_power * (10 ** (snr_db / 20.0)))
                waveform = waveform + scale * noise
        
        # --------------- 增强 2：低比特量化 ---------------
        # 模拟微信语音、VoIP 等场景的极致压缩
        if torch.rand(1).item() < self.p:
            bit_depth = torch.randint(QUANT_BITS_MIN, QUANT_BITS_MAX + 1, (1,)).item()
            scale_factor = float(2 ** bit_depth)
            waveform = torch.round(waveform * scale_factor) / scale_factor
        
        # --------------- 增强 3：电话频段滤波 ---------------
        # 模拟电话通话（300Hz~3400Hz），高频和低频被丢弃
        # 触发概率降低（设为 p*0.3）避免过度限制频带
        if torch.rand(1).item() < self.p * 0.3:
            try:
                waveform = aF.highpass_biquad(waveform, SAMPLE_RATE, 300.0)
                waveform = aF.lowpass_biquad(waveform,  SAMPLE_RATE, 3400.0)
            except Exception:
                pass  # 滤波器在极短音频上可能失败，跳过
        
        # 归一化防止裁剪（振幅超出 [-1, 1] 后截断）
        max_val = waveform.abs().max()
        if max_val > 1.0:
            waveform = waveform / max_val
        
        return waveform


# ============================================================
# 退化处理类（考场B：测试阶段）
# ============================================================

class HeavyDegradation:
    """
    重度退化处理器（考场B专用）。
    
    对测试集音频施加三重叠加退化：
    1. 高强度背景噪声（SNR = 5dB，极嘈杂环境）
    2. 电话频段滤波（300Hz~3400Hz，丢弃大量频率信息）  
    3. 低比特量化（4-bit，模拟微信极致压缩）
    
    核心目的：
        在最极端条件下考验模型，拉大模型2（无增强）与模型3（有增强）的差距
    """
    
    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        对音频施加确定性（非随机）的重度退化处理。
        
        注意：此处不随机化，确保考场B的测试条件对所有模型完全相同。
        """
        # 步骤1：高强度背景噪声（SNR = DEGRADE_SNR dB）
        noise = torch.randn_like(waveform)
        signal_power = waveform.norm(p=2)
        noise_power  = noise.norm(p=2)
        if noise_power > 1e-8:
            scale = signal_power / (noise_power * (10 ** (DEGRADE_SNR / 20.0)))
            waveform = waveform + scale * noise
        
        # 步骤2：电话频段滤波（300Hz~3400Hz）
        try:
            waveform = aF.highpass_biquad(waveform, SAMPLE_RATE, float(DEGRADE_HP_FREQ))
            waveform = aF.lowpass_biquad(waveform,  SAMPLE_RATE, float(DEGRADE_LP_FREQ))
        except Exception:
            pass
        
        # 步骤3：4-bit 量化（模拟极度压缩）
        scale_factor = float(2 ** DEGRADE_QUANT_BITS)
        waveform = torch.round(waveform * scale_factor) / scale_factor
        
        # 归一化
        max_val = waveform.abs().max()
        if max_val > 1.0:
            waveform = waveform / max_val
        
        return waveform
