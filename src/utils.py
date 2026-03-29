"""
utils.py - 工具函数集合
包含：随机种子设置、EER计算、日志记录、检查点保存/加载
"""

import os
import json
import random
import logging
import numpy as np
import torch
from sklearn.metrics import roc_curve


def set_global_seed(seed: int = 42) -> None:
    """
    固定所有随机源，保证实验的完全可复现性。
    
    参数:
        seed: 随机种子值，默认42
    
    注意:
        设置 cudnn.deterministic=True 会牺牲约 10-15% 的速度，
        但对控制变量实验的严谨性至关重要。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多 GPU 场景下同时固定
    # 关闭 cuDNN 自动选择最优卷积算法，换取确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[工具] 全局随机种子已固定为 {seed}")


def setup_logger(log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """
    配置并返回一个标准化的日志记录器。
    
    参数:
        log_file: 日志文件路径（None 则只输出到控制台）
        level: 日志等级
    """
    logger = logging.getLogger("deepfake_det")
    logger.setLevel(level)
    
    formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # 控制台处理器
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    
    # 文件处理器（可选）
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    return logger


def compute_eer(labels: np.ndarray, scores: np.ndarray) -> tuple:
    """
    计算等错误率（Equal Error Rate, EER）。
    
    EER 是语音防伪领域的国际共识标准评估指标：
    - 对应 FRR（误拒率）与 FAR（误放率）相等时的错误率
    - EER 越低，系统越稳健
    
    参数:
        labels: 真实标签数组（0=真实, 1=伪造）
        scores: 模型输出的伪造置信度分数（经过 sigmoid 后的概率值）
    
    返回:
        (eer, threshold): EER 值 和 对应的最优判定阈值
    """
    # sklearn 的 roc_curve 使用 pos_label=1（即伪造=1 为正类）
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    
    # FNR（漏检率）= 1 - TPR（召回率）
    fnr = 1.0 - tpr
    
    # 找到 FAR（fpr）和 FRR（fnr）最接近的交叉点
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    
    # EER 取 FAR 和 FRR 的平均（减少舍入误差）
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
    threshold = float(thresholds[eer_idx])
    
    return eer, threshold


def save_checkpoint(model: torch.nn.Module, 
                    path: str, 
                    metadata: dict = None) -> None:
    """
    保存模型权重检查点。
    
    参数:
        model: PyTorch 模型
        path: 保存路径（.pth 文件）
        metadata: 附加元信息（如当前 epoch、EER 等）
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    save_dict = {
        "model_state_dict": model.state_dict(),
        "metadata": metadata or {}
    }
    torch.save(save_dict, path)
    print(f"[检查点] 模型已保存至 → {path}")


def load_checkpoint(model: torch.nn.Module, path: str) -> dict:
    """
    从检查点加载模型权重。
    
    参数:
        model: 目标 PyTorch 模型（权重将被载入）
        path: 检查点文件路径
    
    返回:
        metadata: 检查点中附带的元信息字典
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"[检查点] 权重已从 {path} 载入")
    return checkpoint.get("metadata", {})


def save_results(results: dict, path: str) -> None:
    """
    将实验结果（EER 数据）保存为 JSON 文件。
    
    参数:
        results: 结果字典（不包含 NumPy 数组，仅标量数据）
        path: 保存路径
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # 将 numpy 数值转换为 Python 原生类型，以便 JSON 序列化
    serializable = {}
    for key, val in results.items():
        if isinstance(key, tuple):
            str_key = f"{key[0]}__{key[1]}"
        else:
            str_key = str(key)
        
        if isinstance(val, dict):
            entry = {}
            for k, v in val.items():
                if isinstance(v, (np.ndarray, list)) and k in ("labels", "scores"):
                    entry[k] = v.tolist() if isinstance(v, np.ndarray) else v
                elif isinstance(v, (np.floating, np.integer)):
                    entry[k] = float(v)
                else:
                    entry[k] = v
            serializable[str_key] = entry
        else:
            serializable[str_key] = val
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    
    print(f"[结果] 实验数据已保存至 → {path}")


class EarlyStopping:
    """
    早停策略：当验证集 EER 连续 patience 个 epoch 不降低时停止训练。
    EER 越低越好，因此监控最小值。
    """
    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        """
        参数:
            patience: 允许的最大连续不改善 epoch 数
            min_delta: 被认为是"改善"的最小降低量
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_eer = float("inf")
        self.should_stop = False
    
    def step(self, val_eer: float) -> bool:
        """
        更新早停状态。
        
        参数:
            val_eer: 当前 epoch 的验证集 EER
        
        返回:
            True（应该停止）或 False（继续训练）
        """
        if val_eer < self.best_eer - self.min_delta:
            self.best_eer = val_eer
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def print_gpu_info() -> None:
    """打印当前 GPU 信息，用于记录实验硬件环境。"""
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[硬件] GPU: {gpu} | 总显存: {total_mem:.1f} GB")
        print(f"[硬件] CUDA 版本: {torch.version.cuda}")
        print(f"[硬件] PyTorch 版本: {torch.__version__}")
    else:
        print("[硬件] 警告：未检测到可用 GPU，将使用 CPU 训练（速度极慢）")
