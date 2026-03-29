"""
evaluate.py - 模型评估与结果可视化模块
包含：
  - evaluate_model: 在指定测试集上推理，收集分数和标签
  - plot_det_curves: 绘制 DET 曲线（多模型对比）
  - plot_eer_comparison: 绘制 EER 分组柱状图
  - print_results_table: 控制台打印结果表格
  - run_all_evaluations: 运行全部 3×2=6 组评估
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")  # 无头模式（服务器无显示器时使用）
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.metrics import det_curve

from utils import compute_eer
from config import EVAL_BATCH_SIZE, FIGURE_DPI, FIGURES_DIR


# ============================================================
# 核心评估函数
# ============================================================

def evaluate_model(model: nn.Module, 
                   test_loader: DataLoader, 
                   device: torch.device,
                   desc: str = "评估中") -> tuple:
    """
    在指定测试集上运行模型推理，返回 EER 和原始预测数据。
    
    参数:
        model: 待评估的 PyTorch 模型（已加载权重）
        test_loader: 测试集 DataLoader
        device: 推理设备（cuda/cpu）
        desc: 进度条描述文字
    
    返回:
        (eer, threshold, all_labels, all_scores)
        - eer: 等错误率（0~1）
        - threshold: EER 对应的最优判定阈值
        - all_labels: 真实标签 NumPy 数组（0=真实，1=伪造）
        - all_scores: 模型输出的伪造概率分数（0~1）
    """
    model.eval()
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for waveforms, labels in tqdm(test_loader, desc=desc, leave=False):
            waveforms = waveforms.to(device)
            
            # 前向传播获取 logits
            logits = model(waveforms)
            
            # Sigmoid 激活：将 logit 转换为伪造概率分数（0~1）
            scores = torch.sigmoid(logits).cpu().numpy()
            
            all_scores.extend(scores.tolist())
            all_labels.extend(labels.numpy().tolist())
    
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    eer, threshold = compute_eer(all_labels, all_scores)
    
    return eer, threshold, all_labels, all_scores


def run_all_evaluations(models_dict: dict,
                         test_loaders_dict: dict,
                         device: torch.device) -> dict:
    """
    系统化地运行全部 3×2=6 组评估实验。
    
    参数:
        models_dict: {"模型名称": model_object, ...}
        test_loaders_dict: {"测试集名称": test_loader, ...}
        device: 推理设备
    
    返回:
        results: {(model_name, test_name): {"eer", "threshold", "labels", "scores"}}
    """
    results = {}
    
    print("\n" + "="*60)
    print("  开始运行全部 6 组消融实验评估")
    print("="*60)
    
    for model_name, model in models_dict.items():
        for test_name, test_loader in test_loaders_dict.items():
            print(f"\n→ 评估: [{model_name}] on [{test_name}]")
            
            eer, threshold, labels, scores = evaluate_model(
                model, test_loader, device,
                desc=f"{model_name[:8]} @ {test_name}"
            )
            
            results[(model_name, test_name)] = {
                "eer": eer,
                "threshold": threshold,
                "labels": labels,
                "scores": scores
            }
            
            print(f"  ✓ EER = {eer*100:.2f}% | 最优阈值 = {threshold:.4f}")
    
    return results


# ============================================================
# 控制台结果打印
# ============================================================

def print_results_table(results: dict) -> None:
    """
    以表格形式在控制台打印全部实验结果。
    
    参数:
        results: run_all_evaluations() 的返回值
    """
    model_names = ["Model1_Baseline", "Model2_FrozenSSL", "Model3_FineTuned"]
    test_names  = ["Clean", "Degraded"]
    
    display_names = {
        "Model1_Baseline":  "模型1: MFCC + CNN",
        "Model2_FrozenSSL": "模型2: 冻结SSL + ResNet",
        "Model3_FineTuned": "模型3: 微调SSL + 增强(本方案)"
    }
    
    print("\n" + "="*68)
    print("  消融实验结果汇总（EER，越低越好）")
    print("="*68)
    print(f"{'模型':<32} {'Test-A (Clean)':<16} {'Test-B (Degraded)':<16}")
    print("-"*68)
    
    for model_name in model_names:
        eer_clean   = results.get((model_name, "Clean"),    {}).get("eer", float("nan"))
        eer_degraded = results.get((model_name, "Degraded"), {}).get("eer", float("nan"))
        name = display_names.get(model_name, model_name)
        print(f"{name:<32} {eer_clean*100:.2f}%{'':<12} {eer_degraded*100:.2f}%")
    
    print("="*68)
    
    # 三段式论证
    m1_clean = results.get(("Model1_Baseline", "Clean"), {}).get("eer", 0)
    m2_clean = results.get(("Model2_FrozenSSL", "Clean"), {}).get("eer", 0)
    m2_deg   = results.get(("Model2_FrozenSSL", "Degraded"), {}).get("eer", 0)
    m3_deg   = results.get(("Model3_FineTuned", "Degraded"), {}).get("eer", 0)
    
    print("\n论证 1：大模型优于传统特征")
    if m1_clean > 0:
        improvement = (m1_clean - m2_clean) / m1_clean * 100
        print(f"  模型1 EER {m1_clean*100:.2f}% → 模型2 EER {m2_clean*100:.2f}%，"
              f"降低 {improvement:.1f}%")
    
    print("\n论证 2：常规大模型在退化环境下脆弱")
    if m2_clean > 0:
        degradation = (m2_deg - m2_clean) / m2_clean * 100
        print(f"  模型2 干净:{m2_clean*100:.2f}% → 退化:{m2_deg*100:.2f}%，"
              f"恶化 {degradation:.1f}%")
    
    print("\n论证 3：本方案有效提升跨信道鲁棒性")
    if m2_deg > 0:
        improvement = (m2_deg - m3_deg) / m2_deg * 100
        print(f"  模型2退化:{m2_deg*100:.2f}% → 模型3退化:{m3_deg*100:.2f}%，"
              f"改善 {improvement:.1f}%")
    print()


# ============================================================
# DET 曲线绘制
# ============================================================

def plot_det_curves(results: dict, 
                    test_name: str, 
                    save_path: str) -> None:
    """
    绘制指定测试集上三个模型的 DET 曲线对比图。
    
    DET（Detection Error Tradeoff）曲线比 ROC 曲线更适合语音防伪场景：
    X 轴为 FAR（误放率），Y 轴为 FRR（误拒率），
    曲线越靠近左下角代表性能越好。
    
    参数:
        results: run_all_evaluations() 的返回值
        test_name: "Clean" 或 "Degraded"
        save_path: 图片保存路径
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, ax = plt.subplots(1, 1, figsize=(9, 7))
    
    # 颜色和标签配置
    model_configs = [
        ("Model1_Baseline",  "#e74c3c", "Model 1: MFCC + CNN (Baseline)"),
        ("Model2_FrozenSSL", "#3498db", "Model 2: Frozen Wav2Vec2 + ResNet"),
        ("Model3_FineTuned", "#2ecc71", "Model 3: Fine-tuned + Augmentation (Ours)"),
    ]
    
    for model_name, color, display_name in model_configs:
        key = (model_name, test_name)
        if key not in results:
            continue
        
        data   = results[key]
        labels = data["labels"]
        scores = data["scores"]
        eer    = data["eer"]
        
        # 计算 DET 曲线
        fpr_det, fnr_det, _ = det_curve(labels, scores)
        
        ax.plot(
            fpr_det * 100, fnr_det * 100,
            color=color,
            linewidth=2.5,
            label=f"{display_name}\n  EER = {eer*100:.2f}%"
        )
        
        # 在 EER 点处标注
        eer_pct = eer * 100
        ax.scatter([eer_pct], [eer_pct], color=color, 
                   s=80, zorder=5, marker="o")
    
    # 获取最大 EER 以动态调整坐标轴（最低显示到 50%）
    test_eers = [results[(m, test_name)]["eer"] for m, _, _ in model_configs if (m, test_name) in results]
    max_eer_val = max(test_eers) if test_eers else 0.5
    
    # 区分 Clean 和 Degraded 测试集：Clean 保持原样，Degraded 强制设为 100
    if test_name == "Clean":
        limit = max(50.0, min(100.0, max_eer_val * 110.0))  # 留出 10% 余量，最高 100%
    else:
        limit = 100.0  # Degraded 测试集强制使用最大值 100
    
    # 等错误率参考线（y = x）
    ax.plot([0, limit], [0, limit], "k--", alpha=0.35, linewidth=1.5, 
            label="EER Reference Line (y = x)")
    
    # 轴标签与标题
    ax.set_xlabel("False Accept Rate / FAR (%)", fontsize=13)
    ax.set_ylabel("False Reject Rate / FRR (%)", fontsize=13)
    
    title_prefix = "Clean Test Set" if test_name == "Clean" else "Degraded Test Set (Test-B)"
    ax.set_title(f"DET Curve Comparison — {title_prefix}", fontsize=14, fontweight="bold")
    
    ax.set_xlim([0, limit])
    ax.set_ylim([0, limit])

    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="upper right", fontsize=10, framealpha=0.85)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    
    print(f"[可视化] DET 曲线已保存 → {save_path}")


# ============================================================
# EER 分组柱状图
# ============================================================

def plot_eer_comparison(results: dict, save_path: str) -> None:
    """
    绘制 6 组 EER 数据的分组柱状图。
    
    每个模型显示两个柱：考场A（干净）和考场B（退化），
    形成直观的"视觉冲击"——模型2在退化条件下的柱子高耸，
    模型3则在两种条件下都保持低位。
    
    参数:
        results: run_all_evaluations() 的返回值
        save_path: 图片保存路径
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    model_names = ["Model1_Baseline", "Model2_FrozenSSL", "Model3_FineTuned"]
    x_labels    = ["Model 1\n(MFCC+CNN)", "Model 2\n(Frozen SSL)", "Model 3\n(Ours)"]
    
    eer_clean    = [results.get((m, "Clean"),    {}).get("eer", 0) * 100 for m in model_names]
    eer_degraded = [results.get((m, "Degraded"), {}).get("eer", 0) * 100 for m in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(11, 7))
    
    bars1 = ax.bar(x - width / 2, eer_clean,    width, 
                   label="Test Set A (Clean)", color="#3498db", alpha=0.85)
    bars2 = ax.bar(x + width / 2, eer_degraded, width, 
                   label="Test Set B (Degraded)", color="#e74c3c", alpha=0.85)
    
    # 在每个柱子顶部标注数值
    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.annotate(
            f"{h:.2f}%",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center", va="bottom",
            fontweight="bold", fontsize=11
        )
    
    ax.set_ylabel("Equal Error Rate / EER (%)  ↓ lower is better", fontsize=12)
    ax.set_title("Ablation Study: EER Comparison Across Models and Test Conditions",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    
    # 设置 Y 轴上限比最大值多 20%，留出标注空间
    max_eer = max(eer_clean + eer_degraded)
    ax.set_ylim([0, max_eer * 1.25 + 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    
    print(f"[可视化] EER 对比柱状图已保存 → {save_path}")


# ============================================================
# 训练曲线绘制
# ============================================================

def plot_training_curves(history: dict, model_name: str, save_path: str) -> None:
    """
    绘制训练过程中的 Loss 和验证 EER 曲线。
    
    参数:
        history: {"train_loss": [...], "val_eer": [...]}
        model_name: 模型名称（用于标题）
        save_path: 图片保存路径
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    epochs = range(1, len(history["train_loss"]) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 训练损失曲线
    axes[0].plot(epochs, history["train_loss"], "b-o", linewidth=2, markersize=4)
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Training Loss (BCE)", fontsize=12)
    axes[0].set_title(f"{model_name} — Training Loss", fontsize=13)
    axes[0].grid(True, alpha=0.3)
    
    # 验证 EER 曲线
    val_eer_pct = [v * 100 for v in history["val_eer"]]
    axes[1].plot(epochs, val_eer_pct, "r-o", linewidth=2, markersize=4)
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Validation EER (%)", fontsize=12)
    axes[1].set_title(f"{model_name} — Validation EER", fontsize=13)
    axes[1].grid(True, alpha=0.3)
    
    # 标注最低 EER 点
    min_eer_epoch = np.argmin(val_eer_pct) + 1
    min_eer_val   = min(val_eer_pct)
    axes[1].scatter([min_eer_epoch], [min_eer_val], color="red", 
                    s=100, zorder=5, label=f"Best: {min_eer_val:.2f}%")
    axes[1].legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    
    print(f"[可视化] 训练曲线已保存 → {save_path}")
