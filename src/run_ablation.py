"""
run_ablation.py - 消融实验评估脚本
加载三个已训练的模型，在两个测试考场上运行评估，生成全部结果图表

用法:
    conda activate llmdevelop
    cd src
    python run_ablation.py
"""

import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import torch
import numpy as np
from torch.utils.data import DataLoader

from config import (
    EVAL_AUDIO_DIR, EVAL_PROTOCOL,
    SEED, NUM_WORKERS, EVAL_BATCH_SIZE,
    CHECKPOINT_DIR, FIGURES_DIR, RESULTS_DIR
)
from utils import set_global_seed, load_checkpoint, save_results, print_gpu_info
from dataset import ASVspoofDataset, HeavyDegradation
from models import BaselineModel, FrozenSSLModel, FineTunedSSLModel
from evaluate import (
    run_all_evaluations,
    plot_det_curves,
    plot_eer_comparison,
    print_results_table
)


def run_ablation_evaluation(plot_only=False):
    """
    消融实验全量评估入口。
    
    参数:
        plot_only: 如果为 True，则跳过模型推理，直接从之前保存的 .npz 文件加载数据进行图表重绘。
        
    流程：
    1. 加载三个已训练的模型权重
    2. 构建考场A（干净）和考场B（退化）的测试数据集
    3. 运行 3×2=6 组评估，收集 EER 数据
    4. 绘制 DET 曲线和 EER 对比柱状图
    5. 保存原始结果数据（JSON）供报告引用
    """
    
    set_global_seed(SEED)
    print_gpu_info()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[评估] 使用设备: {device}")
    
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    results = {}
    
    if plot_only:
        print("\n[模式] 仅重绘模式 (--plot-only) 已开启，跳过模型评估。")
        full_results_path = os.path.join(RESULTS_DIR, "ablation_results_full.npz")
        if not os.path.exists(full_results_path):
            print(f"[错误] 找不到已保存的评估结果文件：{full_results_path}")
            print("请先执行完整评估（不要加 --plot-only）。")
            return
            
        print("\n[加载已保存结果] 正在从 npz 文件加载评估结果用于重新绘图...")
        loaded_data = np.load(full_results_path)
        
        # 提取模型和测试集组合名称
        prefixes = set()
        for k in loaded_data.files:
            if "__" in k:
                parts = k.split("__")
                if len(parts) >= 2:
                    prefixes.add((parts[0], parts[1]))
                    
        for model_name, test_name in prefixes:
            prefix = f"{model_name}__{test_name}"
            try:
                results[(model_name, test_name)] = {
                    "labels": loaded_data[f"{prefix}__labels"],
                    "scores": loaded_data[f"{prefix}__scores"],
                    "eer": float(loaded_data[f"{prefix}__eer"][0]),
                    "threshold": float(loaded_data.get(f"{prefix}__threshold", [0.0])[0])  # 如果不存在提供默认值
                }
            except KeyError as e:
                print(f"警告：无法从 npz 中加载键 {e}，可能数据不完整。")
        
        if not results:
            print("[错误] 未能成功加载任何结果数据！")
            return
            
        print("  ✓ 结果加载完成")
        
    else:
        # ============================================================
        # 1. 加载三个模型（仅在正常模式下执行）
        # ============================================================
        print("\n[模型加载] 加载三个消融实验模型...")
    
        model1 = BaselineModel()
        model2 = FrozenSSLModel()
        model3 = FineTunedSSLModel()
        
        # 检查检查点文件是否存在
        ckpt_paths = {
            "Model1_Baseline":  os.path.join(CHECKPOINT_DIR, "model1_baseline.pth"),
            "Model2_FrozenSSL": os.path.join(CHECKPOINT_DIR, "model2_frozen_ssl.pth"),
            "Model3_FineTuned": os.path.join(CHECKPOINT_DIR, "model3_finetuned_aug.pth"),
        }
        models_map = {
            "Model1_Baseline":  model1,
            "Model2_FrozenSSL": model2,
            "Model3_FineTuned": model3,
        }
        
        loaded_models = {}
        for name, model in models_map.items():
            ckpt_path = ckpt_paths[name]
            if os.path.exists(ckpt_path):
                metadata = load_checkpoint(model, ckpt_path)
                model.to(device)
                model.eval()
                loaded_models[name] = model
                print(f"  ✓ {name} | 最佳验证 EER: {metadata.get('val_eer', 'N/A') if isinstance(metadata.get('val_eer'), float) else 'N/A'}")
            else:
                print(f"  ✗ {name} | 检查点未找到: {ckpt_path}")
                print(f"     请先运行对应的训练脚本！")
        
        if not loaded_models:
            print("\n[错误] 没有找到任何已训练模型的检查点！")
            print("请按顺序运行：train_model1.py → train_model2.py → train_model3.py")
            return
        
        # ============================================================
        # 2. 构建测试数据集
        # ============================================================
        print("\n[数据集] 构建两个测试考场...")
        
        # 考场A：原版干净测试集
        test_dataset_clean = ASVspoofDataset(
            audio_dir=EVAL_AUDIO_DIR,
            protocol_file=EVAL_PROTOCOL,
            augment_fn=None  # 无任何处理
        )
        test_loader_clean = DataLoader(
            test_dataset_clean,
            batch_size=EVAL_BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS
        )
        
        # 考场B：重度退化测试集（噪声 + 频段滤波 + 量化压缩）
        degradation = HeavyDegradation()
        test_dataset_degraded = ASVspoofDataset(
            audio_dir=EVAL_AUDIO_DIR,
            protocol_file=EVAL_PROTOCOL,
            augment_fn=degradation  # 应用重度退化
        )
        test_loader_degraded = DataLoader(
            test_dataset_degraded,
            batch_size=EVAL_BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS
        )
        
        print(f"  考场A（干净）: {len(test_dataset_clean)} 条样本")
        print(f"  考场B（退化）: {len(test_dataset_degraded)} 条样本")
        
        # ============================================================
        # 3. 运行全部评估
        # ============================================================
        test_loaders = {
            "Clean":    test_loader_clean,
            "Degraded": test_loader_degraded
        }
        
        results = run_all_evaluations(
            models_dict=loaded_models,
            test_loaders_dict=test_loaders,
            device=device
        )
    
    # ============================================================
    # 4. 打印结果表格
    # ============================================================
    print_results_table(results)
    
    # ============================================================
    # 5. 绘制可视化图表
    # ============================================================
    print("\n[可视化] 生成实验图表...")
    
    # DET 曲线：考场A
    plot_det_curves(
        results=results,
        test_name="Clean",
        save_path=os.path.join(FIGURES_DIR, "det_curve_clean.png")
    )
    
    # DET 曲线：考场B（退化），这是最重要的对比图
    plot_det_curves(
        results=results,
        test_name="Degraded",
        save_path=os.path.join(FIGURES_DIR, "det_curve_degraded.png")
    )
    
    # EER 分组柱状图
    plot_eer_comparison(
        results=results,
        save_path=os.path.join(FIGURES_DIR, "eer_comparison.png")
    )
    
    # ============================================================
    # 6. 保存原始结果数据
    # ============================================================
    # 构建精简结果字典（不含 NumPy 数组，便于 JSON 序列化）
    summary = {}
    for (model_name, test_name), data in results.items():
        key = f"{model_name}__{test_name}"
        summary[key] = {
            "eer": float(data["eer"]),
            "eer_percent": float(data["eer"] * 100),
            "threshold": float(data["threshold"]),
            "n_samples": int(len(data["labels"])),
            "n_bonafide": int((data["labels"] == 0).sum()),
            "n_spoof": int((data["labels"] == 1).sum()),
        }
    
    results_path = os.path.join(RESULTS_DIR, "ablation_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[结果] 实验数据已保存至 → {results_path}")
    
    # 同时保存包含完整 scores/labels 的版本（用于后续精细分析）
    if not plot_only: # --plot-only模式下不需要重新保存原始数据
        full_results_path = os.path.join(RESULTS_DIR, "ablation_results_full.npz")
        save_data = {}
        for (model_name, test_name), data in results.items():
            prefix = f"{model_name}__{test_name}"
            save_data[f"{prefix}__labels"] = data["labels"]
            save_data[f"{prefix}__scores"] = data["scores"]
            save_data[f"{prefix}__eer"]    = np.array([data["eer"]])
            save_data[f"{prefix}__threshold"] = np.array([data.get("threshold", 0.0)])
        np.savez(full_results_path, **save_data)
        print(f"[结果] 完整数据（含scores/labels）已保存至 → {full_results_path}")
    
    print("\n[完成] 消融实验评估全部完成！")
    print(f"  图表目录: {FIGURES_DIR}")
    print(f"  结果目录: {RESULTS_DIR}")
    
    return results, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行消融实验评估与绘图")
    parser.add_argument("--plot-only", action="store_true", help="跳过评估，只读取上次保存的 .npz 文件进行画图")
    args = parser.parse_args()
    
    run_ablation_evaluation(plot_only=args.plot_only)
