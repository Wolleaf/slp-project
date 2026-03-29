"""
run_all.py - 主实验入口脚本（本地流程跑通用）
依次训练模型1、模型2、模型3，最后运行消融实验评估

这个脚本用于在本地 5060 GPU 上跑通完整流程。
在云服务器 5090 上请使用 run_cloud.sh 批量提交。

用法:
    conda activate llmdevelop
    cd src
    python run_all.py [--model 1|2|3|all] [--eval_only]

示例:
    python run_all.py                  # 全量运行（训练1/2/3 + 评估）
    python run_all.py --model 1        # 只训练模型1
    python run_all.py --eval_only      # 只做评估（需要已有检查点）
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import time

from config import CHECKPOINT_DIR, FIGURES_DIR, RESULTS_DIR
from utils import print_gpu_info, set_global_seed
from config import SEED


def parse_args():
    parser = argparse.ArgumentParser(description="语音真伪检测消融实验 - 完整流程")
    parser.add_argument("--model", type=str, default="all",
                        choices=["1", "2", "3", "all"],
                        help="选择训练哪个模型（默认：all）")
    parser.add_argument("--eval_only", action="store_true",
                        help="跳过训练，只运行消融实验评估")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 创建必要目录
    for d in [CHECKPOINT_DIR, FIGURES_DIR, RESULTS_DIR]:
        os.makedirs(d, exist_ok=True)
    
    print("\n" + "="*65)
    print("  语音真伪检测（Audio Deepfake Detection）消融实验")
    print("  ASVspoof 2019 LA | 三模型消融 | 两场景测试")
    print("="*65)
    print_gpu_info()
    set_global_seed(SEED)
    
    total_start = time.time()
    
    if not args.eval_only:
        
        # ================== 训练模型1 ==================
        if args.model in ("1", "all"):
            print("\n" + "─"*65)
            print("  [阶段 1/4] 训练模型1：MFCC + CNN Baseline")
            print("─"*65)
            t0 = time.time()
            
            from train_model1 import train_model1
            history1, eer1 = train_model1()
            
            elapsed = (time.time() - t0) / 60
            print(f"\n  ✓ 模型1训练完成 | 耗时 {elapsed:.1f} 分钟 | 最佳验证EER: {eer1*100:.2f}%")
        
        # ================== 训练模型2 ==================
        if args.model in ("2", "all"):
            print("\n" + "─"*65)
            print("  [阶段 2/4] 训练模型2：冻结 Wav2Vec2 + ResNet")
            print("─"*65)
            t0 = time.time()
            
            from train_model2 import train_model2
            history2, eer2 = train_model2()
            
            elapsed = (time.time() - t0) / 60
            print(f"\n  ✓ 模型2训练完成 | 耗时 {elapsed:.1f} 分钟 | 最佳验证EER: {eer2*100:.2f}%")
        
        # ================== 训练模型3 ==================
        if args.model in ("3", "all"):
            print("\n" + "─"*65)
            print("  [阶段 3/4] 训练模型3：微调 Wav2Vec2 + ResNet + 数据增强")
            print("─"*65)
            t0 = time.time()
            
            from train_model3 import train_model3
            history3, eer3 = train_model3()
            
            elapsed = (time.time() - t0) / 60
            print(f"\n  ✓ 模型3训练完成 | 耗时 {elapsed:.1f} 分钟 | 最佳验证EER: {eer3*100:.2f}%")
    
    # ================== 消融实验评估 ==================
    if args.model == "all" or args.eval_only:
        print("\n" + "─"*65)
        print("  [阶段 4/4] 消融实验评估（3×2=6 组实验）")
        print("─"*65)
        t0 = time.time()
        
        from run_ablation import run_ablation_evaluation
        results, summary = run_ablation_evaluation()
        
        elapsed = (time.time() - t0) / 60
        print(f"\n  ✓ 评估完成 | 耗时 {elapsed:.1f} 分钟")
    
    total_elapsed = (time.time() - total_start) / 60
    print(f"\n{'='*65}")
    print(f"  全部实验完成！总耗时：{total_elapsed:.1f} 分钟")
    print(f"  检查点目录：{CHECKPOINT_DIR}")
    print(f"  图表目录：  {FIGURES_DIR}")
    print(f"  结果目录：  {RESULTS_DIR}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
