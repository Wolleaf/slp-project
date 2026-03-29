#!/bin/bash
# ============================================================
# run_cloud.sh - 云服务器（RTX 5090）一键全自动化实验脚本
#
# 适配说明：
#   1. 自动切换云端配置（config_cloud.py -> config.py）
#   2. 顺序执行指定模型的训练与全量消融实验
#
# 使用方法：
#   chmod +x run_cloud.sh
#   ./run_cloud.sh                          # 运行全部（模型1/2/3 + 消融评估）
#   ./run_cloud.sh --models 1               # 只运行模型1（阶段A）
#   ./run_cloud.sh --models 2               # 只运行模型2（阶段B）
#   ./run_cloud.sh --models 3               # 只运行模型3（阶段C）
#   ./run_cloud.sh --models 1,2             # 运行模型1和模型2
#   ./run_cloud.sh --models 2,3 --ablation  # 运行模型2/3 + 消融评估
#   ./run_cloud.sh 2>&1 | tee run_cloud.log
# ============================================================

set -e

# ---- 解析命令行参数 ----
RUN_MODELS=""    # 为空表示运行全部
RUN_ABLATION=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --models)
            RUN_MODELS="$2"
            shift 2
            ;;
        --ablation)
            RUN_ABLATION=1
            shift
            ;;
        *)
            echo "未知参数: $1"
            echo "用法: $0 [--models 1,2,3] [--ablation]"
            exit 1
            ;;
    esac
done

# 判断是否要运行某个模型（1/2/3）
should_run() {
    local model="$1"
    if [[ -z "$RUN_MODELS" ]]; then
        return 0  # 未指定 → 运行全部
    fi
    echo "$RUN_MODELS" | tr ',' '\n' | grep -qx "$model"
}

echo "========================================================"
echo "  语音真伪检测（Audio Deepfake Detection）自动化实验"
echo "  开始时间：$(date '+%Y-%m-%d %H:%M:%S')"
if [[ -n "$RUN_MODELS" ]]; then
    echo "  指定运行模型：$RUN_MODELS"
    [[ "$RUN_ABLATION" -eq 1 ]] && echo "  + 消融评估阶段"
else
    echo "  运行模式：全部模型 (1, 2, 3) + 消融评估"
fi
echo "========================================================"

# ---- 1. 自动化配置切换 ----
echo ""
echo "[1/4] 正在应用云端配置文件 (config_cloud.py)..."
if [ -f "src/config_cloud.py" ]; then
    if [ -f "src/config.py" ]; then
        mv src/config.py src/config_local_backup.py
        echo "  - 已备份本地配置至 src/config_local_backup.py"
    fi
    cp src/config_cloud.py src/config.py
    echo "  - 已成功将 config_cloud.py 应用为当前 config.py"
else
    echo "  ! 警告: 未找到 src/config_cloud.py，将使用现有 config.py"
fi

# ---- 3. 硬件环境检查 ----
echo ""
echo "[2/4] 硬件环境快照："
python -c "
import torch
import sys
print(f'  - Python: {sys.version.split()[0]}')
print(f'  - PyTorch: {torch.__version__}')
print(f'  - CUDA 可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  - GPU: {torch.cuda.get_device_name(0)}')
    print(f'  - 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"

# ---- 4. 依次执行训练与评估 ----
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [3/4] 核心实验流程启动"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 确保在 src 目录下运行脚本
cd src

if should_run 1; then
    echo ""
    echo ">> 阶段 A: 训练模型1 (MFCC + CNN Baseline)"
    python train_model1.py
fi

if should_run 2; then
    echo ""
    echo ">> 阶段 B: 训练模型2 (Frozen SSL + ResNet)"
    python train_model2.py
fi

if should_run 3; then
    echo ""
    echo ">> 阶段 C: 训练模型3 (Fine-tuned SSL + Augmentation)"
    python train_model3.py
fi

# 消融评估：未指定模型时默认运行；或者指定了 --ablation 时运行
if [[ -z "$RUN_MODELS" ]] || [[ "$RUN_ABLATION" -eq 1 ]]; then
    echo ""
    echo ">> 阶段 D: 全量消融实验评估与图表生成"
    python run_ablation.py
fi

# ---- 5. 结果汇总 ----
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [4/4] 实验任务全部完成！"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  结束时间：$(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "  [数据] 汇总 JSON: ../results/ablation_results.json"
echo "  [可视化] 对比图: ../figures/eer_comparison.png"
echo "  [模型] 最佳权重: ../checkpoints/"
echo ""
ls -lh ../results/ablation_results.json 2>/dev/null || echo "  (未找到结果文件)"
echo "========================================================"
