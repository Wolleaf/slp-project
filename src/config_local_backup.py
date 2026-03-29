"""
config_cloud.py - 云服务器（RTX 5090 32GB）专用配置
迁移至 5090 后，将此文件替换 config.py 中的对应参数

主要变化：
  1. SSL_MODEL_NAME → wav2vec2-large（更强大，768→1024维特征）
  2. SSL_FEATURE_DIM → 1024
  3. BATCH_SIZE → 64（32GB显存足够）
  4. EVAL_BATCH_SIZE → 128
  5. UNFREEZE_LAYERS → Large 模型的后 2/4 层（layer 22-23 / 20-23）
  6. NUM_WORKERS → 8（云服务器通常 CPU 核心更多）
"""

import os

# ============================================================
# 路径配置（与 config.py 相同，根据云服务器实际路径调整）
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT, "archive", "LA", "LA")

TRAIN_AUDIO_DIR = os.path.join(DATA_ROOT, "ASVspoof2019_LA_train", "flac")
DEV_AUDIO_DIR   = os.path.join(DATA_ROOT, "ASVspoof2019_LA_dev",   "flac")
EVAL_AUDIO_DIR  = os.path.join(DATA_ROOT, "ASVspoof2019_LA_eval",  "flac")

PROTOCOL_DIR   = os.path.join(DATA_ROOT, "ASVspoof2019_LA_cm_protocols")
TRAIN_PROTOCOL = os.path.join(PROTOCOL_DIR, "ASVspoof2019.LA.cm.train.trn.txt")
DEV_PROTOCOL   = os.path.join(PROTOCOL_DIR, "ASVspoof2019.LA.cm.dev.trl.txt")
EVAL_PROTOCOL  = os.path.join(PROTOCOL_DIR, "ASVspoof2019.LA.cm.eval.trl.txt")

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
FIGURES_DIR    = os.path.join(PROJECT_ROOT, "figures")
RESULTS_DIR    = os.path.join(PROJECT_ROOT, "results")

# ============================================================
# 全局种子（保持与本地一致）
# ============================================================
SEED = 42

# ============================================================
# 音频预处理（与本地一致）
# ============================================================
SAMPLE_RATE = 16000
MAX_LENGTH  = 64000
LABEL_MAP = {"bonafide": 0, "spoof": 1}

# ============================================================
# MFCC 特征参数（与本地一致）
# ============================================================
N_MFCC    = 40
N_FFT     = 512
HOP_LENGTH = 160
N_MELS    = 80

# ============================================================
# 训练超参数（5090 优化版）
# ============================================================
BATCH_SIZE    = 64         # 32GB 显存可以翻倍
NUM_EPOCHS    = 20         # 更多轮次，更完整的收敛
BACKEND_LR    = 1e-3
SSL_LR        = 1e-5
WEIGHT_DECAY  = 1e-4
NUM_WORKERS   = 8          # 云服务器多核 CPU
EARLY_STOP_PATIENCE = 5
GRAD_CLIP_NORM = 1.0

# ============================================================
# 大模型配置（5090 专用：Large 版本）
# ============================================================
# 使用纯预训练版 Large（非 ASR 微调版），输出特征更稳定
# 注意：wav2vec2-large-960h 是 ASR 专用（Wav2Vec2ForCTC），用 Wav2Vec2Model
# 加载会导致 masked_spec_embed 缺失、lm_head 残留，最终输出 NaN。
SSL_MODEL_NAME  = "facebook/wav2vec2-large"
SSL_FEATURE_DIM = 1024     # Large 版本输出1024维（base版本为768维）

# ============================================================
# 渐进式解冻配置（Large 模型共 24 层 Transformer）
# ============================================================
UNFREEZE_EPOCH_1 = 5
UNFREEZE_EPOCH_2 = 8
# Large 模型：encoder.layers.0 ~ encoder.layers.23
UNFREEZE_LAYERS_1 = [22, 23]              # 最后 2 层
UNFREEZE_LAYERS_2 = [20, 21, 22, 23]     # 最后 4 层

# ============================================================
# 数据增强配置（与本地一致）
# ============================================================
AUG_PROB       = 0.5
SNR_MIN        = 5.0
SNR_MAX        = 20.0
QUANT_BITS_MIN = 4
QUANT_BITS_MAX = 8

# ============================================================
# 退化测试集配置（与本地一致）
# ============================================================
DEGRADE_SNR       = 5.0
DEGRADE_HP_FREQ   = 300
DEGRADE_LP_FREQ   = 3400
DEGRADE_QUANT_BITS = 4

# ============================================================
# 评估配置（5090 优化版）
# ============================================================
EVAL_BATCH_SIZE = 128      # 更大的评估批次，更快
FIGURE_DPI = 300
