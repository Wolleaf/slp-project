# 语音真伪检测实验流程说明 (flow.md)

## 项目概览

本项目基于 **ASVspoof 2019 LA** 数据集，通过三组消融实验系统论证了两大创新点的有效性：

1. **渐进式解冻微调（Progressive Unfreezing）** — 提升大模型在防伪任务中的特征适配能力
2. **GPU 实时数据增强（On-the-fly Augmentation）** — 提升跨信道鲁棒性

实验采用 **「3 模型 × 2 测试场景」** 的交叉设计，输出 **6 组 EER 数据** 形成完整论证链。

---

## 项目文件结构

```
Project/
├── archive/                        # ASVspoof 2019 数据集
│   └── LA/LA/
│       ├── ASVspoof2019_LA_train/flac/    # 训练音频（25,380 条）
│       ├── ASVspoof2019_LA_dev/flac/      # 验证音频（24,844 条）
│       ├── ASVspoof2019_LA_eval/flac/     # 测试音频（71,933 条）
│       └── ASVspoof2019_LA_cm_protocols/  # 标签协议文件
├── src/
│   ├── config.py          # 全局配置（超参数、路径）
│   ├── config_cloud.py    # 云服务器 5090 专用配置
│   ├── utils.py           # 工具函数（EER、早停、检查点）
│   ├── dataset.py         # 数据集加载、增强、退化处理
│   ├── models.py          # 三个模型定义 + 渐进式解冻
│   ├── evaluate.py        # 评估函数 + DET/EER 图表绘制
│   ├── train_model1.py    # 模型1 训练脚本（本地跑通用）
│   ├── train_model2.py    # 模型2 训练脚本
│   ├── train_model3.py    # 模型3 训练脚本
│   ├── run_ablation.py    # 消融实验评估（加载三模型，生成图表）
│   └── run_all.py         # 主入口（依次训练+评估）
├── checkpoints/           # 模型权重保存目录
│   ├── model1_baseline.pth
│   ├── model2_frozen_ssl.pth
│   └── model3_finetuned_aug.pth
├── figures/               # 实验图表输出目录
│   ├── training_model1.png
│   ├── training_model2.png
│   ├── training_model3.png
│   ├── det_curve_clean.png
│   ├── det_curve_degraded.png
│   └── eer_comparison.png
├── results/               # 实验数据输出目录
│   ├── train_model1.log
│   ├── train_model2.log
│   ├── train_model3.log
│   ├── ablation_results.json
│   └── ablation_results_full.npz
├── run_cloud.sh           # 云服务器一键运行脚本
├── README.md              # 项目说明
├── experiment.md          # 实验详细流程文档
├── flow.md                # 本文件：实验流程记录
└── learn.md               # 学习笔记
```

---

## 实验设计说明

### 三个对比模型

| 模型 | 前端特征 | 后端分类器 | 数据增强 | 参数微调 |
|------|---------|-----------|---------|---------|
| **Model 1** (Baseline) | MFCC（手工特征） | 轻量级 CNN | ❌ | 无大模型 |
| **Model 2** (Frozen SSL) | Wav2Vec2（冻结） | ResNet-like | ❌ | ❌ |
| **Model 3** (Ours) | Wav2Vec2（渐进解冻） | ResNet-like | ✅ 实时增强 | ✅ 渐进解冻 |

### 两个测试场景

| 场景 | 描述 | 处理方式 |
|------|------|---------|
| **考场A** (Clean) | 原版干净测试集 | 无任何退化 |
| **考场B** (Degraded) | 恶劣现实场景 | SNR=5dB噪声 + 电话频段滤波 + 4bit量化 |

### 6 组 EER 结果矩阵（实际实验结果）

| 模型 | 考场A (干净) | 考场B (退化) |
|------|-------------|-------------|
| 模型1: MFCC+CNN | 10.13% | 54.53% (完全失效) |
| 模型2: 冻结SSL | 13.65% | 35.96% |
| 模型3: 本方案 | **2.68%** | **24.32%** |

---

## 实验执行顺序

### 第一步：环境准备

```bash
conda activate llmdevelop
# 已安装依赖（本次实验确认可用）：
# torch 2.9.1+cu130, torchaudio 2.9.1, transformers, scikit-learn
# matplotlib, pandas, soundfile, tqdm
```

> ⚠️ **注意**：torchaudio 2.9.x 的 `torchaudio.load()` 需要 `torchcodec`，
> 本项目改用 `soundfile` 直接读取 .flac 文件，无需额外安装。

### 第二步：训练模型1（本地已完成 ✅）

```bash
cd src
python train_model1.py
```

**实际结果**（RTX 5060，约 25 分钟）：

| Epoch | Loss | Val EER |
|-------|------|---------|
| 1 | 0.2139 | 13.35% |
| 2 | 0.2124 | 11.67% |
| 3 | 0.1822 | 10.14% |
| 4 | 0.1566 | 8.55% |
| 5 | 0.1372 | 8.40% |
| 6 | 0.1230 | 7.73% |
| 7 | 0.1076 | 7.46% |
| 8 | 0.0987 | 6.21% |
| 9 | 0.0890 | 6.39% |
| 10 | 0.0836 | 6.39% |
| 11 | 0.0753 | 5.80% |
| 12 | 0.0719 | 5.65% |
| 13 | 0.0667 | 5.62% |
| 14 | 0.0635 | 5.53% |
| **15** | **0.0644** | **5.40% ★** |

- **最终最佳验证 EER: 5.40%**（15个epoch全部完成）
- 检查点已保存：`checkpoints/model1_baseline.pth`
- 训练曲线：`figures/training_model1.png`

### 第三步：训练模型2（云服务器已完成 ✅）

```bash
python train_model2.py
```

- 云服务器 5090（32GB显存）：已完成 Wav2Vec2-large + ResNet 的联合评估。

### 第四步：训练模型3（云服务器已完成 ✅）

```bash
python train_model3.py
```

- 包含渐进式解冻（Epoch 5/8）+ 实时数据增强。
- 最终成功防止了灾难性遗忘并取得了最佳 EER。

### 第五步：消融实验评估（云服务器已完成 ✅）

```bash
python run_ablation.py
# 或一次性运行全部：
python run_all.py --eval_only  # 模型已训练完成，直接提取 6 组交叉对照结果
```

---

## 关键技术实现说明

### 1. 数据加载（dataset.py）

- 使用 `soundfile.read()` 替代 `torchaudio.load()`（兼容新版本）
- 支持随机截取（训练时）和固定截取（评估时）
- `OnTheFlyAugmentor`：三种随机增强（白噪声、量化压缩、电话频段滤波）
- `HeavyDegradation`：三重叠加退化（考场B专用，确定性处理）

### 2. 模型架构（models.py）

- **BaselineModel**：MFCC → 3层CNN → 全连接（约10万参数）
- **FrozenSSLModel**：`torch.no_grad()` 下运行 Wav2Vec2，只训练 ResNet
- **FineTunedSSLModel**：渐进式解冻，Epoch 5 解冻最后2层，Epoch 8 解冻最后4层

### 3. EER 计算（utils.py）

使用 `sklearn.metrics.roc_curve` 推导：
```
FAR = FPR, FRR = 1 - TPR
EER = (FAR + FRR) / 2 at FAR ≈ FRR
```

### 4. 早停机制（utils.py: EarlyStopping）

- 监控验证集 EER（越低越好）
- patience=5：连续5个epoch未降低则停止

---

## 云服务器迁移注意事项

1. **替换配置**：将 `config.py` 中的以下内容修改（或使用 `config_cloud.py`）：
   - `SSL_MODEL_NAME = "facebook/wav2vec2-large-960h"` （1024维特征）
   - `SSL_FEATURE_DIM = 1024`
   - `BATCH_SIZE = 64`
   - `UNFREEZE_LAYERS_1 = [22, 23]`
   - `UNFREEZE_LAYERS_2 = [20, 21, 22, 23]`

2. **运行命令**：
   ```bash
   chmod +x run_cloud.sh
   ./run_cloud.sh 2>&1 | tee run_cloud.log
   ```

3. **数据同步**：确保 `archive/` 目录已完整上传（约 25GB）

---

## 实验输出文件说明

| 文件 | 内容 | 用途 |
|------|------|------|
| `checkpoints/model1_baseline.pth` | 模型1权重 | 评估和报告 |
| `checkpoints/model2_frozen_ssl.pth` | 模型2权重 | 评估和报告 |
| `checkpoints/model3_finetuned_aug.pth` | 模型3权重 | 评估和报告 |
| `figures/training_model*.png` | 训练曲线（Loss+EER） | 报告插图 |
| `figures/det_curve_clean.png` | 考场A的DET曲线对比 | 报告核心图 |
| `figures/det_curve_degraded.png` | 考场B的DET曲线对比 | 报告核心图 |
| `figures/eer_comparison.png` | EER分组柱状图 | 报告核心图 |
| `results/ablation_results.json` | 6组EER数值（JSON） | 填写报告数据 |
| `results/train_model*.log` | 训练日志 | 调试和记录 |

---

## 当前进度

- [x] 环境配置（自动化 requirements.txt 生成与安装）
- [x] 代码编写（config, utils, dataset, models, evaluate, train x3, ablation, run_cloud）
- [x] 数据集验证（121,461 条音频解析与预设截断）
- [x] 流程测试（smoke test 通过）
- [x] 模型1训练完成（本地 RTX 5060）
- [x] 模型2训练完成（云服务器 RTX 5090）
- [x] 模型3训练完成（云服务器 RTX 5090 + 实时数据增强）
- [x] 消融实验评估（6组交叉测试 EER 数据提取成功）
- [x] 图表生成（DET曲线、EER柱状图已保存至 figures/）
- [x] **报告撰写（LaTeX 论文级报告和 Reference 生成完毕，等待最后答辩）**
