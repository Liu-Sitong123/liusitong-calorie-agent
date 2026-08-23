# liusitong-calorie-agent
# 🍽️ 智能食物营养分析系统

> 基于视觉语言模型、多模型融合与大语言对话的完整食物分析解决方案

---

## 📋 项目简介

本项目是一个完整的智能食物营养分析系统，涵盖三个核心模块：

| 模块 | 功能 | 核心模型 |
|------|------|----------|
| **模块一：食物识别** | 零样本食物分类 | English-CLIP + 检测器 + CoT |
| **模块二：分量估计** | 从图像估计食物重量 | CalorieCLIP + 类别校准 |
| **模块三：营养分析系统** | 多盘菜分割 + 营养查询 + 智能问答 | YOLOv8 + SAM + CalorieCLIP + Qwen2.5 |

### 核心成果

- **食物识别**：Top-1 准确率 **68.64%**（55类高频食物）
- **分量估计**：MAE **21.34g**，RE **14.34%**（CalorieCLIP + 类别校准）
- **多盘菜分析**：YOLOv8+SAM 精准分割，自动输出每盘菜的营养信息

---

## 📁 项目结构

```
E:/大作业/
├── food_recognition_experiment.py   # 模块一：食物识别实验
├── portion_estimation.py            # 模块二：分量估计实验
├── nutrition_query.py               # 模块三：多盘菜营养分析
├── food_agent.py                    # 智能体核心（命令行交互）
├── food_agent_web.py                # Gradio Web 界面
├── outputs/                         # 实验结果输出目录
├── CalorieClip/                     # CalorieCLIP 模型（本地）
├── yolov8n-seg.pt                   # YOLOv8 分割模型权重
├── sam_vit_h_4b8939.pth             # SAM 模型权重（约2.4GB）
└── china-food-composition-data-main/ # 中国食物成分数据库
```

---

## 🚀 快速开始

### 1. 环境搭建

#### 基础依赖
```bash
# Python 3.10+
pip install torch torchvision torchaudio
pip install numpy pandas pillow tqdm
pip install transformers open-clip-torch
pip install gradio requests
```

#### 图像处理与分割
```bash
pip install opencv-python
pip install ultralytics  # YOLOv8
pip install segment-anything  # SAM
```

#### 大语言模型（可选，用于智能问答）
```bash
# 安装 Ollama（Windows/Linux/Mac）
# 访问 https://ollama.com/download 下载安装

# 拉取 Qwen2.5 模型
ollama pull qwen2.5:7b
```

### 2. 模型准备

| 模型 | 下载地址 | 存放位置 |
|------|----------|----------|
| **CalorieCLIP** | 本地路径 | `E:/大作业/CalorieClip/` |
| **YOLOv8n-seg** | `ultralytics` 自动下载 | `E:/大作业/yolov8n-seg.pt` |
| **SAM ViT-H** | [下载链接](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) | `E:/大作业/sam_vit_h_4b8939.pth` |

### 3. 数据集配置

| 数据集 | 用途 | 路径 |
|--------|------|------|
| **MM-Food-100K** | 食物识别 | `E:/大作业/dataset/` |
| **ECUSTFD-resized** | 分量估计 | `E:/大作业/ECUSTFD-resized--master/` |
| **中国食物成分数据库** | 营养查询 | `E:/大作业/china-food-composition-data-main/` |

---

## 🔧 各模块使用指南

### 模块一：食物识别实验

```bash
python food_recognition_experiment.py
```

**功能**：
- English-CLIP vs Chinese-CLIP vs Qwen2-VL 基线对比
- 检测器（OWL-ViT）和 CoT 推理消融实验
- 跨场景（标准/真实）泛化分析
- 结果可视化与 CSV 导出

**输出**：
- `experiment_results.csv`：详细识别结果
- `experiment_summary.json`：实验汇总
- `experiment_results.png`：结果图表

### 模块二：分量估计实验

```bash
python portion_estimation.py
```

**功能**：
- CalorieCLIP 热量预测 → 密度换算 → 重量估计
- 按类别分层训练/测试集划分（8:2）
- 类别线性校准（仅训练集拟合，测试集评估）
- 多角度融合实验（同一食物多张图片取中位数）

**输出**：
- `outputs/calorieclip_results.csv`：校准后结果

**关键结果**：

| 评估方式 | MAE (g) | RE (%) |
|----------|---------|--------|
| 原始预测 | 126.96 | 145.08 |
| 类别校准后 | **21.34** | **14.34** |

### 模块三：多盘菜营养分析

```bash
python nutrition_query.py --image /path/to/food.jpg
# 或批量处理目录
python nutrition_query.py --dir E:/大作业/cc_food_100/rgb
```

**功能**：
- YOLOv8 检测盘子/碗区域
- SAM 精分割食物区域
- CalorieCLIP 预测每盘菜热量
- 中国食物成分数据库查询营养信息
- 汇总输出结构化营养报告

### 模块四：智能问答交互

```bash
# 命令行交互模式
python food_agent.py

# Web 界面模式（需要 gradio）
python food_agent.py --web

# 直接分析单张图片
python food_agent.py --image /path/to/food.jpg --query "这个热量高吗？"

# Gradio Web 界面（独立启动）
python food_agent_web.py
```

**交互命令**：

| 命令 | 说明 |
|------|------|
| `/image /path/to/food.jpg` | 分析图片 |
| `/goal lose_weight` | 设置目标（减脂） |
| `/status` | 查看状态 |
| `/clear` | 清除记忆 |
| `/quit` | 退出 |

---

## 📊 实验全景汇总

| 问题 | 最佳方案 | 核心指标 |
|------|----------|----------|
| 食物识别 | English-CLIP + 检测器 + CoT | Top-1: **68.64%** |
| 分量估计 | CalorieCLIP + 类别校准 | MAE: **21.34g**, RE: **14.34%** |
| 多盘菜分析 | YOLOv8 + SAM + 营养数据库 | 2盘菜识别，总热量874.7 kcal |
| 智能问答 | Ollama + Qwen2.5 | 多轮对话，上下文记忆 |

---

## 🔬 核心算法说明

### 1. 分量估计：从热量反推重量

```
图像 → CalorieCLIP → 热量(kcal) → 类别校准 → 重量(g)
```

类别线性校准：$\hat{y} = \alpha_c \cdot y + \beta_c$

其中 $\alpha_c, \beta_c$ 仅在训练集上拟合，测试集不参与。

### 2. 多盘菜分割策略

```
YOLO 检测盘子 → SAM 精分割 → OpenCV 兜底 → 网格切分（最后）
```

**设计原则**：只有完全检测不到区域时才使用网格切分，避免过度分割。

### 3. 智能体对话架构

```
用户输入 → 记忆管理 → 上下文构建 → Qwen2.5 → 个性化回复
              ↑
        系统提示词 + 食物数据 + 用户画像
```

---

## 📁 输出文件说明

| 文件 | 来源 | 说明 |
|------|------|------|
| `experiment_results.csv` | 模块一 | 每张图片的识别结果 |
| `experiment_summary.json` | 模块一 | 实验汇总指标 |
| `experiment_results.png` | 模块一 | 结果可视化图表 |
| `outputs/calorieclip_results.csv` | 模块二 | 校准后的分量估计结果 |
| `experiment_comparison.csv` | 模块二 | 多模型融合对比 |
| `Exp1_A_B_C.csv` ~ `Exp4_*.csv` | 模块二 | 各融合实验详细结果 |

---

## ⚠️ 常见问题

### Q1: CalorieCLIP 加载失败
**A**: 检查路径是否正确，确保 `E:/大作业/CalorieClip/calorie_clip.py` 存在。

### Q2: SAM 加载卡住
**A**: SAM 权重约 2.4GB，首次加载需要时间。如果卡住，可以暂时禁用 SAM（注释掉 `load_local_segmentation_models` 中的 SAM 加载部分）。

### Q3: Ollama 连接失败
**A**: 确保 Ollama 服务已启动：`ollama serve` 或检查系统托盘是否有 Ollama 图标。

### Q4: Gradio Web 界面报错
**A**: 降低 Gradio 版本：`pip install gradio==4.44.1`

### Q5: 内存不足
**A**: 
- 减少 `max_dishes` 参数
- 使用 `sample_size` 参数采样测试
- 禁用 SAM（内存占用最大）

---

## 📝 引用说明

本项目基于以下开源模型与数据集：

- **CLIP**: OpenAI (MIT License)
- **YOLOv8**: Ultralytics (AGPL-3.0)
- **SAM**: Meta (Apache-2.0)
- **Qwen2.5**: Alibaba (Apache-2.0)
- **MM-Food-100K**: 学术研究数据集
- **ECUSTFD**: 学术研究数据集
- **中国食物成分数据库**: 中国疾控中心营养与健康所

---
