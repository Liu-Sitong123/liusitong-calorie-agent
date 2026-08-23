"""
CalorieCLIP 分量估计 + 评估 (第二问)
流程:
1. 从 ECUSTFD 数据集加载图片和真实重量 (GT)
2. 用 CalorieCLIP 预测每张图片 → 反推重量
3. 计算 MAE 和 RE
4. 支持多角度融合 (同一食物多个角度取中位数)
"""

import os
import sys
import re
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
import warnings

# Windows 控制台可能使用 GBK，导致 emoji/中文打印失败；强制 UTF-8 输出更稳定
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

warnings.filterwarnings('ignore')

# ============================================================
# 1. 配置
# ============================================================

CALORIE_CLIP_PATH = "E:/大作业/CalorieCLIP"
ECUSTFD_ROOT = "E:/大作业/ECUSTFD-resized--master"
DENSITY_XLS = "E:/大作业/ECUSTFD-resized--master/density.xls"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# 食物密度表 (用于热量→重量)
FOOD_CALORIE_DENSITY = {
    'apple': 0.52, 'banana': 0.89, 'bread': 2.65, 'bun': 2.23,
    'doughnut': 3.50, 'egg': 1.44, 'fired_dough_twist': 4.00,
    'grape': 0.69, 'lemon': 0.29, 'litchi': 0.66, 'mango': 0.60,
    'mix': 0.90, 'mooncake': 4.20, 'orange': 0.47, 'pear': 0.57,
    'peach': 0.39, 'plum': 0.46, 'qiwi': 0.61, 'sachima': 4.70,
    'tomato': 0.18, 'default': 0.90
}


# ============================================================
# 2. 加载 CalorieCLIP (从本地)
# ============================================================

def load_calorieclip(local_path):
    if local_path not in sys.path:
        sys.path.insert(0, local_path)
    
    try:
        from calorie_clip import CalorieCLIP
        model = CalorieCLIP.from_pretrained(local_path).to(DEVICE)
        model.eval()
        print("✅ CalorieCLIP 加载成功!")
        return model
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return None


# ============================================================
# 3. 加载 ECUSTFD 数据集
# ============================================================

def load_ecustfd_data(density_xls, ecustfd_root):
    """加载数据集，提取图片路径、类别、真实重量"""
    
    df = pd.read_excel(density_xls, sheet_name=None)
    
    # 构建 ID → 重量和类别
    id_to_info = {}
    temp_records = defaultdict(list)
    
    for sheet_name, sheet_df in df.items():
        for _, row in sheet_df.iterrows():
            vals = [str(v).strip() for v in row.values]
            if all(v == '' or v == 'nan' for v in vals):
                continue
            first = vals[0].lower()
            if first in FOOD_CALORIE_DENSITY and len(vals) < 4:
                continue
            try:
                img_id = vals[0].lower()
                rec_type = vals[1].lower()
                weight = float(vals[3])
                temp_records[img_id].append((rec_type, weight))
            except:
                continue
    
    for img_id, records in temp_records.items():
        main_category = records[0][0]
        total_weight = sum(w for _, w in records)
        id_to_info[img_id] = {
            'category': main_category,
            'weight': total_weight
        }
    
    # 扫描图片文件夹
    jpg_folder = os.path.join(ecustfd_root, "JPEGImages")
    if not os.path.exists(jpg_folder):
        jpg_folder = os.path.join(ecustfd_root, "JPGImages")
    if not os.path.exists(jpg_folder):
        jpg_folder = ecustfd_root
    
    img_info = []
    for filename in os.listdir(jpg_folder):
        if not filename.upper().endswith(('.JP', '.JPG', '.JPEG', '.PNG')):
            continue
        name = os.path.splitext(filename)[0]
        match = re.match(r'([a-zA-Z]+)(\d+)', name)
        base_id = match.group(1).lower() + match.group(2) if match else name.lower()
        
        if base_id in id_to_info:
            info = id_to_info[base_id]
            img_info.append({
                'img_id': base_id,
                'category': info['category'],
                'weight': info['weight'],
                'path': os.path.join(jpg_folder, filename),
                'filename': filename
            })
    
    print(f"加载 {len(img_info)} 张图片")
    return img_info


# ============================================================
# 4. 单张图片预测
# ============================================================

def predict_weight(model, image_path, category=None):
    """用 CalorieCLIP 预测重量"""
    try:
        img = Image.open(image_path).convert("RGB")

        # 这里 model 是本地加载的 CalorieCLIP 实例，使用它自带的 preprocess
        # 不能再访问不存在的 model.clip_model / model.clip_model.visual
        img_tensor = model.preprocess(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            calories = model(img_tensor)
            if isinstance(calories, torch.Tensor):
                calories = calories.detach().cpu().reshape(-1)[0].item()
            else:
                calories = float(calories)

        # 确定密度
        density = FOOD_CALORIE_DENSITY.get(category, FOOD_CALORIE_DENSITY['default'])
        weight_g = calories / density if density and density > 0 else 0.0

        return weight_g, calories

    except Exception:
        return None, None


# ============================================================
# 5. 评估函数 (计算 MAE 和 RE)
# ============================================================

def evaluate_predictions(gt_weights, pred_weights):
    """计算 MAE 和 RE"""
    gt = np.array(gt_weights)
    pred = np.array(pred_weights)
    
    # 过滤无效预测
    valid = pred > 0
    if not np.any(valid):
        return {'mae': float('inf'), 're': float('inf'), 'n': 0, 'total': len(pred)}
    
    gt_valid = gt[valid]
    pred_valid = pred[valid]
    
    mae = np.mean(np.abs(pred_valid - gt_valid))
    re = np.mean(np.abs(pred_valid - gt_valid) / (gt_valid + 1e-8)) * 100
    
    return {
        'mae': round(mae, 2),
        're': round(re, 2),
        'n': len(pred_valid),
        'total': len(pred)
    }


def split_train_test_by_category(df, test_ratio=0.2, random_seed=42):
    """按类别分层划分 train/test，避免同一类别样本跨集泄露。"""
    train_frames = []
    test_frames = []

    for category, group in df.groupby('category', sort=False):
        if len(group) < 2:
            # 只有 1 个样本时，放入训练集，测试集为空
            train_frames.append(group.copy())
            continue

        rng = np.random.RandomState(random_seed + hash(category) % 10000)
        idx = np.arange(len(group))
        rng.shuffle(idx)

        test_size = max(1, int(round(len(group) * test_ratio)))
        if test_size >= len(group):
            test_size = len(group) - 1
        if test_size <= 0:
            test_size = 1

        test_idx = idx[:test_size]
        train_idx = idx[test_size:]

        if len(train_idx) == 0:
            train_idx = idx[:-1]
        if len(test_idx) == 0:
            test_idx = idx[-1:]

        train_frames.append(group.iloc[train_idx].copy())
        test_frames.append(group.iloc[test_idx].copy())

    train_df = pd.concat(train_frames, ignore_index=True) if train_frames else pd.DataFrame(columns=df.columns)
    test_df = pd.concat(test_frames, ignore_index=True) if test_frames else pd.DataFrame(columns=df.columns)
    return train_df, test_df


def fit_category_calibration(train_df):
    """仅在训练集上拟合每个类别的线性校准参数 y = a*x + b。"""
    params = {}
    for category, group in train_df.groupby('category', sort=False):
        pred = group['pred_weight'].to_numpy(dtype=float)
        gt = group['gt_weight'].to_numpy(dtype=float)

        if len(pred) < 2 or np.std(pred) < 1e-8:
            params[category] = {'a': 1.0, 'b': 0.0}
            continue

        try:
            slope, intercept = np.polyfit(pred, gt, 1)
            if not np.isfinite(slope) or not np.isfinite(intercept):
                raise ValueError('Non-finite calibration coefficients')
            params[category] = {'a': float(slope), 'b': float(intercept)}
        except Exception:
            params[category] = {'a': 1.0, 'b': 0.0}

    return params


def apply_category_calibration(df, params):
    """对每个类别应用已学习的校准参数。"""
    out = df.copy()
    out['pred_weight_calibrated'] = out.apply(
        lambda r: params.get(r['category'], {'a': 1.0, 'b': 0.0})['a'] * r['pred_weight'] +
                  params.get(r['category'], {'a': 1.0, 'b': 0.0})['b'],
        axis=1
    )
    return out


# ============================================================
# 6. 主实验
# ============================================================

def run_experiment(sample_size=None):
    print("=" * 60)
    print("🍎 第二问: CalorieCLIP 分量估计评估")
    print("=" * 60)
    
    # 1. 加载模型
    print("\n[1] 加载模型...")
    model = load_calorieclip(CALORIE_CLIP_PATH)
    if model is None:
        return
    
    # 2. 加载数据
    print("\n[2] 加载数据...")
    all_data = load_ecustfd_data(DENSITY_XLS, ECUSTFD_ROOT)
    
    # 采样 (如果指定)
    if sample_size and sample_size < len(all_data):
        import random
        random.seed(42)
        all_data = random.sample(all_data, sample_size)
        print(f"采样 {len(all_data)} 张")
    
    # 3. 推理
    print("\n[3] 推理中...")
    
    results = []
    for item in tqdm(all_data, desc="  Progress"):
        pred_w, _ = predict_weight(model, item['path'], item['category'])
        if pred_w is not None:
            results.append({
                'img_id': item['img_id'],
                'category': item['category'],
                'gt_weight': item['weight'],
                'pred_weight': pred_w,
                'filename': item['filename']
            })
    
    # 4. 评估（先划分 train/test，再在训练集拟合校准，测试集上评估）
    print(f"\n[4] 评估 ({len(results)} 张有效图片)...")

    raw_df = pd.DataFrame(results)
    if raw_df.empty:
        print("\n⚠️ 没有有效预测结果，无法继续评估。")
        return {'mae': float('inf'), 're': float('inf'), 'n': 0, 'total': 0}, raw_df

    train_df, test_df = split_train_test_by_category(raw_df, test_ratio=0.2, random_seed=42)

    if test_df.empty:
        print("\n⚠️ 训练/测试划分后测试集为空，无法进行严格评估。")
        return evaluate_predictions(raw_df['gt_weight'].tolist(), raw_df['pred_weight'].tolist()), raw_df

    # 原始测试集指标
    raw_gt = test_df['gt_weight'].tolist()
    raw_pred = test_df['pred_weight'].tolist()
    raw_metrics = evaluate_predictions(raw_gt, raw_pred)

    # 仅在训练集拟合校准参数
    params = fit_category_calibration(train_df)
    calibrated_test_df = apply_category_calibration(test_df, params)
    calibrated_test_df['pred_weight'] = calibrated_test_df['pred_weight_calibrated']
    metrics = evaluate_predictions(calibrated_test_df['gt_weight'].tolist(), calibrated_test_df['pred_weight'].tolist())

    print("\n" + "=" * 60)
    print("📊 评估结果（train/test 分离）")
    print("=" * 60)
    print(f"  训练集样本: {len(train_df)}")
    print(f"  测试集样本: {len(test_df)}")
    print(f"  原始测试 MAE: {raw_metrics['mae']:.2f} g")
    print(f"  原始测试 RE: {raw_metrics['re']:.2f} %")
    print(f"  校准后测试 MAE: {metrics['mae']:.2f} g")
    print(f"  校准后测试 RE: {metrics['re']:.2f} %")
    print(f"  有效图片: {metrics['n']} / {metrics['total']}")

    # 5. 保存结果
    print("\n[5] 保存结果...")
    os.makedirs("outputs", exist_ok=True)
    calibrated_test_df.to_csv("outputs/calorieclip_results.csv", index=False)
    print("  💾 已保存: outputs/calorieclip_results.csv")

    # 6. 按类别查看（仅测试集）
    print("\n[6] 各类别 MAE（测试集）:")
    calibrated_test_df['error'] = abs(calibrated_test_df['gt_weight'] - calibrated_test_df['pred_weight'])
    category_mae = calibrated_test_df.groupby('category')['error'].mean().sort_values()
    for cat, mae in category_mae.items():
        print(f"  {cat}: {mae:.2f} g")

    return metrics, calibrated_test_df


# ============================================================
# 7. 多角度融合实验
# ============================================================

def run_multi_angle_experiment():
    """
    多角度融合: 同一食物的多个角度取中位数
    看融合后 MAE 是否降低
    """
    print("\n" + "=" * 60)
    print("📐 多角度融合实验")
    print("=" * 60)
    
    # 1. 加载模型
    model = load_calorieclip(CALORIE_CLIP_PATH)
    if model is None:
        return
    
    # 2. 加载数据
    all_data = load_ecustfd_data(DENSITY_XLS, ECUSTFD_ROOT)
    
    # 3. 按 img_id 分组
    groups = defaultdict(list)
    for item in all_data:
        groups[item['img_id']].append(item)
    
    # 4. 只保留有多个角度的组
    multi_groups = {k: v for k, v in groups.items() if len(v) >= 2}
    print(f"有多个角度的食物: {len(multi_groups)} 个")
    
    # 5. 对每组进行融合
    all_predictions = []
    for img_id, items in tqdm(multi_groups.items(), desc="  融合中"):
        preds = []
        for item in items:
            pred_w, _ = predict_weight(model, item['path'], item['category'])
            if pred_w is not None:
                preds.append(pred_w)
        
        if len(preds) >= 2:
            # 中位数融合
            fused_pred = np.median(preds)
            gt_weight = items[0]['weight']
            all_predictions.append({
                'img_id': img_id,
                'category': items[0]['category'],
                'gt_weight': gt_weight,
                'fused_pred': fused_pred,
                'n_angles': len(preds),
                'std': np.std(preds)
            })
    
    # 6. 评估
    gt = [r['gt_weight'] for r in all_predictions]
    pred = [r['fused_pred'] for r in all_predictions]
    metrics = evaluate_predictions(gt, pred)
    
    print(f"\n融合后:")
    print(f"  有效样本: {metrics['n']}")
    print(f"  MAE: {metrics['mae']:.2f} g")
    print(f"  RE: {metrics['re']:.2f} %")
    
    # 7. 对比单张 vs 融合
    print("\n  各食物误差变化:")
    for r in all_predictions[:10]:
        single_preds = [r['fused_pred']]  # 简化
        print(f"    {r['img_id']}: GT={r['gt_weight']}g, 融合={r['fused_pred']:.1f}g, {r['n_angles']}角度")
    
    return metrics


# ============================================================
# 8. 主入口
# ============================================================

if __name__ == "__main__":
    # 标准评估 (全量数据)
    metrics, df = run_experiment(sample_size=200)
    
    # 多角度融合评估
    # run_multi_angle_experiment()