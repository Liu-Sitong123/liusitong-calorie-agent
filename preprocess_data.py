# preprocess_data.py
# 将 MM-Food-100K 数据集划分为 train/val/test
# 并生成 metadata.csv 文件

import os
import json
import pandas as pd
import numpy as np
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from collections import Counter
from datasets import load_dataset

# ==================== 配置 ====================
DATASET_ROOT = "E:/大作业/dataset"
IMAGE_SRC_DIR = os.path.join(DATASET_ROOT, "images")
TRAIN_DIR = os.path.join(DATASET_ROOT, "train")
VAL_DIR = os.path.join(DATASET_ROOT, "val")
TEST_DIR = os.path.join(DATASET_ROOT, "test")

# 划分比例
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

# ==================== 创建目录 ====================
for dir_path in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    os.makedirs(dir_path, exist_ok=True)

print("=" * 60)
print("MM-Food-100K 数据预处理")
print("=" * 60)

# ==================== 加载数据集 ====================
print("正在加载数据集标注...")
ds = load_dataset(
    "Humanbased-AI/MM-Food-100K",
    cache_dir="E:/大作业/mm_food_cache",
    split="train"
)
print(f"数据集加载完成，共 {len(ds)} 条标注")

# ==================== 解析字段 ====================
def parse_nutritional_profile(nutritional_str):
    """解析 nutritional_profile JSON 字符串"""
    try:
        if isinstance(nutritional_str, str):
            data = json.loads(nutritional_str)
        else:
            data = nutritional_str
        return {
            'calories': data.get('calories_kcal', None),
            'protein': data.get('protein_g', None),
            'fat': data.get('fat_g', None),
            'carbs': data.get('carbohydrate_g', None),
            'sugar': data.get('sugar_g', None),
            'fiber': data.get('fiber_g', None),
            'sodium': data.get('sodium_mg', None)
        }
    except:
        return None

def parse_portion_size(portion_str):
    """解析 portion_size JSON 字符串"""
    try:
        if isinstance(portion_str, str):
            data = json.loads(portion_str)
        else:
            data = portion_str
        return data
    except:
        return None

# ==================== 构建完整数据表 ====================
print("正在解析数据...")
data_records = []

for i in tqdm(range(len(ds)), desc="解析进度"):
    item = ds[i]
    
    # 基础信息
    record = {
        'index': i,
        'image_filename': f"{i:06d}.jpg",
        'image_path': os.path.join(IMAGE_SRC_DIR, f"{i:06d}.jpg"),
        'dish_name': item.get('dish_name', ''),
        'food_type': item.get('food_type', ''),
        'cooking_method': item.get('cooking_method', ''),
        'ingredients': item.get('ingredients', ''),
        'camera_or_phone_prob': float(item.get('camera_or_phone_prob', 0)),
        'food_prob': float(item.get('food_prob', 0)),
        'sub_dt': int(item.get('sub_dt', 0))
    }
    
    # 解析营养成分
    nutrition = parse_nutritional_profile(item.get('nutritional_profile', ''))
    if nutrition:
        record.update(nutrition)
    else:
        record.update({'calories': None, 'protein': None, 'fat': None, 'carbs': None})
    
    # 解析分量
    portion = parse_portion_size(item.get('portion_size', ''))
    if portion:
        record['portion_size'] = json.dumps(portion, ensure_ascii=False)
        # 计算总重量
        total_weight = 0
        if isinstance(portion, list):
            for p in portion:
                if isinstance(p, dict):
                    weight_str = p.get('weight', '')
                    if 'g' in weight_str:
                        try:
                            total_weight += float(weight_str.replace('g', '').strip())
                        except:
                            pass
        record['total_weight_g'] = total_weight if total_weight > 0 else None
    else:
        record['portion_size'] = None
        record['total_weight_g'] = None
    
    # 中餐标记（检测dish_name是否含中文字符）
    import re
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
    record['is_chinese'] = bool(chinese_pattern.search(record['dish_name']))
    
    data_records.append(record)

df = pd.DataFrame(data_records)
print(f"数据表构建完成，共 {len(df)} 条记录")

# ==================== 数据清洗 ====================
print("\n数据清洗...")
print(f"原始记录数: {len(df)}")

# 筛选有图片且有卡路里的记录
df = df[df['calories'].notna() & df['calories'] > 0]
print(f"有卡路里标注的记录: {len(df)}")

# 筛选食物置信度
df = df[df['food_prob'] > 0.5]
print(f"food_prob > 0.5 的记录: {len(df)}")

# 检查图片是否存在
def check_image_exists(path):
    return os.path.exists(path)

df['image_exists'] = df['image_path'].apply(check_image_exists)
df = df[df['image_exists'] == True]
print(f"图片已下载的记录: {len(df)}")

# 删除重复的dish_name（保留一个）
# df = df.drop_duplicates(subset=['dish_name'])
# print(f"去重后的记录: {len(df)}")

# 过滤异常卡路里（1-1500 kcal）
df = df[(df['calories'] >= 10) & (df['calories'] <= 1500)]
print(f"卡路里范围过滤后的记录: {len(df)}")

# ==================== 划分数据集 ====================
print("\n划分数据集...")

# 分层采样：按 food_type 分层
if len(df) > 100:
    # 按 food_type 分层划分
    train_df, temp_df = train_test_split(
        df, 
        train_size=TRAIN_RATIO,
        stratify=df['food_type'],
        random_state=RANDOM_SEED
    )
    val_df, test_df = train_test_split(
        temp_df,
        train_size=VAL_RATIO / (VAL_RATIO + TEST_RATIO),
        stratify=temp_df['food_type'],
        random_state=RANDOM_SEED
    )
else:
    # 数据太少，随机划分
    train_df, temp_df = train_test_split(
        df, train_size=TRAIN_RATIO, random_state=RANDOM_SEED
    )
    val_df, test_df = train_test_split(
        temp_df, train_size=VAL_RATIO / (VAL_RATIO + TEST_RATIO), random_state=RANDOM_SEED
    )

print(f"训练集: {len(train_df)}")
print(f"验证集: {len(val_df)}")
print(f"测试集: {len(test_df)}")

# ==================== 复制图片到对应文件夹 ====================
def copy_images(df, target_dir, desc):
    """复制图片到目标目录"""
    print(f"\n复制 {desc} 图片...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=desc):
        src = row['image_path']
        dst = os.path.join(target_dir, row['image_filename'])
        if os.path.exists(src):
            shutil.copy2(src, dst)
        else:
            print(f"警告: 文件不存在 {src}")

copy_images(train_df, TRAIN_DIR, "训练集")
copy_images(val_df, VAL_DIR, "验证集")
copy_images(test_df, TEST_DIR, "测试集")

# ==================== 保存元数据 ====================
print("\n保存元数据...")

# 保存完整元数据
train_df.to_csv(os.path.join(TRAIN_DIR, "metadata.csv"), index=False, encoding='utf-8-sig')
val_df.to_csv(os.path.join(VAL_DIR, "metadata.csv"), index=False, encoding='utf-8-sig')
test_df.to_csv(os.path.join(TEST_DIR, "metadata.csv"), index=False, encoding='utf-8-sig')

# 保存整体统计信息
stats = {
    "total_images": len(df),
    "train_count": len(train_df),
    "val_count": len(val_df),
    "test_count": len(test_df),
    "chinese_count": int(df[df['is_chinese'] == True].shape[0]),
    "food_type_distribution": df['food_type'].value_counts().to_dict(),
    "calories_stats": {
        "min": float(df['calories'].min()),
        "max": float(df['calories'].max()),
        "mean": float(df['calories'].mean()),
        "std": float(df['calories'].std())
    }
}

with open(os.path.join(DATASET_ROOT, "dataset_stats.json"), "w", encoding='utf-8') as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)

# ==================== 输出统计 ====================
print("\n" + "=" * 60)
print("数据集预处理完成！")
print("=" * 60)
print(f"最终数据目录: {DATASET_ROOT}")
print(f"  train: {len(train_df)} 张")
print(f"  val:   {len(val_df)} 张")
print(f"  test:  {len(test_df)} 张")
print(f"总计: {len(df)} 张")
print(f"中餐占比: {stats['chinese_count']}/{len(df)} ({stats['chinese_count']/len(df)*100:.1f}%)")
print(f"食物类型分布:")
for k, v in stats['food_type_distribution'].items():
    print(f"  {k}: {v} 张")
print(f"\n元数据已保存到各子目录的 metadata.csv")
print(f"统计信息已保存到: {DATASET_ROOT}/dataset_stats.json")
print("=" * 60)