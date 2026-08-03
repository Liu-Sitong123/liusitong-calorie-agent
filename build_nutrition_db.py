# build_nutrition_db.py (完整版)
import json
from datasets import load_dataset
import pandas as pd
import numpy as np

print("正在加载 MM-Food-100K 标注数据...")
ds = load_dataset("Humanbased-AI/MM-Food-100K", split="train")

nutrition_db = []
all_dish_names = set()

for i in range(len(ds)):
    item = ds[i]
    
    dish_name = item.get("dish_name", "")
    if not dish_name:
        continue
    
    all_dish_names.add(dish_name)
    
    nutrition_str = item.get("nutritional_profile", "{}")
    try:
        nutrition = json.loads(nutrition_str) if isinstance(nutrition_str, str) else nutrition_str
    except:
        continue
    
    nutrition_db.append({
        "dish_name": dish_name,
        "calories_per_100g": nutrition.get("calories_kcal", 0),
        "protein_per_100g": nutrition.get("protein_g", 0),
        "fat_per_100g": nutrition.get("fat_g", 0),
        "carbs_per_100g": nutrition.get("carbohydrate_g", 0),
    })

df = pd.DataFrame(nutrition_db)
df = df.drop_duplicates(subset=["dish_name"])

print(f"原始数据: {len(df)} 种食物")

# ========== Bonus①: 数据清洗与异常值处理 ==========
def clean_nutrition_data(df):
    """清洗营养数据，过滤和修复异常值"""
    original_count = len(df)
    
    # 1. 过滤热量为0或负值的（标注缺失）
    df = df[df["calories_per_100g"] > 0]
    print(f"  过滤热量<=0: {original_count - len(df)} 条")
    
    # 2. 过滤明显异常：热量超过 800 kcal/100g 或低于 20 kcal/100g
    df = df[(df["calories_per_100g"] >= 20) & (df["calories_per_100g"] <= 800)]
    print(f"  过滤热量异常(<20或>800): {original_count - len(df)} 条（累计）")
    
    # 3. 蛋白质、脂肪、碳水不能为负数
    before = len(df)
    df["protein_per_100g"] = df["protein_per_100g"].clip(lower=0)
    df["fat_per_100g"] = df["fat_per_100g"].clip(lower=0)
    df["carbs_per_100g"] = df["carbs_per_100g"].clip(lower=0)
    
    # 4. 三个宏量营养素不能全为0
    df = df[(df["protein_per_100g"] > 0) | (df["fat_per_100g"] > 0) | (df["carbs_per_100g"] > 0)]
    print(f"  过滤三大营养素全为0: {original_count - len(df)} 条（累计）")
    
    return df

df = clean_nutrition_data(df)

# ========== Bonus④: 添加卡路里等级标签 ==========
def get_calorie_label(calories):
    """根据每100g热量返回等级标签"""
    if calories < 100:
        return "🟢 低卡"
    elif calories < 250:
        return "🟡 中卡"
    else:
        return "🔴 高卡"

df["calorie_label"] = df["calories_per_100g"].apply(get_calorie_label)

# 统计各等级数量
print(f"\n卡路里等级分布:")
print(df["calorie_label"].value_counts())

# 保存到 CSV
df.to_csv("nutrition_db.csv", index=False, encoding="utf-8-sig")

print(f"\n✅ 营养数据库已构建！共 {len(df)} 种食物（清洗后）")
print(f"   保存路径: nutrition_db.csv")