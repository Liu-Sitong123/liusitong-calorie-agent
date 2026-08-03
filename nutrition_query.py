# nutrition_query.py (完整版)
import pandas as pd
import json

# ========== Bonus③: 标准份量字典 ==========
STANDARD_PORTIONS = {
    "Fried Chicken": 150,
    "Noodle Soup": 350,
    "Pho": 400,
    "Rice": 150,
    "Noodle Stir-Fry": 300,
    "Mixed Asian Dish": 350,
    "Bananas": 120,
    "Shrimp and Noodle Appetizer": 200,
    "Dried Noodles": 80,
    "Oranges": 150,
    "Pizza": 200,
    "Hamburger": 250,
    "Sandwiches": 200,
    "Spaghetti": 250,
    "Steak": 200,
    "Chicken": 150,
    "Salad": 200,
    "Sushi": 200,
    "Ramen": 400,
    "Curry": 300,
    "Tempura": 200,
    "Udon": 350,
    "Fried Rice": 250,
    "Omelet": 150,
    "Toast": 50,
    "Croissant": 60,
    "Pancake": 100,
    "Cake": 100,
    "Ice Cream": 120,
}

class NutritionModule:
    def __init__(self, db_path="nutrition_db.csv"):
        self.df = pd.read_csv(db_path)
        self.name_to_row = {row["dish_name"]: idx for idx, row in self.df.iterrows()}
        print(f"✅ 已加载 {len(self.df)} 种食物的营养数据")

    def query(self, food_name, weight_g):
        """
        查询单种食物的营养信息（含异常检测 + 卡路里等级 + 标准份量对比）
        """
        # 1. 精确匹配
        if food_name in self.name_to_row:
            row = self.df.loc[self.name_to_row[food_name]]
        else:
            # 2. 模糊匹配
            matched = False
            for name in self.name_to_row.keys():
                if food_name.lower() in name.lower() or name.lower() in food_name.lower():
                    row = self.df.loc[self.name_to_row[name]]
                    matched = True
                    break
            if not matched:
                return {
                    "name": food_name,
                    "weight_g": weight_g,
                    "calories_kcal": None,
                    "protein_g": None,
                    "fat_g": None,
                    "carbs_g": None,
                    "error": "未找到营养数据"
                }

        # 按重量换算
        scale = weight_g / 100.0
        
        result = {
            "name": food_name,
            "weight_g": weight_g,
            "calories_kcal": float(row["calories_per_100g"] * scale),
            "protein_g": float(row["protein_per_100g"] * scale),
            "fat_g": float(row["fat_per_100g"] * scale),
            "carbs_g": float(row["carbs_per_100g"] * scale),
            "calorie_label": row["calorie_label"],  # Bonus④
        }
        
        # ========== Bonus③: 标准份量对比 ==========
        standard = STANDARD_PORTIONS.get(food_name, None)
        if standard is None:
            # 尝试模糊匹配标准份量
            for key in STANDARD_PORTIONS.keys():
                if food_name.lower() in key.lower() or key.lower() in food_name.lower():
                    standard = STANDARD_PORTIONS[key]
                    break
        
        if standard:
            result["standard_portion_g"] = standard
            diff_percent = (weight_g - standard) / standard * 100
            if diff_percent > 20:
                result["vs_standard"] = f"⬆️ 比标准份量多 {diff_percent:.0f}%"
            elif diff_percent < -20:
                result["vs_standard"] = f"⬇️ 比标准份量少 {abs(diff_percent):.0f}%"
            else:
                result["vs_standard"] = "✅ 接近标准份量"
        
        return result

    def calculate_meal(self, foods):
        """
        计算整餐的营养信息（含各食物热量占比）
        """
        results = []
        total_calories = 0
        total_protein = 0
        total_fat = 0
        total_carbs = 0

        for food in foods:
            result = self.query(food["name"], food["weight_g"])
            results.append(result)

            if result.get("calories_kcal") is not None:
                total_calories += result["calories_kcal"]
                total_protein += result.get("protein_g", 0) or 0
                total_fat += result.get("fat_g", 0) or 0
                total_carbs += result.get("carbs_g", 0) or 0

        # ========== Bonus⑤: 各食物热量占比 ==========
        for result in results:
            if result.get("calories_kcal") and total_calories > 0:
                result["calories_percentage"] = round(result["calories_kcal"] / total_calories * 100, 1)

        return {
            "foods": results,
            "total_calories_kcal": round(total_calories, 1),
            "total_protein_g": round(total_protein, 1),
            "total_fat_g": round(total_fat, 1),
            "total_carbs_g": round(total_carbs, 1),
        }


# ============ 测试 ============
if __name__ == "__main__":
    nutrition = NutritionModule()

    # 测试1：查询单种食物（完整功能）
    print("=" * 60)
    print("测试1：单食物查询（含卡路里等级 + 标准份量对比）")
    print("=" * 60)
    result = nutrition.query("Fried Chicken", 180)
    print(json.dumps(result, ensure_ascii=False, indent=2))

    print("\n" + "=" * 60)
    print("测试2：整餐营养汇总（含各食物热量占比）")
    print("=" * 60)
    meal = nutrition.calculate_meal([
        {"name": "Fried Chicken", "weight_g": 180},
        {"name": "Noodle Soup", "weight_g": 300},
        {"name": "Rice", "weight_g": 150}
    ])
    print(json.dumps(meal, ensure_ascii=False, indent=2))