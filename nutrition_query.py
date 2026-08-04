# nutrition_query.py (完整版 + CoT + 不确定性量化)
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
        # 记录匹配类型，用于不确定性量化
        self.last_match_type = None
        print(f"✅ 已加载 {len(self.df)} 种食物的营养数据")

    def query(self, food_name, weight_g):
        """
        查询单种食物的营养信息（含异常检测 + 卡路里等级 + 标准份量对比）
        """
        # 1. 精确匹配
        if food_name in self.name_to_row:
            row = self.df.loc[self.name_to_row[food_name]]
            self.last_match_type = "exact"
        else:
            # 2. 模糊匹配
            matched = False
            for name in self.name_to_row.keys():
                if food_name.lower() in name.lower() or name.lower() in food_name.lower():
                    row = self.df.loc[self.name_to_row[name]]
                    matched = True
                    self.last_match_type = "fuzzy"
                    break
            if not matched:
                self.last_match_type = "none"
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
        
        # ========== 创新点⑥: Chain-of-Thought 推理 ==========
        result["reasoning"] = self._generate_cot_reasoning(
            result["name"], result["weight_g"], 
            row["calories_per_100g"], result["calories_kcal"]
        )
        
        # ========== 创新点②: 不确定性量化 ==========
        result["uncertainty"] = self._calculate_uncertainty(food_name, weight_g)
        
        return result

    def _generate_cot_reasoning(self, food_name, weight_g, calories_per_100g, total_calories):
        """
        生成 CoT 推理过程（创新点⑥）
        步骤：识别 → 查表 → 换算 → 结果
        """
        reasoning = f"""【CoT 推理过程】
步骤1 - 食物识别：识别为「{food_name}」
步骤2 - 查营养表：该食物每100g含 {calories_per_100g} kcal
步骤3 - 重量换算：当前重量 {weight_g}g = {weight_g/100:.1f} × 100g
步骤4 - 计算热量：{calories_per_100g} × {weight_g/100:.1f} = {total_calories:.1f} kcal"""
        return reasoning

    def _calculate_uncertainty(self, food_name, weight_g):
        """
        计算不确定性（创新点②）
        基于匹配类型和重量估算精度，输出置信区间
        """
        # 基础误差
        base_error = 0.05  # 基础误差 ±5%
        
        # 匹配类型影响
        if self.last_match_type == "exact":
            match_error = 0.02  # 精确匹配，误差小
        elif self.last_match_type == "fuzzy":
            match_error = 0.10  # 模糊匹配，误差大
        else:
            match_error = 0.25  # 未匹配，误差最大
        
        # 重量估算误差（没有实际重量测量，假设 ±10%）
        weight_error = 0.10
        
        # 总误差（平方和开根号，简单叠加）
        total_error = (base_error**2 + match_error**2 + weight_error**2) ** 0.5
        
        # 转换为百分比
        error_percent = total_error * 100
        
        # 根据误差大小给出定性描述
        if error_percent < 10:
            level = "高置信度"
        elif error_percent < 20:
            level = "中置信度"
        else:
            level = "低置信度"
        
        return {
            "error_percent": round(error_percent, 1),
            "confidence_level": level,
            "description": f"估计值 ±{error_percent:.1f}%，{level}"
        }

    def query_with_cot(self, food_name, weight_g):
        """
        专门调用 CoT 推理的接口
        """
        result = self.query(food_name, weight_g)
        if "reasoning" in result:
            print(result["reasoning"])
        return result

    def query_with_uncertainty(self, food_name, weight_g):
        """
        专门调用不确定性量化的接口
        """
        result = self.query(food_name, weight_g)
        if "uncertainty" in result:
            print(f"置信区间: {result['uncertainty']['description']}")
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

        # 整餐的 CoT 推理
        meal_cot = self._generate_meal_cot_reasoning(results, total_calories)
        
        # 整餐的不确定性
        meal_uncertainty = self._calculate_meal_uncertainty(results)

        return {
            "foods": results,
            "total_calories_kcal": round(total_calories, 1),
            "total_protein_g": round(total_protein, 1),
            "total_fat_g": round(total_fat, 1),
            "total_carbs_g": round(total_carbs, 1),
            "meal_reasoning": meal_cot,  # 整餐 CoT
            "meal_uncertainty": meal_uncertainty  # 整餐不确定性
        }

    def _generate_meal_cot_reasoning(self, results, total_calories):
        """整餐的 CoT 推理"""
        reasoning = "【整餐 CoT 推理】\n"
        for r in results:
            if r.get("calories_kcal") is not None:
                reasoning += f"- {r['name']}: {r['weight_g']}g × {r['calories_kcal']/r['weight_g']*100:.1f} kcal/100g = {r['calories_kcal']:.1f} kcal\n"
        reasoning += f"总热量: {total_calories:.1f} kcal"
        return reasoning

    def _calculate_meal_uncertainty(self, results):
        """整餐的不确定性"""
        errors = []
        for r in results:
            if "uncertainty" in r:
                errors.append(r["uncertainty"]["error_percent"])
        if errors:
            avg_error = sum(errors) / len(errors)
            level = "高置信度" if avg_error < 10 else ("中置信度" if avg_error < 20 else "低置信度")
            return {
                "error_percent": round(avg_error, 1),
                "confidence_level": level,
                "description": f"整餐估计值 ±{avg_error:.1f}%，{level}"
            }
        return {
            "error_percent": 15.0,
            "confidence_level": "中置信度",
            "description": "整餐估计值 ±15.0%，中置信度"
        }


# ============ 测试 ============
if __name__ == "__main__":
    nutrition = NutritionModule()

    # 测试1：单食物查询（含 CoT + 不确定性）
    print("=" * 60)
    print("测试1：单食物查询（含 CoT 推理 + 不确定性量化）")
    print("=" * 60)
    result = nutrition.query("Fried Chicken", 180)
    print(json.dumps(result, ensure_ascii=False, indent=2))

    print("\n" + "=" * 60)
    print("测试2：整餐营养汇总（含 CoT + 不确定性）")
    print("=" * 60)
    meal = nutrition.calculate_meal([
        {"name": "Fried Chicken", "weight_g": 180},
        {"name": "Noodle Soup", "weight_g": 300},
        {"name": "Rice", "weight_g": 150}
    ])
    print(json.dumps(meal, ensure_ascii=False, indent=2))
    
    print("\n" + "=" * 60)
    print("测试3：模糊匹配 + 不确定性演示")
    print("=" * 60)
    result_fuzzy = nutrition.query("Fried Chick", 180)  # 故意拼错
    print(json.dumps(result_fuzzy, ensure_ascii=False, indent=2))
