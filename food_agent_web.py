"""
🍽️ 食物营养智能体 - Gradio Web 界面
基于 nutrition_query.py 的多盘菜营养分析系统
"""

import json
import os
import tempfile
from pathlib import Path
from PIL import Image

import gradio as gr

# 导入 nutrition_query 中的所有功能
from nutrition_query import (
    load_calorieclip,
    load_china_food_db,
    process_single_image_nutrition,
    CALORIE_CLIP_PATH,
    CHINA_FOOD_DB_ROOT,
    DEVICE
)

# ============================================================
# 1. 初始化
# ============================================================

print("🚀 正在启动食物营养智能体...")

# 加载 CalorieCLIP
print("[1] 加载 CalorieCLIP...")
model = load_calorieclip(CALORIE_CLIP_PATH)

# 加载营养数据库
print("[2] 加载营养数据库...")
db_records, db_index = load_china_food_db(CHINA_FOOD_DB_ROOT)

print("✅ 智能体已就绪！")


# ============================================================
# 2. 处理函数
# ============================================================

def analyze_food_image(image, message, history):
    """
    分析食物图片并返回结果
    """
    if image is None:
        history = history or []
        history.append(("系统", "⚠️ 请先上传一张食物图片"))
        return history, ""
    
    # 保存临时图片
    try:
        if isinstance(image, Image.Image):
            img = image
        else:
            img = Image.fromarray(image).convert('RGB')
        
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            img.save(f.name)
            temp_path = f.name
        
        # 调用 nutrition_query 的核心分析函数
        print(f"\n📸 分析图片: {temp_path}")
        report = process_single_image_nutrition(
            temp_path,
            model,
            db_index,
            max_dishes=4
        )
        
        # 删除临时文件
        try:
            os.unlink(temp_path)
        except:
            pass
        
        # 格式化输出
        response = format_report(report)
        
        # 如果有用户问题，追加回答
        if message and message.strip():
            response += f"\n\n💬 关于您的问题「{message}」：\n"
            response += "根据上述营养数据，您可以自行判断是否符合您的需求。"
        
        # 修复：使用正确的消息格式
        history = history or []
        user_msg = message if message else "📸 请分析这张食物图片"
        history.append((user_msg, response))
        return history, ""
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        history = history or []
        history.append(("系统", f"⚠️ 分析失败: {str(e)}"))
        return history, ""


def format_report(report):
    """格式化营养分析报告"""
    lines = []
    lines.append("📊 **食物分析结果**")
    lines.append("")
    
    foods = report.get('foods', [])
    if not foods:
        lines.append("⚠️ 未识别到食物")
        return "\n".join(lines)
    
    for i, food in enumerate(foods, 1):
        name = food.get('name', '未知')
        cal = food.get('calories_kcal', 0)
        weight = food.get('weight_g', 0)
        protein = food.get('protein_g', 0)
        fat = food.get('fat_g', 0)
        carbs = food.get('carbs_g', 0)
        
        lines.append(f"🍽️ **菜品 {i}: {name}**")
        lines.append(f"   - 重量: {weight:.1f} g")
        lines.append(f"   - 热量: {cal:.1f} kcal")
        lines.append(f"   - 蛋白质: {protein:.1f} g")
        lines.append(f"   - 脂肪: {fat:.1f} g")
        lines.append(f"   - 碳水: {carbs:.1f} g")
        lines.append("")
    
    lines.append(f"📈 **总计**")
    lines.append(f"   - 总热量: {report.get('total_calories_kcal', 0):.1f} kcal")
    lines.append(f"   - 总重量: {report.get('total_weight_g', 0):.1f} g")
    lines.append(f"   - 菜品数: {report.get('num_dishes', 0)} 盘")
    
    return "\n".join(lines)


def clear_chat():
    """清除对话"""
    return [], ""


# ============================================================
# 3. 构建 Gradio 界面
# ============================================================

with gr.Blocks(title="🍽️ 食物营养智能体", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("""
    # 🍽️ 食物营养智能体 - 小营
    ### 📸 上传食物图片，AI 自动识别并分析营养信息
    
    > 基于 YOLOv8 + SAM + CalorieCLIP + 中国食物成分数据库
    """)
    
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="💬 对话",
                height=450,
            )
        with gr.Column(scale=1):
            image_input = gr.Image(
                label="📸 上传食物图片",
                type="pil",
                height=250,
            )
    
    with gr.Row():
        msg = gr.Textbox(
            label="💬 输入问题（可选）",
            placeholder="例如：这个热量高吗？适合减脂吗？",
            lines=2,
        )
    
    with gr.Row():
        submit_btn = gr.Button("🚀 分析图片", variant="primary", size="lg")
        clear_btn = gr.Button("🗑️ 清除对话", variant="secondary", size="lg")
    
    with gr.Row():
        gr.Markdown("💡 **快捷提问**：点击下方按钮自动填入问题")
    
    with gr.Row():
        quick1 = gr.Button("🔥 这个热量高吗？", size="sm")
        quick2 = gr.Button("💪 有多少蛋白质？", size="sm")
        quick3 = gr.Button("🥗 适合减脂期吃吗？", size="sm")
        quick4 = gr.Button("📊 详细营养分析", size="sm")
        quick5 = gr.Button("🍚 碳水含量多少？", size="sm")
    
    gr.Markdown("""
    ---
    <div style="text-align: center; color: #999; font-size: 0.85em;">
        🔍 YOLOv8 检测盘子 · 🎯 SAM 精分割 · 📊 CalorieCLIP 预测热量 · 📚 中国食物成分数据库
    </div>
    """)
    
    # ===== 事件绑定 =====
    
    # 分析按钮
    submit_btn.click(
        analyze_food_image,
        inputs=[image_input, msg, chatbot],
        outputs=[chatbot, msg]
    )
    
    # 回车提交
    msg.submit(
        analyze_food_image,
        inputs=[image_input, msg, chatbot],
        outputs=[chatbot, msg]
    )
    
    # 清除对话
    clear_btn.click(
        clear_chat,
        outputs=[chatbot, msg]
    )
    
    # 快捷问题（填充到输入框）
    quick1.click(lambda: "这个热量高吗？", outputs=[msg])
    quick2.click(lambda: "有多少蛋白质？", outputs=[msg])
    quick3.click(lambda: "适合减脂期吃吗？", outputs=[msg])
    quick4.click(lambda: "请详细分析这份食物的营养", outputs=[msg])
    quick5.click(lambda: "碳水含量有多少？", outputs=[msg])


# ============================================================
# 4. 启动
# ============================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860, help="端口号")
    parser.add_argument("--share", action="store_true", help="生成公网链接")
    args = parser.parse_args()
    
    print(f"\n🌐 启动 Web 界面: http://127.0.0.1:{args.port}")
    print("   按 Ctrl+C 退出\n")
    
    demo.launch(
        server_name="127.0.0.1",
        server_port=args.port,
        share=args.share,
        debug=False
    )