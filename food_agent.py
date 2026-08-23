"""
🍽️ 食物营养智能体 (Food Nutrition Agent)
基于 Ollama + Qwen2.5 的多轮对话交互系统
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import deque

import requests
from PIL import Image

# 导入 nutrition_query 中的核心功能
from nutrition_query import (
    load_calorieclip,
    load_china_food_db,
    process_single_image_nutrition,
    CHINA_FOOD_DB_ROOT,
    CALORIE_CLIP_PATH,
    DEVICE
)

# ============================================================
# 1. Ollama 客户端
# ============================================================

class OllamaClient:
    """Ollama API 客户端"""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "qwen2.5:7b"):
        self.base_url = base_url
        self.model = model
        self._check_ollama()
    
    def _check_ollama(self):
        """检查 Ollama 服务是否可用"""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if resp.status_code == 200:
                models = [m['name'] for m in resp.json().get('models', [])]
                if self.model not in models:
                    print(f"⚠️ 模型 {self.model} 未在 Ollama 中找到")
                    print(f"   可用模型: {models}")
                    print(f"   请运行: ollama pull {self.model}")
                else:
                    print(f"✅ Ollama 连接成功，使用模型: {self.model}")
                return True
        except Exception as e:
            print(f"⚠️ Ollama 连接失败: {e}")
            print("   请确保 Ollama 已启动: ollama serve")
        return False
    
    def chat(self, messages: List[Dict], stream: bool = False) -> str:
        """发送聊天请求"""
        try:
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": stream,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                }
            }
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=120
            )
            if resp.status_code == 200:
                return resp.json().get("message", {}).get("content", "")
            else:
                return f"⚠️ API 错误: {resp.status_code}"
        except Exception as e:
            return f"⚠️ 请求失败: {e}"


# ============================================================
# 2. 对话记忆管理
# ============================================================

class ConversationMemory:
    """多轮对话上下文管理"""
    
    def __init__(self, max_history: int = 10):
        self.history = deque(maxlen=max_history)
        self.current_food_data = None
        self.user_profile = {
            "goal": None,  # "lose_weight", "gain_muscle", "maintain"
            "age": None,
            "gender": None,
            "height": None,
            "weight": None,
            "activity_level": None,
        }
    
    def add_message(self, role: str, content: str):
        """添加消息到历史"""
        self.history.append({"role": role, "content": content})
    
    def get_history(self) -> List[Dict]:
        """获取完整历史"""
        return list(self.history)
    
    def get_recent(self, n: int = 3) -> List[Dict]:
        """获取最近 n 条消息"""
        return list(self.history)[-n:]
    
    def set_food_data(self, data: Dict):
        """设置当前分析的食物数据"""
        self.current_food_data = data
    
    def get_food_data(self) -> Optional[Dict]:
        """获取当前食物数据"""
        return self.current_food_data
    
    def update_profile(self, **kwargs):
        """更新用户画像"""
        for key, value in kwargs.items():
            if key in self.user_profile:
                self.user_profile[key] = value
    
    def get_profile(self) -> Dict:
        """获取用户画像"""
        return self.user_profile


# ============================================================
# 3. 智能体核心
# ============================================================

class FoodNutritionAgent:
    """食物营养智能体"""
    
    def __init__(self, ollama_model: str = "qwen2.5:7b"):
        self.ollama = OllamaClient(model=ollama_model)
        self.memory = ConversationMemory()
        
        # 加载 CalorieCLIP 和营养数据库
        print("\n[初始化] 加载 CalorieCLIP...")
        self.model = load_calorieclip(CALORIE_CLIP_PATH)
        
        print("[初始化] 加载营养数据库...")
        self.db_records, self.db_index = load_china_food_db(CHINA_FOOD_DB_ROOT)
        
        self.system_prompt = self._build_system_prompt()
        
        print("\n✅ 智能体初始化完成！")
    
    def _build_system_prompt(self) -> str:
        """构建系统提示词"""
        return """你是一个专业的食物营养分析助手，名叫"小营"。

## 你的能力
1. 分析用户上传的食物图片，识别食物种类、估算分量和热量
2. 回答关于食物营养的问题（蛋白质、脂肪、碳水等）
3. 根据用户的健康目标提供饮食建议

## 交互规则
- 当用户上传图片时，你会调用视觉分析模块，返回食物识别和营养数据
- 用户可以用自然语言追问，如：
  - "这个有多少蛋白质？"
  - "如果只吃一半呢？"
  - "这个热量高吗？"
  - "我现在在减脂，适合吃这个吗？"
- 支持指代消解："刚才那道菜"、"左边那盘"

## 回复风格
- 专业、友好、简洁
- 用表格或列表呈现营养数据
- 给出量化的建议

## 用户当前信息
{profile}

## 当前食物数据
{food_data}
"""
    
    def _get_context(self) -> Dict:
        """获取当前上下文"""
        return {
            "profile": json.dumps(self.memory.get_profile(), ensure_ascii=False, indent=2),
            "food_data": json.dumps(self.memory.get_food_data() or {}, ensure_ascii=False, indent=2)
        }
    
    def _format_food_report(self, report: Dict) -> str:
        """格式化食物分析报告"""
        lines = []
        lines.append("📊 **食物分析结果**")
        lines.append("")
        
        for food in report.get('foods', []):
            name = food.get('name', '未知')
            cal = food.get('calories_kcal', 0)
            weight = food.get('weight_g', 0)
            protein = food.get('protein_g', 0)
            fat = food.get('fat_g', 0)
            carbs = food.get('carbs_g', 0)
            
            lines.append(f"🍽️ **{name}**")
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
    
    def analyze_image(self, image_path: str) -> Dict:
        """分析图片并返回营养数据"""
        print(f"\n📸 正在分析图片: {Path(image_path).name}")
        
        if self.model is None:
            return {"error": "CalorieCLIP 模型未加载"}
        
        try:
            report = process_single_image_nutrition(
                image_path,
                self.model,
                self.db_index,
                max_dishes=4
            )
            
            # 存入记忆
            self.memory.set_food_data(report)
            
            return report
        except Exception as e:
            return {"error": f"分析失败: {e}"}
    
    def chat(self, user_input: str, image_path: Optional[str] = None) -> str:
        """
        处理用户输入
        - 如果包含图片路径或用户上传图片，先分析图片
        - 然后调用 LLM 生成回复
        """
        # 1. 处理图片
        if image_path:
            report = self.analyze_image(image_path)
            if "error" not in report:
                food_report = self._format_food_report(report)
                self.memory.add_message("system", f"已分析图片，结果：\n{food_report}")
        
        # 2. 检查用户输入是否包含图片相关指令
        image_keywords = ["图片", "照片", "这张", "分析", "看看", "识别"]
        if any(kw in user_input for kw in image_keywords) and not image_path:
            # 用户可能想上传图片但没有提供
            pass
        
        # 3. 保存用户消息
        self.memory.add_message("user", user_input)
        
        # 4. 构建完整对话上下文
        messages = [
            {"role": "system", "content": self.system_prompt.format(**self._get_context())}
        ]
        messages.extend(self.memory.get_history())
        
        # 5. 调用 Ollama
        print("\n💭 思考中...")
        response = self.ollama.chat(messages)
        
        # 6. 保存助手回复
        self.memory.add_message("assistant", response)
        
        return response
    
    def interactive_shell(self):
        """交互式命令行界面"""
        print("\n" + "=" * 60)
        print("🍽️ 食物营养智能体 - 交互式对话")
        print("=" * 60)
        print("\n📖 使用说明:")
        print("  - 输入图片路径: /image E:/path/to/food.jpg")
        print("  - 直接输入问题: 这个热量高吗？")
        print("  - 设置目标: /goal lose_weight")
        print("  - 查看状态: /status")
        print("  - 清除记忆: /clear")
        print("  - 退出: /quit 或 exit")
        print("\n" + "-" * 60)
        
        while True:
            try:
                user_input = input("\n你> ").strip()
                if not user_input:
                    continue
                
                # 处理命令
                if user_input.lower() in ["/quit", "exit", "quit"]:
                    print("👋 再见！祝您饮食健康！")
                    break
                
                if user_input == "/clear":
                    self.memory = ConversationMemory()
                    print("✅ 对话记忆已清除")
                    continue
                
                if user_input == "/status":
                    print("\n📊 当前状态:")
                    print(f"  用户目标: {self.memory.get_profile().get('goal', '未设置')}")
                    print(f"  对话轮数: {len(self.memory.get_history())}")
                    print(f"  当前食物数据: {'已分析' if self.memory.get_food_data() else '无'}")
                    continue
                
                if user_input.startswith("/goal"):
                    goal = user_input.replace("/goal", "").strip()
                    self.memory.update_profile(goal=goal)
                    print(f"✅ 目标已设置为: {goal}")
                    continue
                
                if user_input.startswith("/image"):
                    image_path = user_input.replace("/image", "").strip()
                    if Path(image_path).exists():
                        response = self.chat("请分析这张图片", image_path=image_path)
                        print(f"\n小营> {response}")
                    else:
                        print(f"⚠️ 文件不存在: {image_path}")
                    continue
                
                # 普通对话
                response = self.chat(user_input)
                print(f"\n小营> {response}")
                
            except KeyboardInterrupt:
                print("\n👋 再见！")
                break
            except Exception as e:
                print(f"⚠️ 错误: {e}")


# ============================================================
# 4. Gradio Web 界面（可选）
# ============================================================

def create_gradio_interface(agent: FoodNutritionAgent):
    """创建 Gradio Web 界面"""
    try:
        import gradio as gr
        
        def chat_with_image(message, history, image):
            """Gradio 聊天函数"""
            if image is not None:
                # 保存临时图片
                temp_path = "temp_food_image.jpg"
                image.save(temp_path)
                response = agent.chat(message, image_path=temp_path)
            else:
                response = agent.chat(message)
            return response
        
        with gr.Blocks(title="🍽️ 食物营养智能体", theme=gr.themes.Soft()) as demo:
            gr.Markdown("""
            # 🍽️ 食物营养智能体 - 小营
            ### 上传食物图片，智能分析营养信息
            """)
            
            with gr.Row():
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(height=500)
                    msg = gr.Textbox(
                        label="输入您的问题",
                        placeholder="例如：这个热量高吗？或者上传图片让我分析...",
                        lines=2
                    )
                    with gr.Row():
                        clear = gr.Button("清除对话")
                        image_input = gr.Image(type="pil", label="上传食物图片")
            
            def respond(message, chat_history, image):
                if not message and image is None:
                    return chat_history, ""
                
                if image is not None:
                    temp_path = "temp_food_image.jpg"
                    image.save(temp_path)
                    response = agent.chat(message or "请分析这张食物图片", image_path=temp_path)
                else:
                    response = agent.chat(message)
                
                chat_history.append((message, response))
                return chat_history, ""
            
            msg.submit(respond, [msg, chatbot, image_input], [chatbot, msg])
            clear.click(lambda: ([], ""), None, [chatbot, msg])
        
        return demo
    except ImportError:
        print("⚠️ Gradio 未安装，跳过 Web 界面")
        return None


# ============================================================
# 5. 主入口
# ============================================================

def main():
    """主入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="食物营养智能体")
    parser.add_argument("--model", default="qwen2.5:7b", help="Ollama 模型名称")
    parser.add_argument("--web", action="store_true", help="启动 Web 界面")
    parser.add_argument("--image", type=str, help="直接分析单张图片")
    parser.add_argument("--query", type=str, help="配合 --image 使用，提问内容")
    args = parser.parse_args()
    
    # 创建智能体
    agent = FoodNutritionAgent(ollama_model=args.model)
    
    # 命令行模式
    if args.image and args.query:
        response = agent.chat(args.query, image_path=args.image)
        print(f"\n小营> {response}")
        return
    
    if args.image:
        report = agent.analyze_image(args.image)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    
    # Web 界面
    if args.web:
        demo = create_gradio_interface(agent)
        if demo:
            demo.launch(share=False, server_name="127.0.0.1", server_port=7860)
        else:
            print("⚠️ 无法启动 Web 界面，回退到命令行模式")
            agent.interactive_shell()
        return
    
    # 默认：交互式命令行
    agent.interactive_shell()


if __name__ == "__main__":
    main()