# download_images.py
# 下载 MM-Food-100K 数据集中的所有图片到本地
# 图片保存在 E:/大作业/dataset/images/ 目录下

import os
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import time
from datasets import load_dataset

# ==================== 配置 ====================
IMAGE_SAVE_DIR = "E:/大作业/dataset/images"
BATCH_SIZE = 1000  # 每下载1000张休息一下
MAX_IMAGES = None  # 设为 None 下载全部，或设数字如 5000 只下载5000张

# ==================== 创建目录 ====================
os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)

# ==================== 加载数据集 ====================
print("正在加载数据集...")
ds = load_dataset(
    "Humanbased-AI/MM-Food-100K",
    cache_dir="E:/大作业/mm_food_cache",
    split="train"
)
print(f"数据集加载完成，共 {len(ds)} 张图片")

# ==================== 下载函数 ====================
def download_image(url, save_path, max_retries=3):
    """下载单张图片，失败时重试"""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=30, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                # 统一转换为 RGB
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img.save(save_path, 'JPEG', quality=85)
                return True
            else:
                print(f"状态码 {response.status_code}: {url}")
                return False
        except Exception as e:
            print(f"尝试 {attempt+1}/{max_retries} 失败: {e}")
            time.sleep(2)
    return False

# ==================== 下载主循环 ====================
print(f"开始下载图片到: {IMAGE_SAVE_DIR}")
print(f"总图片数: {len(ds)}")

# 限制下载数量
total = len(ds)
if MAX_IMAGES is not None and MAX_IMAGES < total:
    total = MAX_IMAGES
    print(f"只下载前 {total} 张")

success_count = 0
fail_count = 0

for i in tqdm(range(total), desc="下载进度"):
    item = ds[i]
    url = item['image_url']
    
    # 生成文件名
    filename = f"{i:06d}.jpg"
    save_path = os.path.join(IMAGE_SAVE_DIR, filename)
    
    # 如果文件已存在，跳过
    if os.path.exists(save_path):
        success_count += 1
        continue
    
    # 下载
    if download_image(url, save_path):
        success_count += 1
    else:
        fail_count += 1
    
    # 每下载一批打印统计
    if (i + 1) % BATCH_SIZE == 0:
        print(f"\n已下载 {i+1}/{total}，成功: {success_count}，失败: {fail_count}")
        time.sleep(0.5)

# ==================== 统计结果 ====================
print("\n" + "=" * 50)
print(f"下载完成！")
print(f"成功: {success_count}")
print(f"失败: {fail_count}")
print(f"总处理: {success_count + fail_count}")
print(f"图片保存目录: {IMAGE_SAVE_DIR}")
print("=" * 50)