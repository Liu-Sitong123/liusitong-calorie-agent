# ============================================================
# 8. 中国食物营养库 + 多盘菜分解版方案（YOLO+SAM 增强版）
# ============================================================

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

try:
    import cv2
except Exception:
    cv2 = None

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

try:
    from segment_anything import sam_model_registry, SamPredictor
except Exception:
    sam_model_registry = None
    SamPredictor = None

BASE_DIR = Path(__file__).resolve().parent
CALORIE_CLIP_PATH = BASE_DIR / "CalorieClip"
YOLO_SEG_MODEL_PATH = BASE_DIR / "yolov8n-seg.pt"
SAM_MODEL_PATH = BASE_DIR / "sam_vit_h_4b8939.pth"
QWEN_VL_MODEL_PATH = Path(r"C:\Users\Lenovo\.cache\huggingface\hub\models--Qwen--Qwen2-VL-2B-Instruct\snapshots\895c3a49bc3fa70a340399125c650a463535e71c")
CLIP_MODEL_PATH = Path(r"C:\Users\Lenovo\.cache\huggingface\hub\models--openai--clip-vit-base-patch16\snapshots\57c216476eefef5ab752ec549e440a49ae4ae5f3")
CHINA_FOOD_DB_ROOT = Path("E:/大作业/china-food-composition-data-main/json_data_vision_251206_Qwen2-5-VL-72B-Instruct").as_posix()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 默认走本地 CLIP，避免 CPU 上强制加载 Qwen 而卡住；只有显式开启时才走大模型分支。
USE_LARGE_MODEL_FOR_CLASSIFICATION = os.environ.get("USE_LARGE_MODEL", "0").strip().lower() in {"1", "true", "yes", "on"}


def load_calorieclip(model_dir):
    """Load the local CalorieCLIP model from the project directory."""
    model_path = Path(model_dir)
    if not model_path.exists():
        alt = BASE_DIR / "CalorieCLIP"
        if alt.exists():
            model_path = alt
        else:
            alt2 = BASE_DIR / "CalorieClip"
            if alt2.exists():
                model_path = alt2

    if not model_path.exists():
        print(f"⚠️ 未找到 CalorieCLIP 路径: {model_path}")
        return None

    try:
        model_dir_str = str(model_path)
        if model_dir_str not in sys.path:
            sys.path.insert(0, model_dir_str)
        from calorie_clip import CalorieCLIP
        model = CalorieCLIP.from_pretrained(model_dir_str, device=str(DEVICE))
        print(f"✅ CalorieCLIP 加载成功: {model_dir_str}")
        return model
    except Exception as e:
        print(f"⚠️ CalorieCLIP 加载失败: {e}")
        return None

# 全局模型缓存
_SEGMENTATION_CACHE = None
_QWEN_VL_CACHE = None
_CLIP_ZERO_SHOT_CACHE = None
# YOLO 可识别的盘子/碗/容器类别（COCO 数据集类别）
# 如果用的是自定义训练模型，需要替换这些 ID
PLATE_BOWL_CLASSES = {41, 42, 43, 44, 45, 46, 47, 48, 49, 50}


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return float(default)


def _box_area(box):
    x1, y1, x2, y2 = box
    return max(0.0, (x2 - x1) * (y2 - y1))


def _filter_candidate_boxes(boxes, img_shape, max_dishes=4):
    h, w = img_shape[:2]
    valid = []
    for box in boxes:
        x1, y1, x2, y2 = [max(0, int(v)) for v in box]
        x1 = min(w - 1, x1)
        y1 = min(h - 1, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        area = (x2 - x1) * (y2 - y1)
        if area < max(2000, 0.015 * w * h):
            continue
        aspect = max(x2 - x1, y2 - y1) / max(min(x2 - x1, y2 - y1), 1)
        if aspect > 8:
            continue
        valid.append((x1, y1, x2, y2))
    valid = sorted(valid, key=lambda b: _box_area(b), reverse=True)
    return valid[:max_dishes]


def clip_text_label(label):
    text = str(label).lower().strip()
    if not text:
        return "米饭"
    if any(k in text for k in ["rice", "steamed", "fried rice", "米饭", "蒸饭", "炒饭"]):
        return "米饭"
    if any(k in text for k in ["veget", "vege", "green", "青菜", "蔬菜", "菜"]):
        return "青菜"
    if any(k in text for k in ["tomato", "番茄", "西红柿"]):
        return "番茄"
    if any(k in text for k in ["chicken", "鸡", "鸡肉"]):
        return "鸡肉"
    if any(k in text for k in ["fish", "鱼", "海鲜"]):
        return "鱼"
    if any(k in text for k in ["beef", "牛", "牛肉"]):
        return "牛肉"
    if any(k in text for k in ["tofu", "豆腐"]):
        return "豆腐"
    if any(k in text for k in ["noodl", "面", "面条"]):
        return "面条"
    if any(k in text for k in ["apple", "苹果"]):
        return "苹果"
    if any(k in text for k in ["pork", "猪", "猪肉"]):
        return "猪肉"
    if any(k in text for k in ["soup", "汤"]):
        return "汤"
    return "米饭"


def normalize_food_name(name):
    if name is None:
        return ""
    s = str(name).strip().lower()
    s = s.replace("（", "(").replace("）", ")")
    s = s.replace("代表值", "")
    s = re.sub(r"[\s_\-]+", "", s)
    return s


def load_china_food_db(db_root=CHINA_FOOD_DB_ROOT):
    """加载中文营养数据库，返回条目列表和姓名索引。"""
    root = Path(db_root)
    if not root.exists():
        print(f"⚠️ 未找到中国食物营养数据库: {db_root}")
        return [], {}

    records = []
    for json_path in sorted(root.glob('*.json')):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                records.extend(data)
        except Exception:
            continue

    name_index = {}
    for item in records:
        if not isinstance(item, dict):
            continue
        food_name = item.get('foodName') or item.get('name') or item.get('item')
        if not food_name:
            continue
        norm = normalize_food_name(food_name)
        if norm:
            name_index.setdefault(norm, []).append(item)

    print(f"✅ 共加载中国食物营养条目: {len(records)}")
    return records, name_index


def find_food_entry_by_name(food_hint, name_index):
    """按名称匹配中文数据库条目。"""
    if not food_hint:
        return None
    hint = normalize_food_name(food_hint)
    if not hint:
        return None

    if hint in name_index:
        return name_index[hint][0]

    # 模糊匹配：如果 hint 是 key 的子串或 key 是 hint 的子串
    for key, items in name_index.items():
        if hint in key or key in hint:
            return items[0]

    # 进一步：按中文名中最显著的几个词粗略匹配
    words = re.findall(r'[\u4e00-\u9fa5]+|[a-zA-Z]+|\d+', food_hint)
    for key, items in name_index.items():
        score = 0
        for word in words:
            if normalize_food_name(word) in key:
                score += 1
        if score > 0:
            return items[0]

    return None


def compute_nutrition_from_calories(food_name, predicted_calories_kcal, nutrition_db_index):
    """根据总热量和食物名查询每100g营养数据并换算为实际重量。"""
    entry = find_food_entry_by_name(food_name, nutrition_db_index)
    if entry is None:
        # 没找到匹配条目，返回保守默认值
        return {
            'name': food_name,
            'weight_g': max(float(predicted_calories_kcal) / 200.0 * 100.0, 0.0),
            'calories_kcal': float(predicted_calories_kcal),
            'protein_g': 0.0,
            'fat_g': 0.0,
            'carbs_g': 0.0,
            'source': 'default'
        }

    energy_100g = float(str(entry.get('energyKCal', '0')).replace('—', '0').replace('Tr', '0'))
    protein_100g = float(str(entry.get('protein', '0')).replace('—', '0').replace('Tr', '0'))
    fat_100g = float(str(entry.get('fat', '0')).replace('—', '0').replace('Tr', '0'))
    carbs_100g = float(str(entry.get('CHO', '0')).replace('—', '0').replace('Tr', '0'))

    if energy_100g <= 0:
        energy_100g = 200.0

    weight_g = 100.0 * float(predicted_calories_kcal) / energy_100g
    ratio = weight_g / 100.0

    return {
        'name': entry.get('foodName', food_name),
        'weight_g': round(max(weight_g, 0.0), 2),
        'calories_kcal': round(float(predicted_calories_kcal), 2),
        'protein_g': round(max(protein_100g * ratio, 0.0), 2),
        'fat_g': round(max(fat_100g * ratio, 0.0), 2),
        'carbs_g': round(max(carbs_100g * ratio, 0.0), 2),
        'source': 'china_food_db'
    }


def estimate_total_calories_for_region(model, image_crop, category_hint=None):
    """对裁剪出的单盘菜区域估计总卡路里。"""
    try:
        if model is None:
            return 80.0
        
        # 如果输入是 PIL Image
        if hasattr(image_crop, 'convert'):
            img = image_crop.convert('RGB')
        else:
            img = Image.fromarray(image_crop).convert('RGB')
        
        if hasattr(model, 'preprocess'):
            image_tensor = model.preprocess(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                cal = float(model(image_tensor).reshape(-1)[0].item())
        else:
            cal = 80.0
        return max(cal, 1.0)
    except Exception as e:
        print(f"⚠️ 热量预测失败: {e}")
        return 80.0


def load_local_segmentation_models():
    """加载本地 YOLO 和 SAM 模型（带缓存）。"""
    global _SEGMENTATION_CACHE
    if _SEGMENTATION_CACHE is not None:
        return _SEGMENTATION_CACHE

    yolo_model = None
    sam_predictor = None
    
    # 1. 加载 YOLO
    try:
        from ultralytics import YOLO
        yolo_model = YOLO(YOLO_SEG_MODEL_PATH)
        print("✅ YOLO 分割模型加载成功!")
    except Exception as e:
        print(f"⚠️ YOLO 加载失败: {e}")
        print(f"   请检查路径: {YOLO_SEG_MODEL_PATH}")

    # 2. 加载 SAM
    try:
        from segment_anything import sam_model_registry, SamPredictor
        if os.path.exists(SAM_MODEL_PATH):
            sam_model = sam_model_registry["vit_h"](checkpoint=SAM_MODEL_PATH)
            sam_predictor = SamPredictor(sam_model)
            print("✅ SAM 模型加载成功!")
        else:
            print(f"⚠️ SAM 权重文件不存在: {SAM_MODEL_PATH}")
    except Exception as e:
        print(f"⚠️ SAM 加载失败: {e}")

    _SEGMENTATION_CACHE = {
        "yolo": yolo_model,
        "sam": sam_predictor,
        "ok": bool(yolo_model or sam_predictor),
    }
    return _SEGMENTATION_CACHE


def detect_dish_regions_with_yolo(image_path, yolo_model):
    """用 YOLO 检测盘子/碗区域，返回更稳的候选框列表。"""
    if yolo_model is None:
        return []

    try:
        results = yolo_model(str(image_path), conf=0.25, iou=0.45, verbose=False)
        boxes = []

        for r in results:
            if getattr(r, 'boxes', None) is None:
                continue
            for i in range(len(r.boxes)):
                conf = float(r.boxes.conf[i].item()) if hasattr(r.boxes.conf[i], 'item') else float(r.boxes.conf[i])
                cls_id = int(r.boxes.cls[i].item()) if hasattr(r.boxes.cls[i], 'item') else int(r.boxes.cls[i])
                if cls_id in PLATE_BOWL_CLASSES and conf >= 0.25:
                    x1, y1, x2, y2 = r.boxes.xyxy[i].cpu().numpy().tolist()
                    area = max(0.0, (x2 - x1) * (y2 - y1))
                    if area > 5000:
                        boxes.append((float(x1), float(y1), float(x2), float(y2)))

        print(f"   YOLO 检测到 {len(boxes)} 个盘子/碗候选区域")
        return sorted(boxes, key=lambda b: _box_area(b), reverse=True)
    except Exception as e:
        print(f"⚠️ YOLO 检测失败: {e}")
        return []


def refine_regions_with_sam(image, boxes, sam_predictor):
    """用 SAM 对 YOLO 候选框做精修，防止边框过大或过小。"""
    if sam_predictor is None or not boxes:
        return boxes

    try:
        import cv2
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                return boxes
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif hasattr(image, 'convert'):
            img = np.array(image.convert('RGB'))
        else:
            img = np.array(image)

        sam_predictor.set_image(img)
        refined_boxes = []
        for x1, y1, x2, y2 in boxes:
            pad_x = max(10, 0.12 * (x2 - x1))
            pad_y = max(10, 0.12 * (y2 - y1))
            box = np.array([
                max(0, x1 - pad_x),
                max(0, y1 - pad_y),
                min(img.shape[1], x2 + pad_x),
                min(img.shape[0], y2 + pad_y),
            ], dtype=np.float32)
            masks, scores, _ = sam_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box,
                multimask_output=True,
            )
            if len(masks) == 0:
                continue
            best_idx = int(np.argmax(scores)) if len(scores) > 0 else 0
            mask = masks[best_idx]
            ys, xs = np.where(mask > 0)
            if len(xs) < 10 or len(ys) < 10:
                continue
            bx1, by1 = int(xs.min()), int(ys.min())
            bx2, by2 = int(xs.max()) + 1, int(ys.max()) + 1
            area = (bx2 - bx1) * (by2 - by1)
            if area > 2000:
                refined_boxes.append((bx1, by1, bx2, by2))

        if refined_boxes:
            print(f"   SAM 精分割后得到 {len(refined_boxes)} 个更精确区域")
            return refined_boxes
        return boxes
    except Exception as e:
        print(f"⚠️ SAM 精分割失败: {e}")
        return boxes


def detect_dish_regions_in_image(image_path, max_dishes=4):
    """
    主入口：YOLO 检测 + SAM 精修 + OpenCV 兜底；
    只有当 YOLO/SAM/OpenCV 完全检测不到任何区域时，才使用网格兜底。
    """
    path = Path(image_path)
    try:
        pil_img = Image.open(path).convert('RGB')
        img = np.array(pil_img)
        h, w = img.shape[:2]
    except Exception:
        return [(0, 0, 1, 1)]

    if w <= 0 or h <= 0:
        return [(0, 0, 1, 1)]

    model_bundle = load_local_segmentation_models()
    yolo_model = model_bundle.get("yolo")
    sam_predictor = model_bundle.get("sam")

    # 第一步：YOLO 检测
    boxes = detect_dish_regions_with_yolo(image_path, yolo_model)
    
    # 第二步：SAM 精分割
    if boxes and sam_predictor is not None:
        boxes = refine_regions_with_sam(image_path, boxes, sam_predictor)

    # 第三步：如果 YOLO+SAM 都没检测到，尝试 OpenCV
    if not boxes:
        print("   🔄 YOLO+SAM 未检测到，尝试 OpenCV...")
        boxes = detect_regions_with_opencv(img, max_dishes)

    # 第四步：只有完全没检测到任何区域时，才使用网格切分
    if not boxes:
        print("   ⚠️ 未检测到任何盘子，使用网格切分兜底...")
        boxes = grid_split_fallback(img, max_dishes)

    # 过滤和去重
    boxes = _filter_candidate_boxes(boxes, img.shape, max_dishes=max_dishes)
    if len(boxes) > max_dishes:
        boxes = boxes[:max_dishes]
    
    if boxes:
        print(f"   ✅ 最终检测到 {len(boxes)} 个食物区域")
    return boxes


def detect_regions_with_opencv(img, max_dishes):
    """OpenCV 传统图像分割作为备用方案"""
    try:
        import cv2
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((5, 5), np.uint8)
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        h, w = img.shape[:2]
        min_area = 0.02 * w * h  # 至少占2%面积
        
        boxes = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area:
                continue
            x, y, ww, hh = cv2.boundingRect(c)
            if ww * hh < min_area:
                continue
            boxes.append((x, y, x + ww, y + hh))
        
        boxes = sorted(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
        return boxes[:max_dishes]
    except Exception:
        return []


def grid_split_fallback(img, max_dishes):
    """网格切分作为最后兜底"""
    h, w = img.shape[:2]
    if w * h > 300000:  # 大图才切分
        cols = min(2, max_dishes) if w > 300 else 1
        rows = min(2, max_dishes // cols + 1) if h > 300 else 1
        cells = []
        for r in range(rows):
            for c in range(cols):
                if len(cells) >= max_dishes:
                    break
                x1 = int(c * w / cols)
                y1 = int(r * h / rows)
                x2 = int((c + 1) * w / cols)
                y2 = int((r + 1) * h / rows)
                if x2 > x1 and y2 > y1:
                    cells.append((x1, y1, x2, y2))
        return cells
    return [(0, 0, w, h)]


def find_local_qwen_dir():
    if QWEN_VL_MODEL_PATH.exists():
        return QWEN_VL_MODEL_PATH

    candidates = [
        Path('E:/大作业/Qwen2-VL-2B-Instruct'),
        Path('E:/大作业/Qwen2.5-VL-3B-Instruct'),
        Path('E:/大作业/Qwen2.5-VL-7B-Instruct'),
        Path.home() / '.cache' / 'huggingface' / 'hub' / 'models--Qwen',
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def load_local_qwen_vl():
    global _QWEN_VL_CACHE
    if _QWEN_VL_CACHE is not None:
        return _QWEN_VL_CACHE

    try:
        import importlib.util
        if importlib.util.find_spec('transformers') is None:
            return None, None
        from transformers import AutoProcessor
    except Exception:
        return None, None

    qwen_dir = find_local_qwen_dir()
    if qwen_dir is None:
        return None, None

    try:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(str(qwen_dir), local_files_only=True, trust_remote_code=True)

        model = None
        for model_name in ['Qwen2VLForConditionalGeneration', 'Qwen2_5_VLForConditionalGeneration']:
            try:
                model_cls = __import__('transformers', fromlist=[model_name]).__dict__[model_name]
                model = model_cls.from_pretrained(
                    str(qwen_dir),
                    local_files_only=True,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                )
                break
            except Exception:
                continue

        if model is None:
            return None, None

        model.to(DEVICE)
        model.eval()
        print(f"✅ 本地 Qwen-VL 识别器加载成功: {qwen_dir}")
        _QWEN_VL_CACHE = (processor, model)
        return processor, model
    except Exception as e:
        print(f"⚠️ 本地 Qwen-VL 加载失败: {e}")
        return None, None


def find_local_chinese_clip_dir():
    candidates = [
        Path.home() / '.cache' / 'huggingface' / 'hub',
        Path('E:/大作业'),
        BASE_DIR,
    ]
    names = [
        'models--OFA-Sys--chinese-clip-vit-base-patch16',
        'models--uer--chinese-clip-vit-base-patch16',
        'ChineseCLIP',
        'chinese-clip-vit-base-patch16',
    ]
    for root in candidates:
        if not root.exists():
            continue
        for name in names:
            p = root / name
            if p.exists():
                return p
        for p in root.rglob('*chinese*clip*'):
            if p.is_dir():
                return p
    return None


def load_local_clip_zero_shot():
    global _CLIP_ZERO_SHOT_CACHE
    if _CLIP_ZERO_SHOT_CACHE is not None:
        return _CLIP_ZERO_SHOT_CACHE

    if CLIP_MODEL_PATH.exists():
        try:
            from transformers import CLIPModel, CLIPProcessor
            processor = CLIPProcessor.from_pretrained(str(CLIP_MODEL_PATH), local_files_only=True)
            model = CLIPModel.from_pretrained(str(CLIP_MODEL_PATH), local_files_only=True)
            model.to(DEVICE)
            model.eval()
            _CLIP_ZERO_SHOT_CACHE = (model, processor)
            return model, processor
        except Exception as e:
            print(f"⚠️ 本地 CLIP 路径加载失败，回退到 open_clip: {e}")

    chinese_clip_dir = find_local_chinese_clip_dir()
    if chinese_clip_dir is not None:
        try:
            from chinese_clip import ChineseCLIP
            model = ChineseCLIP.from_pretrained(str(chinese_clip_dir), local_files_only=True)
            model.to(DEVICE)
            model.eval()
            _CLIP_ZERO_SHOT_CACHE = (model, 'chinese_clip')
            return model, 'chinese_clip'
        except Exception as e:
            print(f"⚠️ 本地 ChineseCLIP 加载失败，继续回退到 open_clip: {e}")

    try:
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
        model.to(DEVICE)
        model.eval()
        _CLIP_ZERO_SHOT_CACHE = (model, preprocess)
        return model, preprocess
    except Exception:
        return None, None


def classify_with_clip_zero_shot(crop):
    model, backend = load_local_clip_zero_shot()
    if model is None or backend is None:
        return None
    try:
        labels = ['rice', 'vegetable', 'chicken', 'fish', 'tofu', 'noodles', 'tomato', 'apple', 'beef', 'pork', 'soup']

        if hasattr(model, 'encode_image') and hasattr(model, 'encode_text'):
            import open_clip
            preprocess = backend
            img = preprocess(crop).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                image_feat = model.encode_image(img)
                text = open_clip.tokenize(labels).to(DEVICE)
                text_feat = model.encode_text(text)
                image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)
                text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
                logits = (100.0 * image_feat @ text_feat.T).softmax(dim=-1)
                idx = int(logits.argmax().item())
                return labels[idx]

        processor = backend
        inputs = processor(text=labels, images=crop, return_tensors='pt', padding=True)
        inputs = {k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
        with torch.no_grad():
            image_feat = model.get_image_features(pixel_values=inputs['pixel_values'])
            text_feat = model.get_text_features(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
            image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
            logits = (100.0 * image_feat @ text_feat.T).softmax(dim=-1)
            idx = int(logits.argmax().item())
            return labels[idx]
    except Exception:
        return None


def local_qwen_classify(crop):
    processor, model = load_local_qwen_vl()
    if processor is None or model is None:
        return None

    try:
        prompt = 'Identify the main food shown in this dish region. Return only one label from: rice, vegetables, chicken, fish, tofu, noodles, tomato, apple, beef, pork, soup, unknown.'
        messages = [{
            'role': 'user',
            'content': [
                {'type': 'image', 'image': crop},
                {'type': 'text', 'text': prompt},
            ]
        }]
        text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=[text], images=[crop], return_tensors='pt', padding=True)
        inputs = {k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
        with torch.no_grad():
            gen_ids = model.generate(
                pixel_values=inputs.get('pixel_values'),
                input_ids=inputs.get('input_ids'),
                attention_mask=inputs.get('attention_mask'),
                max_new_tokens=32,
                do_sample=False,
            )
        out = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
        return clip_text_label(out)
    except Exception:
        return None


def classify_with_large_model(crop):
    # 大模型识别非常重，尤其在 CPU 上会长时间卡住。
    # 这里保留显式开关：若用户显式开启，则强制走本地 Qwen；否则保持轻量分支。
    if not USE_LARGE_MODEL_FOR_CLASSIFICATION:
        return classify_with_clip_zero_shot(crop)

    llm_label = local_qwen_classify(crop)
    if llm_label:
        return llm_label
    return classify_with_clip_zero_shot(crop)


def classify_region_to_food_name(image_crop, candidate_names=None):
    """大模型优先 + CLIP + 颜色规则兜底，增强识别稳定性。"""
    if candidate_names is None:
        candidate_names = ["米饭", "青菜", "鸡肉", "豆腐", "面条", "鱼", "牛肉", "番茄", "苹果", "白菜"]

    llm_label = classify_with_large_model(image_crop)
    if llm_label:
        return llm_label

    try:
        from transformers import AutoProcessor
    except Exception:
        pass

    arr = np.asarray(image_crop.convert('RGB'))
    if arr.size == 0:
        return candidate_names[0]

    flat = arr.reshape(-1, 3).astype(np.float32)
    if flat.size == 0:
        return candidate_names[0]

    mean = flat.mean(axis=0)
    r, g, b = float(mean[0]), float(mean[1]), float(mean[2])
    brightness = (r + g + b) / 3.0
    std = flat.std(axis=0).mean()

    green_like = np.mean((flat[:, 1] >= flat[:, 0]) & (flat[:, 1] >= flat[:, 2]) & (flat[:, 1] - flat.max(axis=1) > 10))
    red_like = np.mean((flat[:, 0] >= flat[:, 1]) & (flat[:, 0] >= flat[:, 2]) & (flat[:, 0] - flat.max(axis=1) > 15))
    white_like = np.mean((flat[:, 0] > 150) & (flat[:, 1] > 140) & (flat[:, 2] > 120))
    yellow_like = np.mean((flat[:, 0] > 130) & (flat[:, 1] > 120) & (flat[:, 2] < 110) & (flat[:, 0] - flat[:, 2] > 30))

    # 汤的专门规则：优先命中，避免和米饭/鸡肉混淆
    if brightness < 170 and std < 45 and (b > max(r, g) * 0.95 or (r < 150 and g < 150 and b < 170)):
        return "汤"
    if brightness < 155 and std < 40 and (r < 180 and g < 180 and b < 190):
        return "汤"

    if brightness > 150 and std < 50 and (white_like > 0.3 or yellow_like > 0.15):
        return "米饭"
    if white_like > 0.5 and brightness > 140:
        return "米饭"
    if green_like > 0.1 and g > 80:
        return "青菜"
    if g > max(r, b) and (g - max(r, b)) > 20:
        return "青菜"
    if red_like > 0.15 and r > 100 and r > g and r > b:
        return "番茄"
    if r > 80 and g > 50 and b < 150 and brightness < 180 and std > 15:
        return "鸡肉"
    if yellow_like > 0.2 and r > 140 and g > 120:
        return "炒饭"
    if b > r and b > g and b > 80:
        return "鱼"
    if brightness > 140:
        return "米饭"
    if g > b and g > r:
        return "青菜"
    return "鸡肉"


def process_single_image_nutrition(image_path, model, nutrition_db_index, max_dishes=4):
    """
    对一张图进行多盘菜识别、分别估算和汇总营养信息。
    使用 YOLO+SAM 精准分割每盘菜。
    """
    image = Image.open(image_path).convert('RGB')
    img_w, img_h = image.width, image.height
    
    print(f"\n🔍 处理图片: {Path(image_path).name}")
    print(f"   图像尺寸: {img_w} x {img_h}")
    
    # 1. 检测食物区域
    boxes = detect_dish_regions_in_image(image_path, max_dishes=max_dishes)
    
    # 2. 对每个区域进行预测
    results = []
    for idx, box in enumerate(boxes[:max_dishes], 1):
        x1, y1, x2, y2 = [int(v) for v in box]
        
        # 确保坐标在图像范围内
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w, x2), min(img_h, y2)
        
        if x2 <= x1 or y2 <= y1:
            continue
        
        crop = image.crop((x1, y1, x2, y2))
        crop_w, crop_h = crop.size
        area = crop_w * crop_h
        
        # 区域太小则跳过
        if area < 2000:
            continue
        
        print(f"\n   📍 区域 {idx}: {crop_w}x{crop_h}, 面积 {area} 像素")
        
        # 预测热量
        cal = estimate_total_calories_for_region(model, crop)
        print(f"      预测热量: {cal:.2f} kcal")
        
        # 食物分类
        predicted_name = classify_region_to_food_name(crop)
        print(f"      识别为: {predicted_name}")
        
        # 查询营养数据库
        item = compute_nutrition_from_calories(predicted_name, cal, nutrition_db_index)
        item['region_index'] = idx
        item['bbox'] = [x1, y1, x2, y2]
        item['area_pixels'] = area
        results.append(item)
    
    # 3. 汇总
    total_cal = round(sum(item['calories_kcal'] for item in results), 2)
    total_weight = round(sum(item['weight_g'] for item in results), 2)
    
    print(f"\n📊 汇总结果:")
    print(f"   共 {len(results)} 盘菜")
    print(f"   总热量: {total_cal:.2f} kcal")
    print(f"   总重量: {total_weight:.2f} g")
    
    return {
        'foods': results,
        'total_calories_kcal': total_cal,
        'total_weight_g': total_weight,
        'num_dishes': len(results)
    }


# ============================================================
# 9. 主入口
# ============================================================

def collect_image_paths(input_images=None, input_dir=None, max_images=None):
    targets = []

    if input_images:
        for item in input_images:
            p = Path(item)
            if p.exists() and p.is_file():
                targets.append(p)

    if not targets and input_dir:
        d = Path(input_dir)
        if d.exists() and d.is_dir():
            for pattern in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"):
                targets.extend(sorted(p for p in d.glob(pattern) if p.is_file()))

    if not targets:
        default_dir = Path("E:/大作业/cc_food_100/rgb")
        if default_dir.exists() and default_dir.is_dir():
            for pattern in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"):
                targets.extend(sorted(p for p in default_dir.glob(pattern) if p.is_file()))

    unique = []
    seen = set()
    for p in targets:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(p)

    if max_images is not None:
        unique = unique[:max_images]
    return unique


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多盘菜营养分析系统")
    parser.add_argument("--image", action="append", default=[], help="单张图片路径，可重复指定")
    parser.add_argument("--dir", default=None, help="图片目录，扫描该目录下的 JPG/PNG 等文件")
    parser.add_argument("--max-images", type=int, default=None, help="最多处理多少张图片")
    args = parser.parse_args()

    print("=" * 60)
    print("🍽️ 多盘菜营养分析系统 (YOLO+SAM 增强版)")
    print("=" * 60)

    print("\n[1] 加载 CalorieCLIP...")
    model = load_calorieclip(CALORIE_CLIP_PATH)

    print("\n[2] 加载营养数据库...")
    db_records, db_index = load_china_food_db(CHINA_FOOD_DB_ROOT)

    if model is not None:
        image_targets = collect_image_paths(
            input_images=args.image,
            input_dir=args.dir,
            max_images=args.max_images,
        )

        if image_targets:
            all_reports = []
            for idx, sample in enumerate(image_targets, 1):
                print(f"\n[3] 测试图片: {Path(sample).name} ({idx}/{len(image_targets)})")
                report = process_single_image_nutrition(
                    str(sample),
                    model,
                    db_index,
                    max_dishes=4,
                )
                all_reports.append(report)

            print("\n" + "=" * 60)
            print("📋 完整营养报告")
            print("=" * 60)
            print(json.dumps({
                "images": [str(p) for p in image_targets],
                "results": all_reports,
            }, ensure_ascii=False, indent=2))
        else:
            print("⚠️ 未找到测试图片")

    print("\n✅ 完成!")