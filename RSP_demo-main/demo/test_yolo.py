"""
YOLOv8 分類模型測試腳本（ONNX Runtime 版本，不需要 ultralytics）
在測試集上評估 YOLO 模型的準確度

使用方式：
  cd RSP_demo-main/demo
  python test_yolo.py

所需套件：pip install onnxruntime opencv-python numpy
"""

import os
import sys
import cv2
import numpy as np
from collections import defaultdict

try:
    import onnxruntime as ort
except ImportError:
    print("❌ 錯誤：onnxruntime 套件未安裝。")
    print("   請執行以下指令安裝：")
    print("   pip install onnxruntime")
    sys.exit(1)


# YOLO 分類模型的類別名稱（與訓練時的資料夾順序一致）
CLASS_NAMES = {0: 'paper', 1: 'rock', 2: 'scissors'}


def preprocess(img, imgsz=224):
    """將圖片前處理為 ONNX 模型輸入格式 (1, 3, 224, 224)"""
    # 縮放
    resized = cv2.resize(img, (imgsz, imgsz))
    # BGR → RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    # 正規化 [0, 255] → [0.0, 1.0]
    normalized = rgb.astype(np.float32) / 255.0
    # HWC → CHW
    chw = np.transpose(normalized, (2, 0, 1))
    # 加 batch 維度 → (1, 3, 224, 224)
    return np.expand_dims(chw, axis=0)


def softmax(x):
    """計算 softmax 機率"""
    e = np.exp(x - np.max(x))
    return e / e.sum()


def predict(session, img):
    """使用 ONNX 模型推論，回傳 (類別名稱, 信心分數)"""
    input_tensor = preprocess(img)
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: input_tensor})[0][0]
    probs = softmax(output)
    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])
    return CLASS_NAMES.get(pred_idx, str(pred_idx)), confidence


def main():
    # 設定路徑
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    model_path = os.path.join(script_dir, 'rps_yolo_model.onnx')
    test_dir = os.path.join(base_dir, 'dataset', 'test')

    # 檢查模型
    if not os.path.exists(model_path):
        print("❌ 錯誤：找不到 ONNX 模型 '{}'".format(model_path))
        print("   請先執行 train/train_yolo.py 訓練並匯出模型。")
        return

    if not os.path.exists(test_dir):
        print("❌ 錯誤：找不到測試資料集 '{}'".format(test_dir))
        return

    # 載入 ONNX 模型
    print("⏳ 載入 ONNX 模型中...")
    session = ort.InferenceSession(model_path)
    print("✅ 模型載入成功！")
    print("   模型: {}".format(os.path.basename(model_path)))
    print("   類別: {}\n".format(CLASS_NAMES))

    # 統計
    total = 0
    correct = 0
    per_class_total = defaultdict(int)
    per_class_correct = defaultdict(int)
    confusion = defaultdict(lambda: defaultdict(int))

    print("🔍 正在逐張測試圖片...")
    for category in ['rock', 'paper', 'scissors']:
        category_path = os.path.join(test_dir, category)

        if not os.path.exists(category_path):
            subdirs = [d for d in os.listdir(test_dir)
                       if os.path.isdir(os.path.join(test_dir, d))]
            if subdirs:
                category_path = os.path.join(test_dir, subdirs[0], category)

        if not os.path.exists(category_path):
            print("  ⚠️ 找不到 {} 的資料夾，略過...".format(category))
            continue

        for filename in os.listdir(category_path):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            img_path = os.path.join(category_path, filename)
            img = cv2.imread(img_path)
            if img is None:
                continue

            pred_name, confidence = predict(session, img)

            total += 1
            per_class_total[category] += 1
            confusion[category][pred_name] += 1

            if pred_name == category:
                correct += 1
                per_class_correct[category] += 1

    # ========== 結果報告 ==========
    print("\n" + "=" * 55)
    print("📊 YOLO 模型測試結果（ONNX Runtime）")
    print("=" * 55)

    if total > 0:
        accuracy = correct / total * 100
        print("\n🎯 整體準確率: {:.2f}% ({}/{})\n".format(accuracy, correct, total))

        # 各類別準確率
        print("{:<12} {:>8} {:>12}".format('類別', '準確率', '正確/總數'))
        print("-" * 35)
        for cat in ['rock', 'paper', 'scissors']:
            cat_total = per_class_total[cat]
            cat_correct = per_class_correct[cat]
            cat_acc = cat_correct / cat_total * 100 if cat_total > 0 else 0
            print("{:<12} {:>7.2f}% {:>5}/{:<5}".format(cat, cat_acc, cat_correct, cat_total))

        # 混淆矩陣
        print("\n📋 混淆矩陣 (列=真實, 欄=預測)")
        cats = ['rock', 'paper', 'scissors']
        header = "{:>12}".format('') + "".join("{:>10}".format(c) for c in cats)
        print(header)
        print("-" * (12 + 10 * len(cats)))
        for true_cat in cats:
            row = "{:>12}".format(true_cat)
            for pred_cat in cats:
                row += "{:>10}".format(confusion[true_cat][pred_cat])
            print(row)
    else:
        print("❌ 沒有成功測試任何圖片。")


if __name__ == "__main__":
    main()
