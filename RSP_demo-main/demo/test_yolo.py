"""
YOLOv8 分類模型測試腳本
在測試集上評估 YOLO 模型的準確度，並與 SVM 做比較

使用方式：
  cd RSP_demo-main/demo
  python test_yolo.py
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict


def main():
    # 設定路徑
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    model_path = os.path.join(script_dir, 'rps_yolo_model.pt')
    test_dir = os.path.join(base_dir, 'dataset', 'test')

    # 檢查模型
    if not os.path.exists(model_path):
        print(f"❌ 錯誤：找不到 YOLO 模型 '{model_path}'")
        print("   請先執行 train/train_yolo.py 訓練模型。")
        return

    if not os.path.exists(test_dir):
        print(f"❌ 錯誤：找不到測試資料集 '{test_dir}'")
        return

    # 載入模型
    print("⏳ 載入 YOLO 模型中...")
    model = YOLO(model_path)
    print("✅ 模型載入成功！\n")

    # 取得模型的類別名稱
    class_names = model.names  # {0: 'paper', 1: 'rock', 2: 'scissors'} 等
    print(f"📋 模型類別: {class_names}\n")

    # 定義本地測試資料夾的類別對應
    label_map = {'rock': 'rock', 'paper': 'paper', 'scissors': 'scissors'}

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
            # 嘗試子資料夾
            subdirs = [d for d in os.listdir(test_dir)
                       if os.path.isdir(os.path.join(test_dir, d))]
            if subdirs:
                category_path = os.path.join(test_dir, subdirs[0], category)

        if not os.path.exists(category_path):
            print(f"  ⚠️ 找不到 {category} 的資料夾，略過...")
            continue

        for filename in os.listdir(category_path):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            img_path = os.path.join(category_path, filename)
            results = model(img_path, verbose=False)

            if results and len(results) > 0:
                result = results[0]
                pred_class_idx = result.probs.top1
                pred_class_name = class_names[pred_class_idx]
                confidence = result.probs.top1conf.item()

                total += 1
                per_class_total[category] += 1
                confusion[category][pred_class_name] += 1

                if pred_class_name == category:
                    correct += 1
                    per_class_correct[category] += 1

    # ========== 結果報告 ==========
    print("\n" + "=" * 55)
    print("📊 YOLO 模型測試結果")
    print("=" * 55)

    if total > 0:
        accuracy = correct / total * 100
        print(f"\n🎯 整體準確率: {accuracy:.2f}% ({correct}/{total})\n")

        # 各類別準確率
        print(f"{'類別':<12} {'準確率':>8} {'正確/總數':>12}")
        print("-" * 35)
        for cat in ['rock', 'paper', 'scissors']:
            cat_total = per_class_total[cat]
            cat_correct = per_class_correct[cat]
            cat_acc = cat_correct / cat_total * 100 if cat_total > 0 else 0
            emoji = {'rock': '🪨', 'paper': '📄', 'scissors': '✂️'}[cat]
            print(f"{emoji} {cat:<8} {cat_acc:>7.2f}% {cat_correct:>5}/{cat_total:<5}")

        # 混淆矩陣
        print(f"\n📋 混淆矩陣 (列=真實, 欄=預測)")
        cats = ['rock', 'paper', 'scissors']
        header = f"{'':>12}" + "".join(f"{c:>10}" for c in cats)
        print(header)
        print("-" * (12 + 10 * len(cats)))
        for true_cat in cats:
            row = f"{true_cat:>12}"
            for pred_cat in cats:
                row += f"{confusion[true_cat][pred_cat]:>10}"
            print(row)
    else:
        print("❌ 沒有成功測試任何圖片。")


if __name__ == "__main__":
    main()
