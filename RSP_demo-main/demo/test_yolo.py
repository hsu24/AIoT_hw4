import os
import sys
import cv2
from collections import defaultdict

try:
    from ultralytics import YOLO
    from sklearn.metrics import classification_report
except ImportError:
    print("❌ 錯誤：缺少必要套件。")
    print("   請執行以下指令安裝：")
    print("   pip install ultralytics scikit-learn")
    sys.exit(1)


# YOLO 分類模型的類別名稱（與訓練時的資料夾順序一致）
CLASS_NAMES = {0: 'paper', 1: 'rock', 2: 'scissors'}


def main():
    # 設定路徑
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    model_path = os.path.join(script_dir, 'rps_yolo_model.pt')
    test_dir = os.path.join(base_dir, 'dataset', 'test')

    # 檢查模型
    if not os.path.exists(model_path):
        print("❌ 錯誤：找不到 YOLO 模型 '{}'".format(model_path))
        print("   請先執行 train/train_yolo.py 訓練模型。")
        return

    if not os.path.exists(test_dir):
        print("❌ 錯誤：找不到測試資料集 '{}'".format(test_dir))
        return

    # 載入 YOLO 模型
    print("⏳ 載入 YOLO 模型中...")
    model = YOLO(model_path)
    print("✅ 模型載入成功！")
    print("   模型: {}".format(os.path.basename(model_path)))
    
    # 取得模型的類別對應
    model_names = model.names
    print("   類別: {}\n".format(model_names))

    # 統計
    y_true = []
    y_pred = []
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
            
            # 使用 YOLO 進行預測
            results = model.predict(img_path, verbose=False)
            if len(results) == 0:
                continue
                
            # 分類任務的第一個預測結果
            result = results[0]
            p_idx = int(result.probs.top1)
            pred_name = model_names[p_idx]

            y_true.append(category)
            y_pred.append(pred_name)
            
            total += 1
            per_class_total[category] += 1
            confusion[category][pred_name] += 1

            if pred_name == category:
                correct += 1
                per_class_correct[category] += 1

    # ========== 結果報告 ==========
    print("\n" + "=" * 55)
    print("📊 YOLO 模型測試結果 (Ultralytics)")
    print("=" * 55)

    if total > 0:
        accuracy = correct / total * 100
        print("\n🎯 整體準確率: {:.2f}% ({}/{})\n".format(accuracy, correct, total))

        # 詳細分類報告 (Precision, Recall, F1)
        print("📝 分類詳細報告 (Precision, Recall, F1-score):")
        target_names = ['paper', 'rock', 'scissors']
        print(classification_report(y_true, y_pred, target_names=target_names))

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

