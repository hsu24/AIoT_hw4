import os
import cv2
import numpy as np
import joblib
from collections import defaultdict
import mediapipe as mp
from sklearn.metrics import classification_report

mp_hands = mp.solutions.hands

def extract_landmarks(img, hands):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    if not results.multi_hand_landmarks:
        return None
    
    hand_landmarks = results.multi_hand_landmarks[0]
    coords = []
    for lm in hand_landmarks.landmark:
        coords.append([lm.x, lm.y, lm.z])
    coords = np.array(coords)
    
    base_coord = coords[0]
    coords = coords - base_coord
    
    max_val = np.max(np.abs(coords))
    if max_val > 0:
        coords = coords / max_val
        
    return coords.flatten()

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    model_path = os.path.join(script_dir, 'rps_mp_model.pkl')
    test_dir = os.path.join(base_dir, 'dataset', 'test')

    if not os.path.exists(model_path):
        print(f"❌ 錯誤：找不到模型 '{model_path}'")
        return

    print("⏳ 載入模型中...")
    model_data = joblib.load(model_path)
    clf = model_data['model']
    CLASS_NAMES = model_data['class_names']
    print(f"✅ 模型載入成功！類別：{CLASS_NAMES}")

    y_true = []
    y_pred = []
    total = 0
    correct = 0
    skipped = 0
    per_class_total = defaultdict(int)
    per_class_correct = defaultdict(int)
    confusion = defaultdict(lambda: defaultdict(int))

    print("🔍 正在逐張測試圖片...")
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        for category in ['rock', 'paper', 'scissors']:
            category_path = os.path.join(test_dir, category)
            if not os.path.exists(category_path):
                continue

            for filename in os.listdir(category_path):
                if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue

                img_path = os.path.join(category_path, filename)
                img = cv2.imread(img_path)
                if img is None:
                    continue

                features = extract_landmarks(img, hands)
                if features is None:
                    skipped += 1
                    continue

                p_idx = clf.predict([features])[0]
                pred_name = CLASS_NAMES[p_idx]

                y_true.append(category)
                y_pred.append(pred_name)
                
                total += 1
                per_class_total[category] += 1
                confusion[category][pred_name] += 1

                if pred_name == category:
                    correct += 1
                    per_class_correct[category] += 1

    print("\n" + "=" * 55)
    print("📊 MediaPipe 模型測試結果")
    print("=" * 55)

    if total > 0:
        accuracy = correct / total * 100
        print(f"\n🎯 整體準確率: {accuracy:.2f}% ({correct}/{total})")
        print(f"⚠️ 未偵測到手部跳過張數: {skipped}\n")

        # 詳細分類報告
        print("📝 分類詳細報告 (Precision, Recall, F1-score):")
        print(classification_report(y_true, y_pred, target_names=['paper', 'rock', 'scissors']))

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
