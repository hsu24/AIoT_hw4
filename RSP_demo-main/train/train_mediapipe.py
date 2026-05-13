"""
MediaPipe + Random Forest 模型訓練腳本
使用 MediaPipe 擷取手部 21 個關節點的三維座標，並訓練隨機森林分類器。

使用方式：
  cd RSP_demo-main
  python train/train_mediapipe.py

所需套件：pip install mediapipe scikit-learn opencv-python numpy joblib
"""

import os
import cv2
import numpy as np
import joblib
import mediapipe as mp
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# MediaPipe Hands 模組
mp_hands = mp.solutions.hands

# 類別對應
CLASS_NAMES = ['paper', 'rock', 'scissors']
LABEL_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}

def extract_landmarks(img, hands):
    """從影像中擷取並正規化 21 個手部關節點 (x, y, z)"""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    if not results.multi_hand_landmarks:
        return None
    
    # 只取第一隻偵測到的手
    hand_landmarks = results.multi_hand_landmarks[0]
    
    # 取出 21 個點的 (x, y, z)
    coords = []
    for lm in hand_landmarks.landmark:
        coords.append([lm.x, lm.y, lm.z])
    coords = np.array(coords)
    
    # 正規化：以手腕 (點 0) 為原點平移，使座標具備平移不變性
    base_coord = coords[0]
    coords = coords - base_coord
    
    # 正規化：縮放，使最大絕對值為 1，具備縮放不變性
    max_val = np.max(np.abs(coords))
    if max_val > 0:
        coords = coords / max_val
        
    return coords.flatten()

def load_dataset(dataset_dir, hands):
    """讀取資料集並擷取特徵"""
    X = []
    y = []
    
    print(f"📂 正在從 {dataset_dir} 擷取特徵...")
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"  ⚠️ 找不到資料夾 {class_name}，略過...")
            continue
            
        count = 0
        skip_count = 0
        for filename in os.listdir(class_dir):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            img_path = os.path.join(class_dir, filename)
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            features = extract_landmarks(img, hands)
            if features is not None:
                X.append(features)
                y.append(LABEL_MAP[class_name])
                count += 1
            else:
                skip_count += 1
                
        print(f"  ✅ {class_name}: 成功擷取 {count} 張, 找不到手部略過 {skip_count} 張")
        
    return np.array(X), np.array(y)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    dataset_base = os.path.join(base_dir, 'dataset')
    train_dir = os.path.join(dataset_base, 'train')
    val_dir = os.path.join(dataset_base, 'val')
    model_output = os.path.join(base_dir, 'demo', 'rps_mp_model.pkl')
    
    if not os.path.exists(train_dir):
        print(f"❌ 錯誤：找不到訓練資料夾 {train_dir}")
        return

    # 初始化 MediaPipe Hands
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        # 讀取訓練集
        X_train, y_train = load_dataset(train_dir, hands)
        
        # 讀取驗證集 (如果有的話)
        if os.path.exists(val_dir):
            X_val, y_val = load_dataset(val_dir, hands)
        else:
            X_val, y_val = None, None
            
    if len(X_train) == 0:
        print("❌ 錯誤：未擷取到任何有效的訓練特徵。")
        return
        
    print(f"\n📊 訓練集資料量: {len(X_train)} 筆, 特徵維度: {X_train.shape[1]}")
    
    # 訓練隨機森林分類器
    print("⏳ 開始訓練 Random Forest 模型...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    print("✅ 訓練完成！")
    
    # 訓練集準確度
    train_pred = clf.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)
    print(f"🎯 訓練集準確度: {train_acc * 100:.2f}%")
    
    # 驗證集準確度
    if X_val is not None and len(X_val) > 0:
        val_pred = clf.predict(X_val)
        val_acc = accuracy_score(y_val, val_pred)
        print(f"🎯 驗證集準確度: {val_acc * 100:.2f}%")
        
    # 儲存模型
    print(f"\n💾 正在儲存模型至: {model_output}")
    # 同時儲存模型和 CLASS_NAMES 以便後續推論對應
    model_data = {
        'model': clf,
        'class_names': CLASS_NAMES
    }
    joblib.dump(model_data, model_output)
    print("✅ 儲存成功！可以使用 demo/rps_camera.py 測試囉！")

if __name__ == "__main__":
    main()
