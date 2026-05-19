# AIoT_hw4 剪刀石頭布即時辨識系統 (Raspberry Pi 4)

這是一個專為 Raspberry Pi 4 等邊緣運算裝置設計的剪刀石頭布即時辨識系統。本專案不僅實作了基礎的手勢辨識功能，更透過多種模型架構的比較與最佳化（ONNX Runtime 加速），達成兼具高準確率與高執行效能的邊緣 AI 應用。

## 🌟 專案特色

1. **多模型即時切換**：整合了三種不同的 AI 模型（SVM、YOLOv8、MediaPipe + Random Forest），可在攝影機即時推論中按下 `m` 鍵無縫切換。
2. **邊緣裝置最佳化**：將所有模型皆轉換為 **ONNX 格式**，並使用 `onnxruntime` 推論，大幅降低資源消耗，非常適合 Raspberry Pi 4。
3. **高準確率表現**：YOLO 與 MediaPipe 模型皆能達到 94% 以上的測試準確率。

---

## 🚀 環境安裝

### 1. 刷機與系統設定 (Raspberry Pi 4)

1. 下載 [Raspberry Pi Imager](https://www.raspberrypi.com/software/)
2. 選擇 **Raspberry Pi 4 64-bit** 作業系統。
3. 插上 MicroSD 讀卡機。
4. 輸入主機名、設定時區為 `Asia/Taipei`。
5. 設定用戶名及密碼，並**開啟 SSH 功能**。
6. 完成寫入後，插入樹莓派並開機。

### 2. 安裝 Python 依賴套件

請在終端機中執行以下指令安裝所需套件：

```bash
# 安裝 ONNX Runtime、MediaPipe 與 OpenCV 等推論所需套件
pip install onnxruntime mediapipe opencv-python numpy

# (選擇性) 若需重新訓練模型或轉換 ONNX，需額外安裝：
pip install ultralytics torch torchvision scikit-learn joblib skl2onnx
```

---

## 🎮 運行方式

### 即時攝影機辨識 — `rps_camera.py`

開啟攝影機，即時辨識手勢為剪刀、石頭或布。系統將自動使用 ONNX 引擎進行高速推論。

```bash
cd RSP_demo-main/demo
python rps_camera.py
```

**操作說明：**

| 按鍵 | 功能 |
|------|------|
| `空白鍵` | 擷取畫面並辨識手勢 |
| `M` 鍵 | 切換使用的模型 (MediaPipe → YOLO → SVM) |
| `Q` 鍵 | 離開程式 |

**執行流程：**
1. 程式會自動搜尋可用的攝影機與 ONNX 模型。
2. 畫面中央將顯示綠色 ROI 框（MediaPipe 模式會額外顯示手部骨架節點）。
3. 將手放入框中，按 `空白鍵` 進行辨識，結果會顯示在畫面下方。

---

## 📂 專案結構

```
AIoT_hw4/
├── RSP_demo-main/
│   ├── demo/
│   │   ├── rps_camera.py       # 攝影機即時辨識程式 (主程式)
│   │   ├── test.py             # SVM 模型測試
│   │   ├── test_yolo.py        # YOLO 模型測試
│   │   ├── test_mediapipe.py   # MediaPipe 模型測試
│   │   ├── rps_*_model.onnx    # 轉換後的 ONNX 模型權重 (推論用)
│   │   ├── rps_svm_model.pkl   # SVM 原始模型權重
│   │   └── rps_yolo_model.pt   # YOLO 原始模型權重
│   ├── train/
│   │   ├── train_svm.py        # SVM 訓練腳本
│   │   ├── train_yolo.py       # YOLO 訓練腳本
│   │   ├── train_mediapipe.py  # MediaPipe (Random Forest) 訓練腳本
│   │   └── convert_to_onnx.py  # 將模型轉換為 ONNX 格式的腳本
│   └── README.md               # 原始 README
├── conversation_log.md         # 專案開發對話紀錄
├── log.md                      # 模型成效分析與對話紀錄
├── report.md                   # 模型成效比較報告
└── README.md                   # 本份整合說明文件
```

---

## 📊 HW4 報告與評分標準回應

本專案符合 AIoT_hw4 的各項要求，以下為報告詳細說明：

### 1. 成功在 Raspberry Pi 4 上執行手勢辨識 (50%)
本系統利用 `onnxruntime` 大幅降低了執行運算負擔，能夠在 Raspberry Pi 4 上流暢執行攝影機的即時擷取與推論。

### 2. Demo 展示影片 (carema) (15%)
影片展示使用 `rps_camera.py` 進行 15 個手勢（剪刀、石頭、布各 5 次）的辨識，並展示使用 `M` 鍵動態切換不同模型的推論結果。

### 3. 模型架構修改與成效分析 (35%)

為了改善基礎模型的效能，我們額外導入了 **YOLO** 與 **MediaPipe** 兩種架構，並評估了三個模型的效能指標：

#### 📈 模型成效比較表

| 模型架構 | Accuracy (準確率) | Precision (精確率) | Recall (召回率) | F1-Score | 備註 / 測試數量 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **原模型 (SVM)** | 68.28% | 0.69 | 0.68 | 0.68 | 測試 372 張圖片 |
| **模型一 (MediaPipe)**| 94.31% | 0.95 | 0.94 | 0.94 | 測試 369 張圖片 (有 3 張未偵測到手部跳過) |
| **模型二 (YOLO)** | **94.62%** | **0.95** | **0.95** | **0.95** | 測試 372 張圖片，準確度最高 |

#### 💡 更換模型原因及比較差異

1. **原模型 (SVM) 效能不足**：
   原本的模型使用影像灰階化、縮放至 64x64 後攤平的方式直接輸入 SVM。這種作法保留了太多背景雜訊，且失去了二維空間的結構特徵。其準確率僅約 68.28%，在實際互動應用中會造成極差的使用者體驗。

2. **導入模型一 (MediaPipe + Random Forest)**：
   - **原因**：為了解決背景雜訊問題並提升執行速度，我們改用 Google 的 MediaPipe 擷取手部的 21 個 3D 關鍵節點。
   - **差異與優勢**：我們以這 63 個座標特徵 (21×3) 作為輸入來訓練 Random Forest 模型。此架構將準確率大幅提升至 **94.31%**，且由於特徵維度極低（僅 63 維），推論速度非常快，非常適合資源受限的 Raspberry Pi 4。此外，只要 MediaPipe 抓得到手骨架，就不會受到複雜背景干擾。

3. **導入模型二 (YOLOv8-cls)**：
   - **原因**：為了探索使用深度學習卷積神經網路 (CNN) 處理此任務的潛力。
   - **差異與優勢**：YOLOv8-cls 達到了最高的準確率 **94.62%**。它直接輸入 224x224 的彩色影像，模型能自動學習手勢的紋理與邊緣特徵。相比於 SVM，其特徵擷取能力遠遠超出；而相較於 MediaPipe，即使手部部分被遮擋或 MediaPipe 無法精準抓取骨架時，CNN 依然有機會做出正確判斷。唯一的缺點是 CNN 的計算量較大。

**總結最佳化策略**：
為了讓 YOLO 等複雜模型也能在樹莓派上順暢運行，我們開發了 `convert_to_onnx.py` 將所有模型（SVM, MediaPipe Random Forest, YOLO）全數轉換為 `.onnx` 格式，讓它們能共享 `onnxruntime` 的底層 C++ 加速優勢，最終實現了即時順暢的剪刀石頭布辨識系統！
