# Conversation Log

## 2026-05-12 10:30 — 初始化
- 使用者要求建立 `conversation_log.md` 來記錄對話過程。

## 2026-05-12 10:40 — 結合 test.py 與 carema.py
- **使用者需求**：將 `demo` 資料夾中的 `test.py` 和 `carema.py` 結合，建立一個可以即時辨識剪刀、石頭、布的 Python 程式。
- **分析結果**：
  - `test.py`：載入 SVM 模型 (`rps_svm_model.pkl`)，對圖片做灰階→64x64 縮放→攤平→正規化的前處理，再進行預測。
  - `carema.py`：開啟攝影機顯示即時畫面。
  - `train_svm.py`：使用 SVM (RBF kernel) 訓練模型，前處理流程與 test.py 一致。
- **產出檔案**：`demo/rps_camera.py`
  - 功能：開啟攝影機即時顯示畫面，按空白鍵擷取 ROI 區域並辨識手勢（石頭/布/剪刀）
  - 使用與訓練時一致的前處理流程
  - 含 UI 框線提示、辨識結果顯示、操作說明

## 2026-05-12 10:57 — 修復攝影機無法開啟問題
- **使用者回報**：`rps_camera.py` 無法透過 index 開啟攝影機。
- **問題原因**：`cv2.VideoCapture(0)` 在 Windows 上使用預設 backend（MSMF）時，經常無法以 index 開啟攝影機。
- **修復方式**：新增 `open_camera()` 函式，依序嘗試多種 backend 與 index 組合：
  1. **DirectShow (`CAP_DSHOW`)**：Windows 上最穩定的 backend，index 0–4
  2. **Media Foundation (`CAP_MSMF`)**：index 0–4
  3. **自動偵測 (`CAP_ANY`)**：index 0–4
- **額外改進**：
  - 每次嘗試不僅檢查 `isOpened()`，還會實際讀取一幀確認攝影機可用
  - Console 輸出每次嘗試的結果，方便除錯
  - 若全部失敗，提示使用者檢查裝置管理員中的攝影機驅動

## 2026-05-12 11:08 — 訓練 YOLOv8 分類模型提升準確度
- **使用者需求**：新訓練一個 YOLO 模型取代 SVM，提升剪刀石頭布辨識準確度。
- **確認事項**：無 GPU，使用最輕量 `yolov8n-cls` 模型。
- **資料集**：train 2520 張 / test 372 張 / validation 33 張（已重新整理成 YOLO 分類格式）
- **新增/修改檔案**：
  1. **`train/train_yolo.py`** [NEW]：YOLO 分類模型訓練腳本
  2. **`demo/test_yolo.py`** [NEW]：YOLO 測試腳本
  3. **`demo/rps_camera.py`** [MODIFIED]：支援 YOLO + SVM 雙模型
  4. **`requirements.txt`** [MODIFIED]：新增 `ultralytics`, `torch`, `torchvision`
- **訓練結果**：
  - 🎯 測試集 Top-1 準確率：**94.89%**
  - 🎯 測試集 Top-5 準確率：**100.00%**
  - 模型大小：**2.8 MB**

## 2026-05-13 10:49 — 系統指示紀錄
- **使用者要求**：將接下來的聊天紀錄都整理並存到 `conversation_log.md`。
- **系統回應**：已確認，將會在每次任務完成後，將對話與任務執行結果記錄到此檔案中。

## 2026-05-13 10:53 — 修改 test_yolo.py 支援樹莓派舊版 Python
- **使用者需求**：需要在樹莓派上執行 `test_yolo.py`，要求修改腳本以確保相容性。
- **修改內容**：將 f-string 改寫為 `.format()`，並移除 Emoji 避免編碼錯誤。

## 2026-05-13 10:59 — 嘗試推送至 GitHub
- **使用者需求**：將所有檔案 push 到 GitHub。
- **執行狀況**：偵測到環境中未安裝 Git，提供安裝建議。

## 2026-05-13 11:02 — 再次嘗試推送至 GitHub
- **使用者需求**：已安裝 Git，再次嘗試推送。
- **執行狀況**：執行 `git pull` 與 `git push`，過程中處理了合併衝突。

## 2026-05-13 11:15 — 再次解決 Git 衝突與推送
- **執行狀況**：解決 `conversation_log.md` 衝突，成功 push 到 GitHub。

## 2026-05-13 11:34 — 復原 Ultralytics 版本與新模型推薦
- **使用者需求**：將 `test_yolo.py` 與 `rps_camera.py` 改回使用 `ultralytics` 套件。
- **執行狀況**：已復原檔案，並提供 MobileNet, ResNet, MediaPipe 等模型推薦。

## 2026-05-13 12:06 — 實作 MediaPipe 手勢辨識模型
- **使用者需求**：訓練並整合 MediaPipe 手勢辨識模型。
- **執行狀況**：
  - 新增 `train/train_mediapipe.py`：擷取 21 個手部關鍵點特徵並訓練隨機森林模型。
  - 新增 `demo/test_mediapipe.py`：測試新模型準確度。
  - 修改 `demo/rps_camera.py`：支援 MediaPipe 骨架顯示與 `m` 鍵即時切換模型功能。
