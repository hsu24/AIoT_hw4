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
