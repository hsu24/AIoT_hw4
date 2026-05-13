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
     - 自動整理 `validation/` 為 YOLO 格式的 `val/{class}/` 子資料夾
     - 使用預訓練 `yolov8n-cls.pt`，epochs=50, imgsz=224, batch=32, CPU 訓練
     - 訓練後自動複製最佳模型到 `demo/rps_yolo_model.pt`
  2. **`demo/test_yolo.py`** [NEW]：YOLO 測試腳本，含逐類別準確率與混淆矩陣
  3. **`demo/rps_camera.py`** [MODIFIED]：支援 YOLO + SVM 雙模型
     - 優先載入 YOLO 模型，找不到則退回 SVM
     - YOLO 推論附帶信心分數顯示在 UI
     - 模型類型標籤顯示在畫面底部
  4. **`requirements.txt`** [MODIFIED]：新增 `ultralytics`, `torch`, `torchvision`
- **訓練結果**：
  - 🎯 測試集 Top-1 準確率：**94.89%**
  - 🎯 測試集 Top-5 準確率：**100.00%**
  - 模型大小：**2.8 MB**（vs SVM 的 27 MB）

## 2026-05-13 10:49 — 系統指示紀錄
- **使用者要求**：將接下來的聊天紀錄都整理並存到 `conversation_log.md`。
- **系統回應**：已確認，將會在每次任務完成後，將對話與任務執行結果記錄到此檔案中。

## 2026-05-13 10:53 — 修改 test_yolo.py 支援樹莓派舊版 Python
- **使用者需求**：需要在樹莓派上執行 `test_yolo.py`，但樹莓派可能不支援較新的 Python 版本（如 f-string 語法等），要求修改腳本以確保相容性。
- **修改內容**：
  - 將所有 f-string (`f"..."`) 改寫為更具相容性的 `.format()` 語法。
  - 移除了程式碼中的 Emoji（圖示）輸出，避免在部分未預設使用 UTF-8 編碼的終端機環境下發生 `UnicodeEncodeError` 錯誤。
- **產出結果**：已成功修改並更新 `RSP_demo-main/demo/test_yolo.py`。

## 2026-05-13 10:59 — 嘗試推送至 GitHub
- **使用者需求**：將所有檔案 push 到 `https://github.com/hsu24/AIoT_hw4.git`。
- **執行狀況**：系統偵測到環境中未安裝 Git 或未將 Git 加入環境變數，因此無法自動執行推送。
- **後續建議**：已提供手動上傳或安裝 Git 的指令說明給使用者參考。

## 2026-05-13 11:02 — 再次嘗試推送至 GitHub
- **使用者需求**：使用者已安裝好 Git，要求再次嘗試推送到 `https://github.com/hsu24/AIoT_hw4.git`。
- **執行狀況**：已設定暫時的 commit 名稱並嘗試執行 `git pull --allow-unrelated-histories` 及 `git push` 指令，目前正在等待使用者完成 GitHub 的授權登入（Credential Manager 彈出視窗）。
