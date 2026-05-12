# 剪刀石頭布辨識系統 — 執行說明

## 環境安裝

```bash
pip install -r demo/requirements.txt
```

---

## 攝影機即時辨識 — `rps_camera.py`

開啟攝影機，即時辨識手勢為剪刀、石頭或布。

```bash
cd demo
python rps_camera.py
```

**操作方式：**

| 按鍵 | 功能 |
|------|------|
| `空白鍵` | 擷取畫面並辨識手勢 |
| `Q` | 離開程式 |

**使用步驟：**
1. 執行程式後，程式會自動搜尋攝影機並載入模型
2. 畫面中央會出現綠色 ROI 框
3. 將手放入框中，按下空白鍵進行辨識
4. 辨識結果與信心分數會顯示在畫面底部
5. 按 Q 離開程式

---

## 測試模型準確率 — `test.py`

使用測試集圖片評估模型準確率。

### YOLO 模型測試

```bash
cd demo
python test_yolo.py
```

輸出整體準確率、各類別準確率與混淆矩陣。

### SVM 模型測試

```bash
cd demo
python test.py
```

輸出整體準確率與分類報告（precision / recall / F1-score）。
