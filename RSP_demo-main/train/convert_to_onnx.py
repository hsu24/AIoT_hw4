"""
將 SVM 與 MediaPipe (Random Forest) 模型轉換為 ONNX 格式
需安裝：pip install onnxruntime skl2onnx joblib
"""

import os
import joblib
import numpy as np

try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
except ImportError:
    print("❌ 錯誤：缺少 skl2onnx 套件。")
    print("   請執行: pip install skl2onnx onnxruntime")
    exit(1)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    demo_dir = os.path.join(os.path.dirname(script_dir), 'demo')

    # ==========================
    # 1. 轉換 SVM 模型
    # ==========================
    svm_pkl = os.path.join(demo_dir, 'rps_svm_model.pkl')
    svm_onnx = os.path.join(demo_dir, 'rps_svm_model.onnx')
    
    if os.path.exists(svm_pkl):
        print(f"⏳ 正在轉換 SVM 模型 ({svm_pkl})...")
        try:
            svm_model = joblib.load(svm_pkl)
            # SVM 預設的特徵維度是 64x64 = 4096
            initial_type_svm = [('float_input', FloatTensorType([None, 4096]))]
            # 轉換模型
            onx_svm = convert_sklearn(svm_model, initial_types=initial_type_svm,
                                      options={id(svm_model): {'zipmap': False}})
            
            with open(svm_onnx, "wb") as f:
                f.write(onx_svm.SerializeToString())
            print(f"✅ SVM 模型轉換成功：{svm_onnx}")
        except Exception as e:
            print(f"❌ SVM 轉換失敗：{e}")
    else:
        print(f"⚠️ 找不到 SVM 模型 ({svm_pkl})，略過轉換。")

    # ==========================
    # 2. 轉換 MediaPipe (Random Forest) 模型
    # ==========================
    mp_pkl = os.path.join(demo_dir, 'rps_mp_model.pkl')
    mp_onnx = os.path.join(demo_dir, 'rps_mp_model.onnx')
    
    if os.path.exists(mp_pkl):
        print(f"\n⏳ 正在轉換 MediaPipe 模型 ({mp_pkl})...")
        try:
            mp_data = joblib.load(mp_pkl)
            rf_model = mp_data['model']
            # 特徵維度是 21 個關節點 * 3 (x,y,z) = 63
            initial_type_mp = [('float_input', FloatTensorType([None, 63]))]
            # 轉換模型 (關閉 zipmap 可以讓輸出直接是 array，不用處理 dictionary，效能更好且對 C++ 推論更友善)
            onx_mp = convert_sklearn(rf_model, initial_types=initial_type_mp, 
                                     options={id(rf_model): {'zipmap': False}})
            
            with open(mp_onnx, "wb") as f:
                f.write(onx_mp.SerializeToString())
            print(f"✅ MediaPipe 模型轉換成功：{mp_onnx}")
        except Exception as e:
            print(f"❌ MediaPipe 轉換失敗：{e}")
    else:
        print(f"⚠️ 找不到 MediaPipe 模型 ({mp_pkl})，略過轉換。")

    print("\n🎉 轉換腳本執行完畢！您可以繼續測試 onnxruntime 推論了。")

if __name__ == "__main__":
    main()
