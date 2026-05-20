import cv2
import numpy as np
import onnxruntime as ort

class YOLOv8ONNX:
    def __init__(self, model_path, conf_thres=0.5, iou_thres=0.45):
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        
        # 載入 ONNX 模型
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        
        # 取得模型的輸入形狀 (例如 1x3x640x640 或 1x3x416x416)
        model_inputs = self.session.get_inputs()
        self.input_name = model_inputs[0].name
        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]
        
        # 分類名稱
        self.classes = ['drowsy_face', 'normal_face']
        # 定義顏色
        self.colors = [(0, 0, 255), (0, 255, 0)] # BGR (紅, 綠)

    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114)):
        # 保持比例縮放圖片，並用 color 填補周圍
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        
        return img, r, (dw, dh)

    def preprocess(self, img):
        # 縮放與 Padding
        img, ratio, (dw, dh) = self.letterbox(img, new_shape=(self.input_height, self.input_width))
        
        # BGR 轉 RGB，HWC 轉 CHW
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        
        # 正規化 0-255 -> 0.0-1.0
        img = img.astype(np.float32) / 255.0
        # 增加 batch 維度
        img = np.expand_dims(img, axis=0)
        
        return img, ratio, (dw, dh)

    def infer(self, img):
        img_tensor, ratio, pad = self.preprocess(img)
        outputs = self.session.run(None, {self.input_name: img_tensor})
        return self.postprocess(outputs, img.shape, ratio, pad)

    def postprocess(self, outputs, orig_shape, ratio, pad):
        # YOLOv8 輸出形狀通常為 (1, 4+num_classes, 8400)
        preds = outputs[0][0] # 變成 (6, 8400) 假設 2 個類別
        preds = preds.T # 轉置成 (8400, 6)
        
        boxes = []
        scores = []
        class_ids = []
        
        for pred in preds:
            # 取得各分類機率
            classes_scores = pred[4:]
            class_id = np.argmax(classes_scores)
            score = classes_scores[class_id]
            
            if score > self.conf_thres:
                # 框的座標 (cx, cy, w, h)
                cx, cy, w, h = pred[0:4]
                
                # 轉換為 (x_min, y_min, w, h) 以符合 OpenCV NMS 要求
                x_min = cx - w / 2
                y_min = cy - h / 2
                
                boxes.append([x_min, y_min, w, h])
                scores.append(float(score))
                class_ids.append(class_id)
                
        # 執行 NMS
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_thres, self.iou_thres)
        
        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                box = boxes[i]
                x_min, y_min, w, h = box[0], box[1], box[2], box[3]
                
                # 還原 Padding
                x_min -= pad[0]
                y_min -= pad[1]
                
                # 還原縮放比例
                x_min /= ratio
                y_min /= ratio
                w /= ratio
                h /= ratio
                
                # 轉換為絕對座標
                x1 = int(round(x_min))
                y1 = int(round(y_min))
                x2 = int(round(x_min + w))
                y2 = int(round(y_min + h))
                
                # 邊界限制
                x1 = max(0, min(x1, orig_shape[1]))
                y1 = max(0, min(y1, orig_shape[0]))
                x2 = max(0, min(x2, orig_shape[1]))
                y2 = max(0, min(y2, orig_shape[0]))
                
                results.append({
                    'box': (x1, y1, x2, y2),
                    'score': scores[i],
                    'class_id': class_ids[i],
                    'class_name': self.classes[class_ids[i]]
                })
                
        return results

    def draw_detections(self, img, results):
        out_img = img.copy()
        for res in results:
            x1, y1, x2, y2 = res['box']
            cls_id = res['class_id']
            score = res['score']
            name = res['class_name']
            color = self.colors[cls_id]
            
            # 畫框
            cv2.rectangle(out_img, (x1, y1), (x2, y2), color, 2)
            
            # 畫標籤
            label = f"{name} {score:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(out_img, (x1, y1 - 20), (x1 + w, y1), color, -1)
            cv2.putText(out_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
        return out_img
