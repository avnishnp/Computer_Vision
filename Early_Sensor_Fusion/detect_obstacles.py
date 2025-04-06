import cv2
import numpy as np
from ultralytics import YOLO  # PyTorch-based YOLO implementation

class YoloOD:
    def __init__(self, tiny_model=True):
        self.tiny_model = tiny_model

        # Load the YOLOv8 model (tiny or normal)
        if self.tiny_model:
            self.model = YOLO("yolov8n.pt")  # YOLOv8n (Nano model, smaller and faster)
        else:
            self.model = YOLO("yolov8s.pt")  # YOLOv8s (Small model, more accurate)

    def run_obstacle_detection(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Run YOLOv8 prediction
        results = self.model.predict(img_rgb)

        # Extracting predictions
        pred_bboxes = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                cls = int(box.cls[0])
                pred_bboxes.append([x1, y1, x2, y2, conf, cls])
                label = f"{self.model.names[cls]} {conf:.2f}"
                
                # Drawing bounding boxes
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return img, pred_bboxes
