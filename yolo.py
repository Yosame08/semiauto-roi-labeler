from ultralytics import YOLO

class Detector:
    def __init__(self):
        self.model = YOLO("yolo11x.pt")

    def run_yolo_detection(self, frame):
        results = self.model.predict(frame, max_det=10)
        annotated_img = results[0].plot(line_width=1)
        boxes = results[0].boxes.xywh.cpu().numpy()
        if len(boxes) > 0:
            print("Detected objects:")
            for i, box in enumerate(boxes):
                print(f"{i}: {box}")
        else:
            print("No objects detected.")
        return annotated_img, boxes