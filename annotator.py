import datetime
import json
import time
import cv2
from pathlib import Path
from yolo import Detector
from utils import print_manual, xywh_to_int, get_keyframes, binary_next_id

timeslot = 10
window_name = "Frame"
last_autosave = time.time()

class VideoAnnotator:
    def __init__(self, file_path):
        # 如果file_path是json文件，则加载标注工程
        if file_path.endswith(".json"):
            # 读取json文件
            with open(file_path, 'r') as f:
                project = json.load(f)
            self.video_path = project["video_path"]
        else:
            self.video_path = file_path

        self.cap = cv2.VideoCapture(self.video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.framerate = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"视频信息：{self.total_frames}帧，{self.frame_w}x{self.frame_h}， {self.framerate}fps")
        self.keyframes = get_keyframes(self.video_path)
        self.detector = Detector()

        self.drawing = False
        self.ix, self.iy = -1, -1
        self.temp_roi = None

        if file_path.endswith(".json"):
            self.current_frame = project["current_frame"]
            self.annotations = {
                int(k): v for k, v in project["annotations"].items()
            }
        else:
            self.annotations = {}  # 存储标注结果 {frame_num: [x,y,w,h]}
            self.current_frame = 0

    def run(self) -> bool:
        global last_autosave
        ret, frame = self._get_frame(self.current_frame)
        if ret:
            print_manual(self.current_frame, self.framerate)
            print(f"该视频的关键帧：{self.keyframes}")
            cv2.namedWindow(window_name)
            cv2.setMouseCallback(window_name, self.mouse_callback)
            if self.current_frame in self.annotations:
                print(f"当前帧RoI: {self.annotations[self.current_frame]}")
            while True:
                if time.time() - last_autosave > 60:
                    self.save_annotation("autosave")
                    last_autosave = time.time()
                    print(f"Auto saved. {str(datetime.datetime.now())}")
                display = frame.copy()
                x = y = w = h = -1
                if self.drawing and self.temp_roi:
                    x, y, w, h = self.temp_roi
                    color = (255, 0, 0)
                elif self.current_frame in self.annotations:
                    x, y, w, h = self.annotations[self.current_frame]
                    color = (0, 255, 0)
                if x != -1:
                    cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)

                cv2.imshow(window_name, display)

                if self.drawing:
                    _ = cv2.waitKey(timeslot)
                else:
                    key = cv2.waitKey(timeslot) & 0xFF
                    if key != 0xFF:
                        return self.handle_keyboard(key, frame)
        else:
            self.current_frame = self.total_frames - 1
            return False

    def handle_keyboard(self, key, frame):
        if key == ord('y'):  # 运行YOLO检测
            self.handle_yolo_result(frame)
        elif key == ord('t'):  # 开始跟踪
            while True:
                track_to = input("从当前帧跟踪到（输入0跟踪到失败或最后一帧）：")
                try:
                    self.auto_track(int(track_to))
                    break
                except ValueError:
                    print("输入错误，请重新输入")
        elif key == ord('s'):  # 保存标注工程
            self.user_save()
        elif key == ord('e'):  # 导出标注结果
            self.export_annotation()
        elif key == ord('q'):
            return True
        elif key == 81:  # 左箭头
            self.current_frame = max(0, self.current_frame - 1)
        elif key == 83:  # 右箭头
            self.current_frame = min(self.total_frames - 1, self.current_frame + 1)
        elif key == 82:  # 上箭头
            self.current_frame = max(0, self.current_frame - self.framerate)
        elif key == 84:  # 下箭头
            self.current_frame = min(self.total_frames - 1, self.current_frame + self.framerate)
        elif len(self.keyframes) > 0 and key == ord('-'):
            self.current_frame = self.keyframes[binary_next_id(self.keyframes, self.current_frame, True)]
        elif len(self.keyframes) > 0 and key == ord('+'):
            self.current_frame = self.keyframes[binary_next_id(self.keyframes, self.current_frame, False)]
        return False

    def handle_yolo_result(self, frame):
        print("Yolo推理中...")
        cv2.setMouseCallback("Frame", lambda *args: None)
        annotated_frame, boxes = self.detector.run_yolo_detection(frame)
        # 根据boxes标注每个box的id
        for i in range(len(boxes)):
            # yolo center x,y 转为左上角坐标
            x, y = boxes[i][:2] - boxes[i][2:] / 2
            w, h = boxes[i][2:]
            x, y, w, h = xywh_to_int(x, y, w, h)
            boxes[i] = (x, y, w, h)
            print(f"[{i}取整后：{x, y, w, h}]")
            cv2.putText(annotated_frame, str(i), (int(round(x)), int(round(y))+12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        yolo_window_name = f"yolo - frame {self.current_frame}"
        cv2.imshow(yolo_window_name, annotated_frame)
        print("输入方框对应的序号作为RoI，按q取消")
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break
            else:
                idx = chr(key)
                if not idx.isdigit():
                    print("输入错误，请重新输入")
                    continue
                idx = int(idx)
                if idx < len(boxes):
                    x, y, w, h = boxes[idx]
                    x, y, w, h = int(x), int(y), int(w), int(h)
                    self.annotations[self.current_frame] = (x, y, w, h)
                    print(f"选择了idx={idx}的RoI: {(x, y, w, h)}")
                    break
                else:
                    print("输入错误，请重新输入")
        cv2.destroyWindow(yolo_window_name)

    def auto_track(self, track_to):
        tracker = cv2.TrackerCSRT_create()
        ret, frame = self._get_frame(self.current_frame)
        x, y, w, h = self.annotations[self.current_frame]
        tracker.init(frame, (x, y, w, h))
        current = self.current_frame + 1
        if track_to == 0:
            track_to = self.total_frames
        while current < min(track_to, self.total_frames):
            ret, frame = self._get_frame(current)
            if not ret:break
            success, box = tracker.update(frame)
            if success:
                x, y, w, h = [int(v) for v in box]
                print(f"\rTracked frame {current}: {x, y, w, h}")
                self.annotations[current] = (x, y, w, h)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.imshow(window_name, frame)
                cv2.waitKey(1)
                current += 1
            else:
                print(f"\rTracking frame {current} failed")
                break
        self.current_frame = current

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"手动框选RoI: {x, y}")
            self.drawing = True
            self.ix, self.iy = x, y
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            dx = x - self.ix
            dy = y - self.iy
            self.temp_roi = (self.ix, self.iy, dx, dy)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if self.temp_roi is not None:
                x, y, w, h = self.temp_roi
                if x + w > self.frame_w:
                    w = self.frame_w - x
                if x + w < 0:
                    w = -x
                if y + h > self.frame_h:
                    h = self.frame_h - y
                if y + h < 0:
                    h = -y
                if w < 0:
                    x += w
                    w = -w
                if h < 0:
                    y += h
                    h = -h
                self.temp_roi = (x, y, w, h)
                self.annotations[self.current_frame] = self.temp_roi
                self.temp_roi = None
                print(f"手动框选了RoI: {self.annotations[self.current_frame]}")
            else:
                print("未框选RoI")

    def _get_frame(self, frame_num):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        return self.cap.read()

    def export_annotation(self):
        print("文件结构：")
        print("selected_folder")
        print("├── images")
        print("│   ├── frame_000000.jpg")
        print("│   ├── frame_000001.jpg")
        print("│   └── ...")
        print("└── labels")
        print("    ├── frame_000000.txt")
        print("    ├── frame_000001.txt")
        print("    └── ...")
        save_dir = Path(input("请输入保存路径（覆盖已有文件）："))
        (save_dir / "images").mkdir(parents=True, exist_ok=True)
        (save_dir / "labels").mkdir(parents=True, exist_ok=True)
        frame_interval = int(input("抽帧频率（输入0保存所有帧，若有未标注帧则顺延）："))
        for frame_num in range(0, self.total_frames, frame_interval):
            while frame_num not in self.annotations and frame_num < self.total_frames:
                frame_num += 1
            if frame_num >= self.total_frames: break
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = self.cap.read()
            if not ret: continue

            # 保存图像
            img_path = save_dir / "images" / f"frame_{frame_num:06d}.jpg"
            cv2.imwrite(str(img_path), frame)

            # 生成标签
            label_path = save_dir / "labels" / f"frame_{frame_num:06d}.txt"
            if frame_num in self.annotations:
                x, y, w, h = self.annotations[frame_num]
                img_h, img_w = frame.shape[:2]

                # 转换为YOLO格式
                x_center = (x + w / 2) / img_w
                y_center = (y + h / 2) / img_h
                width = w / img_w
                height = h / img_h

                with open(label_path, 'w') as f:
                    f.write(f"0 {x_center:.8f} {y_center:.8f} {width:.8f} {height:.8f}\n")

    def user_save(self):
        filename = input("请输入文件名（覆盖已有文件，json文件）：")
        self.save_annotation(filename)

    def save_annotation(self, filename):
        if not filename.endswith(".json"):
            filename += ".json"
        save_path = Path(filename)
        annotation = {
            "video_path": self.video_path,
            "current_frame": self.current_frame,
            "annotations": self.annotations,
        }
        # 保存
        with open(save_path, 'w') as f:
            f.write(json.dumps(annotation))