import os
import subprocess

def print_manual(frame, framerate):
    os.system("clear")
    print(f"第{frame}帧（{frame/framerate:.3f}s）")
    print("左键拖动：手动框选RoI")
    print("Y(yolo)：在该帧运行YOLO物体检测")
    print("T(track)：开始跟踪（需该帧已选RoI）")
    print("S(save)：保存标注工程")
    print("E(export)：导出标注结果")
    print("Q(quit)：直接退出")
    print("方向键（↔）：切换前后的帧")
    print("方向键（↕）：切换至前后一秒")
    print("-：切换至前一关键帧（I帧）")
    print("+：切换至后一关键帧（I帧）")

def xywh_to_int(x, y, w, h):
    x2 = int(round(x + w))
    y2 = int(round(y + h))
    x, y = int(round(x)), int(round(y))
    h, w = y2 - y, x2 - x
    return x, y, w, h

def get_keyframes(video_path):
    """获取关键帧的帧号列表"""
    cmd = f"ffprobe -loglevel error -select_streams v -show_entries frame=pict_type -of csv=p=0 {video_path}"
    output = subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout
    print(output)
    keyframes = [i for i, typ in enumerate(output.split()) if typ == "I"]
    return keyframes

def binary_next_id(val_list, val, prev=False):
    left, right = 0, len(val_list) - 1
    while left < right:
        mid = (left + right) // 2
        if val_list[mid] == val:
            return max(mid - 1, 0) if prev else min(mid + 1, len(val_list) - 1)
        elif val_list[mid] < val:
            left = mid + 1
        else:
            right = mid
    return min(right - 1, 0) if prev else max(left, len(val_list) - 1)