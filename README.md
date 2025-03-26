# RoI半自动标注器

`Usage: python main.py <video_path> or <annotation_file>`

本脚本基于[Yolo v11](https://github.com/aji-li/ultralytics-v11)物体识别和CSRT跟踪器实现。您可以选择Yolo识别的物体作为RoI，或者手动框选RoI，然后使用跟踪器自动跟踪RoI。标注的结果将存储为Yolo v11的数据集形式。

在使用前，请前往[Yolo v11](https://github.com/aji-li/ultralytics-v11)的github页面下载物体识别的预训练模型，并在`yolo.py`中修改对应文件名。

- 运行本脚本需要系统安装`ffmpeg`，若未安装，可将`anotator.py`的30行改为`self.keyframes = []`。这种情况下无法快捷跳转前后I帧。
- 本脚本使用了opencv的GUI，请在有GUI的环境下运行。
- 要求输入单个字符（主菜单、Yolo识别结果）时，请将在GUI位于前端时键入，不要键入到终端。
- 其他情况下（输入帧数、保存文件），请在终端正常输入，回车确认。