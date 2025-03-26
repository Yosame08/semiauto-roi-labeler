import sys
from annotator import VideoAnnotator

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <video_path> or <annotation_file>")
        exit(1)
    annotator = VideoAnnotator(sys.argv[1])
    while True:
        if annotator.run(): break