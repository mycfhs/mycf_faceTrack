# cv2的人脸检测方法
# 详见harrcascade文件夹中的各项文件
CASCADE_FILE_PATH = "haarcascade/haarcascade_frontalface_default.xml"

# cv2的人脸追踪方法
CV2_TRACKER = 'TrackerMIL'
# Choose:[TrackerMIL, TrackerDaSiamRPN, TrackerGOTURN] 后两个需要自行下载模型。个人感觉性能一般。
# TrackerGOTURN: https://github.com/spmallick/goturn-files 然后提取第一个即可，
# TrackerDaSiamRPN:

# 是否抽帧 >=1  1则正常
CHOUZHEN_NUM = 1

# 锚框置信度
THRESHOLD = 0.8

# for sort model
MAX_AGE = 100
MIN_HITS = 1
FACE_MINISIZE = 40  # 最小的人脸图片尺寸

# for align model
align_THRESHOLD = [0.6, 0.7, 0.7]
align_FACTOR = 0.709

# 是否改变输入图像尺寸
# yolo模型对尺寸的要求似乎是2**n
VIDEO_RESIZE = True
RESIZE_WIDTH = 512
RESIZE_HEIGTH = 512

# for yolo model
YOLO_WEIGHT_PATH = 'yolov5/weights/v5l.pt'
YOLO_DEVICE = '0'  # 'n' or 'cpu
YOLO_HALF_MODEL = False  # if true, fp32 -> fp16. ONLY FOR CUDA!!

YOLO_PRUNING = 0    # 剪枝 float 0->1   0不剪 1全剪
