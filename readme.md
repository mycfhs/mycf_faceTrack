### Introduction

Face track mainly by yolov5 and sort.

We do it for a small comptation in our school.

### Installation

conda create -n faceTrack python=3.7

conda activate faceTrack

pip install -r requirements.txt

Install torch at https://pytorch.org/

If use yolov5 function:
pip install -r requirements_yolo.txt

### model

haarcascade:

链接：https://pan.baidu.com/s/1qNbRf8ShCIZc8d-ttA129Q 
提取码：vc7j 

yolov5 model:

链接：https://pan.baidu.com/s/1a8FoyRb93gd-YChpqoxycg 
提取码：amw0



## ERROR MAY OCCUR！！！

AttributeError: 'Upsample' object has no attribute 'recompute_scale_factor'

method:  报错的倒数第二个
  File "D:\Anaconda\envs\YOUR_ENV_NAME\lib\site-packages\torch\nn\modules\upsampling.py", line 154, in forward
    recompute_scale_factor=self.recompute_scale_factor)
   
  remove 'recompute_scale_factor=xxx'.

### Instruction

#### Run

python MyFaceTrack.py

提供了单目标和多目标两种追踪方法，只需在程序中选择人脸提取方法即可。

python OurUI.py

带有UI界面的程序，使用便捷，可直接选择检测方法以及待检测视频流。

#### Note

我们在api_example.py提供了外调本功能的api使用方法。

We have two track method(cv2 and sort) and three face extract method(cv2, align and yolo).

We DON'T support yolo with cv2 combination.

####  track method:

cv2(not recommend): based on MIL tracker in opencv. Also it has other ways to track,
you can change 'CV2_TRACKER' in config.py to choose one. You need to get model by yourself.

sort(recommend): based on KalmanTracker

#### extrack method

cv2: Through opencv. All options are in haarcascade folder.

align: Better than cv2.

yolo: Better than yolo.
