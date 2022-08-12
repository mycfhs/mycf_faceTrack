import cv2
import numpy as np
import os
from config import *
import tensorflow.compat.v1 as tf
from functools import partial
from yolov5.utils.torch_utils import prune
import sys
sys.path.append(os.getcwd()+r'\\yolov5')


class MyFaceTrack(object):
    def __init__(self, extractorName='align', trackerName='sort'):
        self.name = 'MyFaceTracker'

        self.faceExtractorMethod = extractorName
        self.trackMethod = trackerName
        self.videoFile = None
        self.colours = np.random.rand(32, 3)
        self.frame = None

        self.project_dir = os.path.dirname(os.path.abspath(__file__))

        self.yolo_init_done = False

        self.api_init_done = False
        self.api_tracker = None
        self.api_extractor = None
        self.useAPI = False
        self.imgW = None
        self.imgH = None

    def trackSingle(self, videoPath=0):
        """
        单目标追踪
        :return:
        """
        self.trackMethod = 'cv2'
        self.videoFile = cv2.VideoCapture(videoPath)

        tracker = self._getTracker()
        while True:
            for _ in range(CHOUZHEN_NUM):
                ret, frame = self.getImg()

            if not ret:
                break
            timer = cv2.getTickCount()
            ret, bbox = tracker.update(frame)
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

            self._drawBoundingbox(bbox)
            self._drawInfo(fps)
            cv2.imshow("Tracking", self.frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def trackMulti(self, videoPath=0):
        """
        多目标追踪
        :param trackerName:
        :return:
        """
        self.trackMethod = 'sort'
        self.videoFile = cv2.VideoCapture(videoPath)

        extractor = self.getFaceExtractor()
        tracker = self._getTracker()

        ret, frame = self.getImg()

        while True:
            timer = cv2.getTickCount()
            for _ in range(CHOUZHEN_NUM):
                ret, frame = self.getImg()
            if not ret or frame is None:
                break

            faces = extractor(frame)
            if len(faces) > 0:
                face_list = []
                for i, item in enumerate(faces):
                    if len(item)==4 and self.faceExtractorMethod == 'cv2':
                        # [x1, y1, w, h] 转为 [x1, y1, x2, y2]
                        item[2] = item[0] + item[2]
                        item[3] = item[1] + item[3]
                        face_list.append(item)
                    elif self.faceExtractorMethod =='align':
                        if round(item[4], 3) > THRESHOLD:
                            face_list.append(item)
                    elif self.faceExtractorMethod =='yolo':
                        if round(item[4], 3) > THRESHOLD:
                            face_list.append(item)

                final_faces = np.array(face_list)
                trackers = tracker.update(final_faces)

                for bbox in trackers:
                    bbox[2] = bbox[2] - bbox[0]
                    bbox[3] = bbox[3] - bbox[1]
                    self._drawBoundingbox(bbox)

            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

            self._drawInfo(fps)
            cv2.imshow("Tracking", self.frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def track_api(self, rawImg):
        """
        便于调用。 输入图片 输出带框的图片
        :param rawImg:
        :return:
        """
        timer = cv2.getTickCount()
        if not self.api_init_done:
            self._api_init()
        if self.trackMethod=='cv2':
            ret, bbox = self.api_tracker.update(rawImg)
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

            self._drawBoundingbox(bbox)
            self._drawInfo(fps)
            return self.frame

        elif self.trackMethod=='sort':
            faces = self.api_extractor(rawImg)
            if len(faces) > 0:
                face_list = []
                for i, item in enumerate(faces):
                    if len(item) == 4 and self.faceExtractorMethod == 'cv2':
                        # [x1, x2, w, h] 转为 [x1, y1, x2, y2]
                        item[2] = item[0] + item[2]
                        item[3] = item[1] + item[3]
                        face_list.append(item)
                    elif self.faceExtractorMethod == 'align':
                        if round(item[4], 3) > THRESHOLD:
                            face_list.append(item)
                    elif self.faceExtractorMethod == 'yolo':
                        if round(item[4], 3) > THRESHOLD:
                            face_list.append(item)

                final_faces = np.array(face_list)
                trackers = self.api_tracker.update(final_faces)

                for bbox in trackers:
                    bbox[2] = bbox[2] - bbox[0]
                    bbox[3] = bbox[3] - bbox[1]
                    self._drawBoundingbox(bbox)

            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
            self._drawInfo(fps)
            return self.frame

    def _api_init(self):
        if self.trackMethod!='cv2':
            self.api_extractor = self.getFaceExtractor()
        self.api_tracker = self._getTracker()
        self.api_init_done = True
        return None

    def set_video(self, videoPath):
        """
        设置需要追踪的视频。 为了api临时加上的
        :param videoPath:
        :return:
        """
        self.useAPI = True
        self.videoFile = cv2.VideoCapture(videoPath)

    def getImg(self):
        """
        获取视频的每一帧图片
        :return: 是否成功， 图片
        """
        ret, frame = self.videoFile.read()
        if VIDEO_RESIZE:
            frame = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGTH))

        if ret:
            self.imgH, self.imgW, _ = frame.shape
        self.frame = frame  # self.frame is used to display final result while frame to predict bbox

        if self.faceExtractorMethod=='cv2':
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif self.faceExtractorMethod=='align':
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        elif self.faceExtractorMethod=='yolo' and (self.yolo_init_done or self.useAPI):
            from yolov5.utils.augmentations import letterbox
            import torch
            if self.useAPI and (not self.api_init_done):
                self._yolo_init()
            stride = int(self.yolo_model.stride.max())
            img = letterbox(frame, (self.imgH, self.imgW), stride)[0]
            # Convert
            img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if YOLO_HALF_MODEL else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            frame = img
        return ret, frame

    def getFaceExtractor(self):
        """
        获取基于深度学习的人脸提取器
        :param faceExtractorName:
        :return:
        """
        if self.faceExtractorMethod == 'cv2':
            return self._getCascadeFaceExtractor()
        elif self.faceExtractorMethod == 'align':
            return self._getAlignFaceExtractor()
        elif self.faceExtractorMethod == 'yolo':
            return self._getYoloFaceExtractor()
        else:
            raise AssertionError('%s not in [''align'', ''cv2'', ''yolo'']' % self.faceExtractorMethod)

    def _getCascadeFaceExtractor(self):
        """
        获取基于haarCascade的人脸提取器
        :return:
        """

        try:
            faceCascade = cv2.CascadeClassifier(CASCADE_FILE_PATH)
        except:
            raise FileExistsError('Can''t find %s' % CASCADE_FILE_PATH)

        return partial(faceCascade.detectMultiScale,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(FACE_MINISIZE, FACE_MINISIZE),
            flags=cv2.CASCADE_SCALE_IMAGE)

    def _getAlignFaceExtractor(self):
        import align.detect_face as detect_face

        tf.Graph().as_default()
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True),
                                                     log_device_placement=False))
        pnet, rnet, onet = detect_face.create_mtcnn(self.sess, os.path.join(self.project_dir, "align"))
        return partial(detect_face.detect_face, minsize=FACE_MINISIZE, pnet=pnet, rnet=rnet, onet=onet,
                       threshold=align_THRESHOLD, factor=align_FACTOR)

    def _getYoloFaceExtractor(self):
        from yolov5.Dec import yoloDect

        if not self.yolo_init_done:
            self._yolo_init()

        return partial(yoloDect, model=self.yolo_model, conf=THRESHOLD)

    def _yolo_init(self):
        from yolov5.models.experimental import attempt_load
        from yolov5.utils.torch_utils import select_device

        self.device = select_device(YOLO_DEVICE)
        self.yolo_model = attempt_load(YOLO_WEIGHT_PATH, map_location=self.device)  # load FP32 model
        prune(self.yolo_model, YOLO_PRUNING)
        if YOLO_HALF_MODEL:
            self.yolo_model.half()
        self.yolo_init_done = True

    def _getTracker(self):
        """
        获取追踪器
        :param trackerName:
        :return:
        """
        if self.trackMethod == 'cv2':
            return self._getCv2Tracker()
        elif self.trackMethod == 'sort':
            from src.sort import Sort
            return Sort(MAX_AGE, MIN_HITS)
        elif self.trackMethod == 'mySort':
            pass
        else:
            raise AssertionError('%s not in [''cv2'', ''sort'']' % self.trackMethod)

    def _getCv2Tracker(self):
        """
        获取基于OpenCV的追踪器
        :return:
        """
        _, frame = self.getImg()
        tracker = eval(str('cv2.%s_create()'%CV2_TRACKER))
        extractor = self.getFaceExtractor()

        faces = extractor(frame)
        while len(faces) == 0:
            _, frame = self.getImg()
            faces = extractor(frame)
        if self.faceExtractorMethod == 'align':
            faces = faces[0][:4]
            faces[2], faces[3] = faces[2] - faces[0], faces[3]-faces[1]
            faces = [faces]
        tracker.init(frame, faces[0].astype(int))
        return tracker

    def _drawBoundingbox(self, bbox):
        """
        绘制锚框
        :param bbox:
        :return:
        """
        bbox = list(map(int, bbox))
        if len(bbox) == 4:
            index = 0
        else:
            index = int(bbox[4])
        cv2.rectangle(self.frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                      self.colours[index % 32, :] * 255, 2, 1)
        cv2.putText(self.frame, 'ID : %d  DETECT' % (index), (bbox[0], bbox[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    self.colours[index % 32, :] * 255, 2)

    def _drawInfo(self, fps):
        """
        绘制帧率等信息
        :param fps:
        :return:
        """
        cv2.putText(self.frame, self.faceExtractorMethod + " Face Extracteor", (60, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50),
                    2)
        cv2.putText(self.frame, self.trackMethod + " Tracker", (60, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50),
                    2)
        cv2.putText(self.frame, "FPS:" + str(round(fps, 2)), (60, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)


if __name__ == '__main__':
    tracker = MyFaceTrack('yolo', 'sort')
    tracker.trackMulti(0)
    # tracker.trackSingle(0)

