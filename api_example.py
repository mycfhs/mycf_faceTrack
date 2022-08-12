import cv2
from MyFaceTrack import MyFaceTrack

# tracker = MyFaceTrack('yolo', 'sort')
# tracker.trackMulti(0)
tracker = MyFaceTrack('yolo', 'sort')
tracker.set_video(0)

while True:
    ret, frame = tracker.getImg()   # 由于各模型需要的图片形式不同，建议使用这个

    img = tracker.track_api(frame)
    cv2.imshow("Tracking", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break