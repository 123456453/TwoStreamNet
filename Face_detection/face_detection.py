import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
def face_lip_clip(video_path:str,
         model_path:str):
  '''使用mediapipe进行脸部定位与嘴唇裁剪'''
  BaseOptions = mp.tasks.BaseOptions
  FaceDetector = mp.tasks.vision.FaceDetector
  FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
  VisionRunningMode = mp.tasks.vision.RunningMode
  options = FaceDetectorOptions(
      base_options = BaseOptions(model_asset_path = model_path),
      running_mode = VisionRunningMode.VIDEO
  )
  frames = []
  with FaceDetector.create_from_options(options) as detector:
    cap = cv2.VideoCapture(video_path)
    for i in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
      ret, frame = cap.read()
      mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = np.asarray(frame, dtype=np.uint8))
      face_detector_result = detector.detect_for_video(image=mp_image,timestamp_ms=i+1)
      for detection in face_detector_result.detections:
        box = detection.bounding_box
        origin_x, origin_y = box.origin_x,box.origin_y
        width, height = box.width, box.height
        endpoint_x, endpoint_y = origin_x + width, origin_y + height
        frame = frame[origin_y:endpoint_y,origin_x:endpoint_x,:]
        size = (140,140)
        frame = cv2.resize(frame,size,interpolation=cv2.INTER_AREA)
        frame = frame[80:120,:,:]
        frames.append(frame)
    cap.release()
  return frames



