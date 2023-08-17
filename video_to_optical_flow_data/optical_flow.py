import numpy as np
import cv2 as cv
from Face_detection import  face_detection
from matplotlib import pyplot as plt
# cap = cv.VideoCapture(cv.samples.findFile("/content/bwbn2s.mpg"))
def video_to_optical_flow(video_path:str,
                          model_path:str):
    '''将video转换为带有时间关系的optical flow'''
    #建立空列表用于保存转换之后的optical flow
    optical_flow_list = []
    frames = face_detection.face_lip_clip(video_path=video_path,
                model_path=model_path)
    # ret, frame1 = cap.read()
    prvs = cv.cvtColor(frames[0], cv.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frames[0])
    hsv[..., 1] = 255
    for i in range(74):
        # ret, frame2 = cap.read()
        next = cv.cvtColor(frames[i + 1], cv.COLOR_BGR2GRAY)
        flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang*180/np.pi/2
        hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
        bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        optical_flow_list.append(bgr)
    #插入之前被利用的第一个frame
    optical_flow_list.insert(0,frames[0])
    optical_flow_list_all = np.stack(optical_flow_list)
    #返回的是所有optical flow的的stack
    return optical_flow_list_all


# if __name__=='__main__':
#     optical_flow = video_to_optical_flow(video_path='../data/train/s/swwpzs.mpg',
#                                          model_path='../Face_detection/blaze_face_short_range.tflite')
#     print(len(optical_flow))
#     for i in range(len(optical_flow)):
#         print(optical_flow[i].shape)


# 输出的类型为<class 'numpy.ndarray'>
#     print(f'frame:{bgr}')
#     cv.imshow('optical_flow', bgr)
#     k = cv.waitKey(30) & 0xff
#     if k == 27:
#         break
#     elif k == ord('s'):
#         cv.imwrite('opticalfb.png', frames[i + 1])
#         cv.imwrite('opticalhsv.png', bgr)
#     prvs = next
# cv.destroyAllWindows()