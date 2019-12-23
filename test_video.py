import numpy as np
import cupy as cp
import cv2 as cv
from canny_detector import CannyDetectorCuda, CannyDetectorSerial
import time


# init canny
# canny = CannyDetectorCuda()
canny = CannyDetectorSerial()
canny.set_threshold_ratio(0.1, 0.2)
# init video file
cap = cv.VideoCapture('SampleVideo_1280x720_30mb.mp4')
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ts = time.perf_counter()
    ret, frame = cap.read()
    frame = cv.resize(frame, (0, 0), None, .7, .7)
    # 504*896
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # canny edge operation
    if type(canny) == CannyDetectorCuda:
        image = cp.asarray(gray/255.)
    else:
        image = gray/255.
    # task 1
    blurred = canny.gaussian_blur(image)
    # task 2
    gradient, horizontal, vertical = canny.gradient(blurred)
    # task 3
    nms_edge = canny.nms(gradient, horizontal, vertical)
    # task 4
    final = canny.double_threshold(nms_edge)
    # display hstack
    if type(canny) == CannyDetectorCuda:
        final = cp.asnumpy(final)
    result = cv.cvtColor(np.uint8(final*255.), cv.COLOR_GRAY2BGR)
    show_image = np.concatenate((frame, result), axis=1)
    t_all = time.perf_counter() - ts
    fps = np.round(1./t_all, 2)
    cv.putText(show_image, str(fps), (20, 60), cv.FONT_HERSHEY_DUPLEX, 1,
               (0, 0, 255), 1)
    # Display the resulting frame
    cv.imshow('video', show_image)
    if cv.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
