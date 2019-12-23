import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import time

from module.canny_detector import CannyDetectorCuda, CannyDetectorSerial
from module.util import cuda_load_rawimag, serial_load_rawimag


# if __name__ == '__main__':
# make dir to save result images
save_path = 'results'
os.makedirs(save_path, exist_ok=True)

# cuda version
image_list = []
image_list = cuda_load_rawimag('Easy')
# init canny
canny_cuda = CannyDetectorCuda()
canny_cuda.load_image(image_list)
canny_cuda.set_threshold_ratio(0.1, 0.2)
canny_cuda.run_canny_detector()  # warmup
canny_cuda.run_canny_detector()  # get second time data
canny_cuda.run_canny_detector()  # get second time data
canny_cuda.run_canny_detector()  # get second time data
cuda_results_list = canny_cuda.result_image_list_
for i, image in enumerate(cuda_results_list):
    plt.imshow(image, cmap=plt.get_cmap('gray'))
    plt.axis('off')
    plt.title('Final image')
    final_image = image.astype(np.uint8)
    im = Image.fromarray(np.uint8(final_image*255), 'L')
    im.save(os.path.join(save_path, 'final_cuda_'+str(i)+'.png'))
# # serial version
# image_list = serial_load_rawimag('Easy')
# # init canny
# canny_serial = CannyDetectorSerial()
# canny_serial.load_image(image_list)
# canny_serial.set_threshold_ratio(0.1, 0.2)
# canny_serial.run_canny_detector()
# serial_results_list = canny_serial.result_image_list_
# for i, image in enumerate(serial_results_list):
#     plt.imshow(image, cmap=plt.get_cmap('gray'))
#     plt.axis('off')
#     plt.title('Final image')
#     final_image = image.astype(np.uint8)
#     im = Image.fromarray(np.uint8(final_image*255), 'L')
#     im.save(os.path.join(save_path, 'final_serial_'+str(i)+'.png'))
# opencv version
image_list = serial_load_rawimag('Easy')
image_list = [np.uint8(image*255) for image in image_list]
# image_list = image_list*10
cv2.setNumThreads(0)
# cv2.setUseOptimized(False)
opencv_results_list = []
opencv_tall = []
for image in image_list:
    t = time.perf_counter()
    blur_gray = cv2.GaussianBlur(image, (7, 7), 0)
    edges = cv2.Canny(blur_gray, 0, 50, apertureSize=3, L2gradient=True)
    opencv_results_list.append(edges)
    opencv_tall.append(time.perf_counter() - t)

for i, image in enumerate(opencv_results_list):
    plt.imshow(image, cmap=plt.get_cmap('gray'))
    plt.axis('off')
    plt.title('Final image')
    im = Image.fromarray(image, 'L')
    im.save(os.path.join(save_path, 'final_opencv_'+str(i)+'.png'))

# 5 picture
print('Cuda')
offset = 15
print('t1')
for j in range(5):
    print(canny_cuda.t1[j+offset])
print('t2')
for j in range(5):
    print(canny_cuda.t2[j+offset])
print('t3')
for j in range(5):
    print(canny_cuda.t3[j+offset])
print('t4')
for j in range(5):
    print(canny_cuda.t4[j+offset])
print('data transfer')
for j in range(5):
    print(canny_cuda.data_transfer[j+offset])
print('tall')
for j in range(5):
    print(canny_cuda.total_time[j+offset]-canny_cuda.data_transfer[j+offset])
print('')

# print('Serial')
# print('t1')
# for j in canny_serial.t1:
#     print(j)
# print('t2')
# for j in canny_serial.t2:
#     print(j)
# print('t3')
# for j in canny_serial.t3:
#     print(j)
# print('t4')
# for j in canny_serial.t4:
#     print(j)
# print('tall')
# for j in canny_serial.total_time:
#     print(j)
# print('')

print('opencv')
print('tall')
for j in opencv_tall:
    print(j)
