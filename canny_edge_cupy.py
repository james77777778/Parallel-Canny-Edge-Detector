import os
import time
from skimage import data
import matplotlib.pyplot as plt
import numpy as np
# for gpu numpy
import cupy as cp
import cupyx as cpx
import cupyx.scipy.ndimage
from skimage.color import rgb2gray
from PIL import Image


# stack function
def stackImage(padding, stack_image_list):
    stack_num = len(stack_image_list)
    stack_image = stack_image_list[0]
    for i in range(1, stack_num, 1):
        stack_image = np.vstack((stack_image, padding))
        stack_image = np.vstack((stack_image, stack_image_list[i]))
    return stack_image


# utils for read cu files
def read_code(code_filename, params=None):
    with open(code_filename, 'r') as f:
        code = f.read()
    if params is not None:
        for k, v in params.items():
            code = '#define ' + k + ' ' + str(v) + '\n' + code
    return code


# preload cuda kernel function (save time by compiling only once)
cuda_path = os.path.join(os.path.dirname(__file__), 'cuda')
nms_file = os.path.join(cuda_path, 'cu_suppress_non_max.cu')
high_file = os.path.join(cuda_path, 'cu_high.cu')
low_file = os.path.join(cuda_path, 'cu_hysteresis_low.cu')
nms_code = read_code(nms_file, params=None)
nms_kernel = cp.RawKernel(nms_code, 'cu_suppress_non_max')
HThread_code = read_code(high_file, params=None)
HThread_kernel = cp.RawKernel(HThread_code, 'cu_high')
LThread_code = read_code(low_file, params=None)
LThread_kernel = cp.RawKernel(LThread_code, 'cu_hysteresis_low')


# PARAMETERS
# whether to plot
is_plot = False
# number of stacking image
stack_num = 1
# cameraman image
image1 = data.camera()
image2 = rgb2gray(data.astronaut())
cameraman_image = image1/255.
# stack
single_height, single_width = cameraman_image.shape
padding_size = 7  # we have gaussian(7) & gradient(3)
# according to max kernel size & single image width
padding = np.zeros((padding_size, single_width))
stack_num = 1
stack_image_list = []  # put the image to the list for creating image stack
# here we stack the same image
for i in range(stack_num):
    stack_image_list.append(cameraman_image)
cameraman_image = stackImage(padding, stack_image_list)
# repeat number
repeat_n = 100
# get height and width
height, width = cameraman_image.shape
# move to gpu
cameraman_image = cp.asarray(cameraman_image)
# double threshold low, high ratio
low_thres_ratio = 0.1
high_thres_ratio = 0.2
# record time
t1 = t2 = t3 = t4 = 0

# START ALGORITHM
tall = time.perf_counter()
for _ in range(repeat_n):
    # Task1: Gaussian Blur
    ts = time.perf_counter()

    # pre-defined gaussian filter
    kernel = cp.array([
        [0.02040638, 0.0204074, 0.02040845, 0.02040644, 0.02040845,
            0.0204074, 0.02040638],
        [0.0204074, 0.02040842, 0.02040947, 0.02040747, 0.02040947,
            0.02040842, 0.0204074],
        [0.02040845, 0.02040947, 0.02041052, 0.02040851, 0.02041052,
            0.02040947, 0.02040845],
        [0.02040644, 0.02040747, 0.02040851, 0.02040651, 0.02040851,
            0.02040747, 0.02040644],
        [0.02040845, 0.02040947, 0.02041052, 0.02040851, 0.02041052,
            0.02040947, 0.02040845],
        [0.0204074, 0.02040842, 0.02040947, 0.02040747, 0.02040947,
            0.02040842, 0.0204074],
        [0.02040638, 0.0204074, 0.02040845, 0.02040644, 0.02040845,
            0.0204074, 0.02040638]])
    filtered_image = cpx.scipy.ndimage.convolve(cameraman_image, kernel)

    t1 += time.perf_counter()-ts

    # Task 1 plot
    if is_plot:
        fig1, axes_array = plt.subplots(1, 2)
        fig1.set_size_inches(8, 4)
        image_plot = axes_array[0].imshow(
            cp.asnumpy(cameraman_image), cmap=plt.get_cmap('gray'))
        axes_array[0].axis('off')
        axes_array[0].set(title='Original Image')
        blurred_image = cp.asnumpy(filtered_image)  # move to cpu for plot
        image_plot = axes_array[1].imshow(blurred_image,
                                          cmap=plt.get_cmap('gray'))
        axes_array[1].axis('off')
        axes_array[1].set(title='Filtered Image')
        plt.show()

    # Task2: Magnitude
    ts = time.perf_counter()

    smoothed_image = filtered_image.copy()
    horizontal_kernel = cp.array([[1., 2, 1], [0, 0, 0], [-1, -2, -1]])
    vertical_kernel = cp.array([[-1., 0, 1], [-2, 0, 2], [-1, 0, 1]])
    horizontal_edge_image = cpx.scipy.ndimage.convolve(
        smoothed_image, horizontal_kernel)
    vertical_edge_image = cpx.scipy.ndimage.convolve(
        smoothed_image, vertical_kernel)
    height, width = horizontal_edge_image.shape
    gradient_image = cp.sqrt(horizontal_edge_image**2 + vertical_edge_image**2)

    t2 += time.perf_counter()-ts

    # Task 2 plot
    if is_plot:
        plt.axis('off')
        plt.title('Gradient magnitude image')
        x, y = np.meshgrid(np.arange(0, height), np.arange(0, width))
        plt.imshow(cp.asnumpy(gradient_image), cmap=plt.get_cmap('gray'))
        plt.show()

    # Task3: NMS by cuda kernel function
    ts = time.perf_counter()

    height, width = horizontal_edge_image.shape
    block = 128
    grid = (height*width+block-1)//block
    edge_image = cp.zeros((height, width))
    # need cp.asfortranarray for memory direction for kernel function
    gradient_image = cp.asfortranarray(gradient_image, dtype=cp.float32)
    horizontal_edge_image = cp.asfortranarray(
        horizontal_edge_image, dtype=cp.float32)
    vertical_edge_image = cp.asfortranarray(
        vertical_edge_image, dtype=cp.float32)
    edge_image = cp.asfortranarray(edge_image, dtype=cp.float32)
    height = np.int32(cp.asnumpy(height))
    width = np.int32(cp.asnumpy(width))
    args = (gradient_image, horizontal_edge_image, vertical_edge_image,
            edge_image, height, width)
    nms_kernel((grid,), (block,), args=args)
    # remove edge in padding region
    for i in range(stack_num):
        edge_image[
            (i*(single_height+padding_size))+single_height:
            ((i+1)*(single_height+padding_size)), :] = 0
    edge_image_cp = edge_image.copy()

    t3 += time.perf_counter()-ts

    # Task 3 plot
    if is_plot:
        plt.axis('off')
        plt.title('Non maximum suppressed image')
        plt.imshow(cp.asnumpy(edge_image), cmap=plt.get_cmap('gray'))
        plt.show()

    # Task4: Double Threshold
    ts = time.perf_counter()

    unstack_list = []
    for i in range(stack_num):
        unstack_list.append(
            edge_image[(i*(single_height+padding_size)):
                       (i*(single_height+padding_size))+single_height, :])
    final_image_list = []

    # current_stream = cp.cuda.get_current_stream()
    for edge_image in unstack_list:
        high_threshold = edge_image.max()*high_thres_ratio
        low_threshold = high_threshold*low_thres_ratio
        final_image = cp.zeros((single_height, single_width))
        final_image = cp.asfortranarray(final_image, dtype=cp.float32)
        strong_edge_pixel = cp.zeros((single_height, single_width))
        strong_edge_pixel = cp.asfortranarray(strong_edge_pixel,
                                              dtype=cp.float32)
        edge_image = cp.asfortranarray(edge_image, dtype=cp.float32)
        high_threshold = np.float32(cp.asnumpy(high_threshold))
        low_threshold = np.float32(cp.asnumpy(low_threshold))
        argsH = (final_image, edge_image, strong_edge_pixel, high_threshold,
                 single_height, single_width)
        # high
        HThread_kernel((grid,), (block,), args=argsH)
        # low
        weak_edge_pixel = (
            (edge_image >= low_threshold) & (edge_image <= high_threshold))
        weak_edge_pixel = cp.asfortranarray(weak_edge_pixel, dtype=cp.float32)
        argsL = (final_image, edge_image, strong_edge_pixel, weak_edge_pixel,
                 low_threshold, single_height, single_width)
        LThread_kernel((grid,), (block,), args=argsL)
        final_image_list.append(cp.asnumpy(final_image))
        cp.cuda.Stream.null.synchronize()

    t4 += time.perf_counter()-ts

    # Task 4 plot
    if is_plot:
        padding = np.zeros((0, single_width))
        final_image = stackImage(padding, final_image_list)
        plt.imshow(final_image, cmap=plt.get_cmap('gray'))
        plt.axis('off')
        plt.title('Final image')
        plt.show()

    # # save plot
    # padding = np.zeros((0, single_width))
    # final_image = stackImage(padding, final_image_list)
    # final_image = final_image.astype(np.uint8)
    # im = Image.fromarray(np.uint8(final_image*255), 'L')
    # im.save('final_cupy.png')

tall = time.perf_counter() - tall
# Time
print('avg t1:', t1/repeat_n)
print('avg t2:', t2/repeat_n)
print('avg t3:', t3/repeat_n)
print('avg t4:', t4/repeat_n)
print('tall:', tall)
