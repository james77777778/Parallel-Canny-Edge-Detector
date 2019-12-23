import os
import itertools
import numpy as np
import scipy
import scipy.ndimage
import cupy as cp
import cupyx as cpx
import cupyx.scipy.ndimage
import time

from module.util import serial_load_rawimag, cuda_load_rawimag


# utils for read cu files
def read_code(code_filename, params=None):
    with open(code_filename, 'r') as f:
        code = f.read()
    if params is not None:
        for k, v in params.items():
            code = '#define ' + k + ' ' + str(v) + '\n' + code
    return code


class CannyDetectorCuda():
    def __init__(self):
        # preload cuda kernel function (save time by compiling only once)
        cuda_path = os.path.join(os.path.dirname(__file__), '..', 'cuda')
        nms_file = os.path.join(cuda_path, 'cu_suppress_non_max.cu')
        high_file = os.path.join(cuda_path, 'cu_high.cu')
        low_file = os.path.join(cuda_path, 'cu_hysteresis_low.cu')
        nms_code = read_code(nms_file, params=None)
        self.nms_kernel = cp.RawKernel(nms_code, 'cu_suppress_non_max')
        HThread_code = read_code(high_file, params=None)
        self.HThread_kernel = cp.RawKernel(HThread_code, 'cu_high')
        LThread_code = read_code(low_file, params=None)
        self.LThread_kernel = cp.RawKernel(LThread_code, 'cu_hysteresis_low')
        # pre-defined gaussian filter
        self.gaussian_filter = cp.array([
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
        # pre-defined gradient filter
        self.xaxis_filter = cp.array([[1., 2, 1], [0, 0, 0], [-1, -2, -1]])
        self.yaxis_filter = cp.array([[-1., 0, 1], [-2, 0, 2], [-1, 0, 1]])

        # pre-defined necessary objects
        self.image_list_ = []
        # double threshold low, high ratio
        self.low_thres_ratio = 0.1
        self.high_thres_ratio = 0.2
        # # now image height, width
        # self.now_height = 0
        # self.now_width = 0
        self.t1 = []
        self.t2 = []
        self.t3 = []
        self.t4 = []
        self.total_time = []
        self.data_transfer = []

    def show_time_result(self):
        image_list_len = len(self.image_list_)
        print('Canny Detector Cuda Version (img len:', image_list_len, ')')
        print('avg t1: ', self.t1/image_list_len)
        print('avg t2: ', self.t2/image_list_len)
        print('avg t3: ', self.t3/image_list_len)
        print('avg t4: ', self.t4/image_list_len)
        print('avg total:\t', (self.t1+self.t2+self.t3+self.t4)/image_list_len)

    def get_time_result(self):
        return (self.t1, self.t2, self.t3, self.t4,
                self.t1+self.t2+self.t3+self.t4)

    def load_image(self, image_list):
        if type(image_list) is not list:
            self.image_list_.append(image_list)
        else:
            self.image_list_ = image_list

    def clear_image(self):
        self.image_list_ = []

    def set_threshold_ratio(self, low_thres_ratio=0.1, high_thres_ratio=0.2):
        self.low_thres_ratio = low_thres_ratio
        self.high_thres_ratio = high_thres_ratio

    # task 1
    def gaussian_blur(self, image):
        ts = time.perf_counter()
        blurred_image = cpx.scipy.ndimage.convolve(image, self.gaussian_filter)
        cp.cuda.Stream.null.synchronize()
        self.t1.append(time.perf_counter()-ts)
        return blurred_image.copy()

    # task 2
    def gradient(self, image):
        ts = time.perf_counter()
        horizontal_edge = cpx.scipy.ndimage.convolve(image, self.xaxis_filter)
        vertical_edge = cpx.scipy.ndimage.convolve(image, self.yaxis_filter)
        gradient_image = cp.zeros_like(horizontal_edge)
        cp.sqrt(
            cp.power(horizontal_edge, 2) + cp.power(vertical_edge, 2),
            out=gradient_image)
        cp.cuda.Stream.null.synchronize()
        self.t2.append(time.perf_counter()-ts)
        return (gradient_image.copy(), horizontal_edge.copy(),
                vertical_edge.copy())

    # task 3
    def nms(self, in_gradient_image, in_horizontal_edge, in_vertical_edge):
        ts = time.perf_counter()
        height, width = in_gradient_image.shape
        block = 128
        grid = (height*width+block-1)//block
        in_gradient_image = cp.asfortranarray(in_gradient_image,
                                              dtype=cp.float32)
        in_horizontal_edge = cp.asfortranarray(in_horizontal_edge,
                                               dtype=cp.float32)
        in_vertical_edge = cp.asfortranarray(in_vertical_edge,
                                             dtype=cp.float32)
        edge_image = cp.zeros((height, width))
        edge_image = cp.asfortranarray(edge_image, dtype=cp.float32)
        height = np.int32(cp.asnumpy(height))
        width = np.int32(cp.asnumpy(width))
        args = (in_gradient_image, in_horizontal_edge, in_vertical_edge,
                edge_image, width, height)
        self.nms_kernel((grid,), (block,), args=args)
        cp.cuda.Stream.null.synchronize()
        self.t3.append(time.perf_counter()-ts)
        return edge_image.copy()

    # task 4
    def double_threshold(self, in_edge_image):
        ts = time.perf_counter()
        high_threshold = in_edge_image.max()*self.high_thres_ratio
        low_threshold = high_threshold*self.low_thres_ratio
        height, width = in_edge_image.shape
        block = 128
        grid = (height*width+block-1)//block
        final_image = cp.zeros((height, width))
        final_image = cp.asfortranarray(final_image, dtype=cp.float32)
        strong_edge_pixel = cp.zeros((height, width))
        strong_edge_pixel = cp.asfortranarray(strong_edge_pixel,
                                              dtype=cp.float32)
        edge_image = cp.asfortranarray(in_edge_image, dtype=cp.float32)
        high_threshold = np.float32(cp.asnumpy(high_threshold))
        low_threshold = np.float32(cp.asnumpy(low_threshold))
        height = np.int32(cp.asnumpy(height))
        width = np.int32(cp.asnumpy(width))
        argsH = (final_image, edge_image, strong_edge_pixel, high_threshold,
                 width, height)
        # high
        self.HThread_kernel((grid,), (block,), args=argsH)
        # low
        weak_edge_pixel = (
            (edge_image >= low_threshold) & (edge_image <= high_threshold))
        weak_edge_pixel = cp.asfortranarray(weak_edge_pixel, dtype=cp.float32)
        argsL = (final_image, edge_image, strong_edge_pixel, weak_edge_pixel,
                 low_threshold, width, height)
        self.LThread_kernel((grid,), (block,), args=argsL)
        cp.cuda.Stream.null.synchronize()
        self.t4.append(time.perf_counter()-ts)
        return final_image.copy()

    # run all algorithm: task 1~4
    # need to preload image_list, preset threshold ratio
    # results will be saved at self.result_image_list_
    def run_canny_detector(self):
        if len(self.image_list_) == 0:
            print("no data in self.image_list_: need to call load_image()")
            return
        self.result_image_list_ = []
        # run all tasks
        for image in self.image_list_:
            # run algorithm
            tt = time.perf_counter()
            blurred = self.gaussian_blur(image)
            gradient, horizontal, vertical = self.gradient(blurred)
            nms_edge = self.nms(gradient, horizontal, vertical)
            final_image = self.double_threshold(nms_edge)

            # move from gpu to cpu to self.result_image_list_
            ts = time.perf_counter()
            self.result_image_list_.append(cp.asnumpy(final_image))
            cp.cuda.Stream.null.synchronize()
            self.data_transfer.append(time.perf_counter()-ts)

            self.total_time.append(time.perf_counter()-tt)


class CannyDetectorSerial():
    def __init__(self):
        # pre-defined gaussian filter
        self.gaussian_filter = np.array([
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
        # pre-defined gradient filter
        self.xaxis_filter = np.array([[1., 2, 1], [0, 0, 0], [-1, -2, -1]])
        self.yaxis_filter = np.array([[-1., 0, 1], [-2, 0, 2], [-1, 0, 1]])

        # pre-defined necessary objects
        self.image_list_ = []
        # double threshold low, high ratio
        self.low_thres_ratio = 0.1
        self.high_thres_ratio = 0.2
        # # now image height, width
        # self.now_height = 0
        # self.now_width = 0
        # time list
        self.t1 = []
        self.t2 = []
        self.t3 = []
        self.t4 = []
        self.total_time = []

    def show_time_result(self):
        image_list_len = len(self.image_list_)
        print('Canny Detector Serial Version (img len:', image_list_len, ')')
        print('avg t1:\t', self.t1/image_list_len)
        print('avg t2:\t', self.t2/image_list_len)
        print('avg t3:\t', self.t3/image_list_len)
        print('avg t4:\t', self.t4/image_list_len)
        print('avg total:\t', (self.t1+self.t2+self.t3+self.t4)/image_list_len)

    def get_time_result(self):
        return (self.t1, self.t2, self.t3, self.t4,
                self.t1+self.t2+self.t3+self.t4)

    def load_image(self, image_list):
        if type(image_list) is not list:
            self.image_list_.append(image_list)
        else:
            self.image_list_ = image_list

    def clear_image(self):
        self.image_list_ = []

    def set_threshold_ratio(self, low_thres_ratio=0.1, high_thres_ratio=0.2):
        self.low_thres_ratio = low_thres_ratio
        self.high_thres_ratio = high_thres_ratio

    # task 1
    def gaussian_blur(self, image):
        ts = time.perf_counter()
        blurred_image = scipy.ndimage.convolve(image, self.gaussian_filter)
        self.t1.append(time.perf_counter()-ts)
        return blurred_image.copy()

    # task 2
    def gradient(self, image):
        ts = time.perf_counter()
        horizontal_edge = scipy.ndimage.convolve(image, self.xaxis_filter)
        vertical_edge = scipy.ndimage.convolve(image, self.yaxis_filter)
        gradient_image = np.sqrt(horizontal_edge**2 + vertical_edge**2)
        self.t2.append(time.perf_counter()-ts)
        return (gradient_image.copy(), horizontal_edge.copy(),
                vertical_edge.copy())

    # task 3
    def nms(self, in_gradient_image, in_horizontal_edge, in_vertical_edge):
        ts = time.perf_counter()
        height, width = in_gradient_image.shape
        edge_image = np.zeros((height, width))
        # put zero all boundaries of image
        edge_image[0, :] = 0
        edge_image[-1, :] = 0
        edge_image[:, 0] = 0
        edge_image[:, -1] = 0
        deltaX = in_horizontal_edge.copy()
        deltaY = in_vertical_edge.copy()
        mag = in_gradient_image.copy()
        # not the boundaries
        for x, y in itertools.product(range(1, height-1), range(1, width-1)):
            if mag[x, y] == 0.0:
                edge_image[x, y] = 0.0
            else:
                if deltaX[x, y] >= 0.0:
                    if deltaY[x, y] >= 0.0:
                        if (deltaX[x, y]-deltaY[x, y]) >= 0:  # direction 1
                            alpha = float(deltaY[x, y])/deltaX[x, y]
                            mag1 = (1-alpha)*mag[x+1, y]+alpha*mag[x+1, y+1]
                            mag2 = (1-alpha)*mag[x-1, y]+alpha*mag[x-1, y-1]
                        else:  # direction 2
                            alpha = float(deltaX[x, y])/deltaY[x, y]
                            mag1 = (1-alpha)*mag[x, y+1]+alpha*mag[x+1, y+1]
                            mag2 = (1-alpha)*mag[x, y-1]+alpha*mag[x-1, y-1]
                    else:
                        if (deltaX[x, y]+deltaY[x, y]) >= 0:  # direction 8
                            alpha = float(-deltaY[x, y])/deltaX[x, y]
                            mag1 = (1-alpha)*mag[x+1, y]+alpha*mag[x+1, y-1]
                            mag2 = (1-alpha)*mag[x-1, y]+alpha*mag[x-1, y+1]
                        else:  # direction 7
                            alpha = float(deltaX[x, y])/-deltaY[x, y]
                            mag1 = (1-alpha)*mag[x, y+1]+alpha*mag[x-1, y+1]
                            mag2 = (1-alpha)*mag[x, y-1]+alpha*mag[x+1, y-1]
                else:
                    if deltaY[x, y] >= 0.0:
                        if (deltaX[x, y]+deltaY[x, y]) >= 0:  # direction 3
                            alpha = float(-deltaX[x, y])/deltaY[x, y]
                            mag1 = (1-alpha)*mag[x, y+1]+alpha*mag[x-1, y+1]
                            mag2 = (1-alpha)*mag[x, y-1]+alpha*mag[x+1, y-1]
                        else:  # direction 4
                            alpha = float(deltaY[x, y])/-deltaX[x, y]
                            mag1 = (1-alpha)*mag[x-1, y]+alpha*mag[x-1, y+1]
                            mag2 = (1-alpha)*mag[x+1, y]+alpha*mag[x+1, y-1]
                    else:
                        if (-deltaX[x, y]+deltaY[x, y]) >= 0:  # direction 5
                            alpha = float(deltaY[x, y])/deltaX[x, y]
                            mag1 = (1-alpha)*mag[x-1, y]+alpha*mag[x-1, y-1]
                            mag2 = (1-alpha)*mag[x+1, y]+alpha*mag[x+1, y+1]
                        else:  # direction 6
                            alpha = float(deltaX[x, y])/deltaY[x, y]
                            mag1 = (1-alpha)*mag[x, y-1]+alpha*mag[x-1, y-1]
                            mag2 = (1-alpha)*mag[x, y+1]+alpha*mag[x+1, y+1]
                if ((mag[x, y] < mag1) or (mag[x, y] < mag2)):
                    edge_image[x, y] = 0.0
                else:
                    edge_image[x, y] = mag[x, y]
        self.t3.append(time.perf_counter()-ts)
        return edge_image.copy()

    # task 4
    def double_threshold(self, in_edge_image):
        ts = time.perf_counter()
        high_threshold = in_edge_image.max()*self.high_thres_ratio
        low_threshold = high_threshold*self.low_thres_ratio
        height, width = in_edge_image.shape
        final_image = np.zeros((height, width))
        strong_edge_pixel = in_edge_image > high_threshold
        weak_edge_pixel = (
            (in_edge_image >= low_threshold) &
            (in_edge_image <= high_threshold))
        final_image[strong_edge_pixel] = 1
        index_weak_edge_pixel = np.argwhere(weak_edge_pixel)
        for i in index_weak_edge_pixel:
            x = i[0]
            y = i[1]
            if(x > 0 and y > 0 and x < 511 and y < 511):
                north = (x-1, y)
                south = (x+1, y)
                east = (x, y+1)
                west = (x, y-1)
                north_east = (x-1, y+1)
                north_west = (x-1, y-1)
                south_west = (x+1, y-1)
                south_east = (x+1, y+1)
                # check if any of the 8 neighbors is a strong edge pixel
                if((strong_edge_pixel[north] > 0) or
                   (strong_edge_pixel[south] > 0) or
                   (strong_edge_pixel[east] > 0) or
                   (strong_edge_pixel[west] > 0) or
                   (strong_edge_pixel[north_east] > 0) or
                   (strong_edge_pixel[north_west] > 0) or
                   (strong_edge_pixel[south_west] > 0) or
                   (strong_edge_pixel[south_east] > 0)):
                    final_image[x][y] = 1  # classify the pixel as an edge
        self.t4.append(time.perf_counter()-ts)
        return final_image.copy()

    # run all algorithm: task 1~4
    # need to preload image_list, preset threshold ratio
    # results will be saved at self.result_image_list_
    def run_canny_detector(self):
        if len(self.image_list_) == 0:
            print("no data in self.image_list_: need to call load_image()")
            return
        self.result_image_list_ = []
        # run all tasks
        for image in self.image_list_:
            ts = time.perf_counter()
            blurred = self.gaussian_blur(image)
            gradient, horizontal, vertical = self.gradient(blurred)
            nms_edge = self.nms(gradient, horizontal, vertical)
            final_image = self.double_threshold(nms_edge)
            self.total_time.append(time.perf_counter()-ts)
            # move from gpu to cpu to self.result_image_list_
            self.result_image_list_.append(final_image)


# test code
if __name__ == '__main__':
    # cuda version
    # load image from skimage.data and move from cpu to gpu
    image_list = []
    image_list = cuda_load_rawimag('Easy')
    # init canny
    canny_cuda = CannyDetectorCuda()
    canny_cuda.load_image(image_list)
    canny_cuda.set_threshold_ratio(0.1, 0.2)
    canny_cuda.run_canny_detector()  # warmup
    canny_cuda.run_canny_detector()  # get second time data
    # serial version
    image_list = serial_load_rawimag('Easy')
    # init canny
    canny_serial = CannyDetectorSerial()
    canny_serial.load_image(image_list)
    canny_serial.set_threshold_ratio(0.1, 0.2)
    canny_serial.run_canny_detector()
    # 5 picture
    print('Cuda')
    print('t1')
    for j in range(5):
        print(canny_cuda.t1[j+5])
    print('t2')
    for j in range(5):
        print(canny_cuda.t2[j+5])
    print('t3')
    for j in range(5):
        print(canny_cuda.t3[j+5])
    print('t4')
    for j in range(5):
        print(canny_cuda.t4[j+5])
    print('data transfer')
    for j in range(5):
        print(canny_cuda.data_transfer[j+5])
    print('tall')
    for j in range(5):
        print(canny_cuda.total_time[j+5])
    print('')
    print('Serial')
    print('t1')
    for j in canny_serial.t1:
        print(j)
    print('t2')
    for j in canny_serial.t2:
        print(j)
    print('t3')
    for j in canny_serial.t3:
        print(j)
    print('t4')
    for j in canny_serial.t4:
        print(j)
    print('tall')
    for j in canny_serial.total_time:
        print(j)
    print('')
