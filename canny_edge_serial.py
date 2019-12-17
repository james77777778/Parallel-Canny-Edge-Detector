import time
from skimage import data
import matplotlib.pyplot as plt
import scipy
import scipy.ndimage
import numpy as np
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
cameraman_image = np.asarray(cameraman_image)
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
    kernel = np.array([
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
    filtered_image = scipy.ndimage.convolve(cameraman_image, kernel)

    t1 += time.perf_counter()-ts

    # Task 1 plot
    if is_plot:
        fig1, axes_array = plt.subplots(1, 2)
        fig1.set_size_inches(8, 4)
        image_plot = axes_array[0].imshow(
            cameraman_image, cmap=plt.get_cmap('gray'))
        axes_array[0].axis('off')
        axes_array[0].set(title='Original Image')
        image_plot = axes_array[1].imshow(
            filtered_image, cmap=plt.get_cmap('gray'))
        axes_array[1].axis('off')
        axes_array[1].set(title='Filtered Image')
        plt.show()

    # Task2: Magnitude
    ts = time.perf_counter()

    smoothed_image = filtered_image.copy()
    horizontal_kernel = np.array([[1., 2, 1], [0, 0, 0], [-1, -2, -1]])
    vertical_kernel = np.array([[-1., 0, 1], [-2, 0, 2], [-1, 0, 1]])
    horizontal_edge_image = scipy.ndimage.convolve(
        smoothed_image, horizontal_kernel)
    vertical_edge_image = scipy.ndimage.convolve(
        smoothed_image, vertical_kernel)
    # magnitude
    gradient_image = np.sqrt(horizontal_edge_image**2 + vertical_edge_image**2)

    t2 += time.perf_counter()-ts

    if is_plot:
        plt.axis('off')
        plt.title('Gradient magnitude image')
        plt.imshow(gradient_image, cmap=plt.get_cmap('gray'))
        plt.show()

    # Task3: NMS
    ts = time.perf_counter()

    edge_image = np.zeros((height, width))
    # put zero all boundaries of image
    edge_image[0, :] = 0
    edge_image[-1, :] = 0
    edge_image[:, 0] = 0
    edge_image[:, -1] = 0
    deltaX = horizontal_edge_image.copy()
    deltaY = vertical_edge_image.copy()
    mag = gradient_image.copy()
    # not the boundaries
    for x in range(1, height-1):
        for y in range(1, width-1):
            if gradient_image[x, y] == 0.0:
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
    # remove edge in padding region
    for i in range(stack_num):
        edge_image[
            (i*(single_height+padding_size))+single_height:
            ((i+1)*(single_height+padding_size)), :] = 0
    edge_image_np = edge_image.copy()

    t3 += time.perf_counter()-ts

    # Task 3 plot
    if is_plot:
        plt.axis('off')
        plt.title('Non maximum suppressed image')
        plt.imshow(edge_image, cmap=plt.get_cmap('gray'))
        plt.show()

    # Task4: Double Threshold
    ts = time.perf_counter()

    unstack_list = []
    for i in range(stack_num):
        unstack_list.append(
            edge_image[(i*(single_height+padding_size)):
                       (i*(single_height+padding_size))+single_height, :])
    final_image_list = []

    for edge_image in unstack_list:
        final_image = np.zeros((512, 512))
        high_threshold = edge_image.max()*high_thres_ratio
        low_threshold = high_threshold*low_thres_ratio
        strong_edge_pixel = edge_image > high_threshold
        # strong_edge_pixels are a part of the edge
        final_image[strong_edge_pixel] = 1
        weak_edge_pixel = (
            (edge_image >= low_threshold) & (edge_image <= high_threshold))
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
        final_image_list.append(final_image)

    t4 += time.perf_counter()-ts

    if is_plot:
        plt.imshow(final_image, cmap=plt.get_cmap('gray'))
        plt.axis('off')
        plt.title('Final image')
        plt.show()

    # # save plot
    # padding = np.zeros((0, single_width))
    # final_image = stackImage(padding, final_image_list)
    # final_image = final_image.astype(np.uint8)
    # im = Image.fromarray(np.uint8(final_image*255), 'L')
    # im.save('final_serial.png')

tall = time.perf_counter() - tall
# Time
print('avg t1:', t1/repeat_n)
print('avg t2:', t2/repeat_n)
print('avg t3:', t3/repeat_n)
print('avg t4:', t4/repeat_n)
print('tall:', tall)
