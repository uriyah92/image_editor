
from image_editor_helper import *
from typing import Optional


import math
import sys
import image_editor_helper

# separates picture to its red, blue and green colores


def separate_channels(image: ColoredImage) -> List[SingleChannelImage]:
    SingleChannelImage = list()
    SingleChannelImage = [[[0 for i in range(len(image[0]))] for j in range(
        len(image))] for m in range(len(image[0][0]))]
    for i in range(len(image)):
        for j in range(len(image[0])):
            for m in range(len(image[0][0])):
                SingleChannelImage[m][i][j] = image[i][j][m]
    return SingleChannelImage

# combinesred, blue and green colores to form a picture


def combine_channels(channels: List[SingleChannelImage]) -> ColoredImage:
    ColoredImage = list()
    ColoredImage = [[[0 for i in range(len(channels))]
                     for j in range(len(channels[0][0]))] for m in range(len(channels[0]))]
    for i in range(len(channels)):
        for j in range(len(channels[0])):
            for m in range(len(channels[0][0])):
                ColoredImage[j][m][i] = channels[i][j][m]
    return ColoredImage

# round number to the closest int


def round_num(num):
    if int(num)+1-num > num-int(num):
        return int(num)
    else:
        return int(num)+1

# turn colorized picture to black and white


def RGB2grayscale(colored_image: ColoredImage) -> SingleChannelImage:
    SingleChannelImage = list()
    for i in range(len(colored_image)):
        row_black_white = list()
        for j in range(len(colored_image[0])):
            red = 0.299*colored_image[i][j][0]
            green = 0.587*colored_image[i][j][1]
            blue = 0.114*colored_image[i][j][2]
            pixel_black_white = round_num(red+green+blue)
            row_black_white.append(pixel_black_white)
        SingleChannelImage.append(row_black_white)
    return SingleChannelImage

# creat kernel matrix


def blur_kernel(size: int) -> Kernel:
    Kernel = list()
    for i in range(size):
        kernel_row = list()
        for j in range(size):
            kernel_row.append(1/(size*size))
        Kernel.append(kernel_row)
    return Kernel

# check if a givven location is within picture parameters


def is_in_range(row, column, image):
    if row > len(image)-1:
        return False
    if column > len(image[0])-1:
        return False
    if row < 0:
        return False
    if column < 0:
        return False
    else:
        return True

# create a copy of the part of the image that will get blurred


def creat_matrix(image, kernel, row, column):
    matrix = list()
    for i in range(-(int(len(kernel)/2)), int(len(kernel)/2)+1):
        matrix_row = list()
        for j in range(-(int(len(kernel)/2)), int(len(kernel)/2)+1):
            if is_in_range(row+i, column+j, image):
                matrix_row.append((image[row+i][column+j]))
            else:
                matrix_row.append(image[row][column])
        matrix.append(matrix_row)
    return matrix

# calculate the kernel of a givven part of the image


def kernelize(matrix, kernel):
    sum = 0
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            sum += (matrix[i][j] * kernel[i][j])
    if sum < 0:
        sum = 0
    if sum > 255:
        sum = 255
    return round_num(sum)

# blurr picture


def apply_kernel(image: SingleChannelImage, kernel: Kernel) -> SingleChannelImage:
    SingleChannelImage = list()
    for i in range(len(image)):
        SingleChannelImage_row = list()
        for j in range(len(image[0])):
            matrix = creat_matrix(image, kernel, i, j)
            SingleChannelImage_row.append(kernelize(matrix, kernel))
        SingleChannelImage.append(SingleChannelImage_row)
    return SingleChannelImage

# return the smallest int that is bigger then a number


def round_up(num):
    if int(num) != num:
        return int(num)+1
    else:
        return int(num)

# return positive number


def delta(num):
    return num-int(num)

# calculate interpulation of a givven location


def bilinear_interpolation(image: SingleChannelImage, y: float, x: float) -> int:
    if x < 0 or x > len(image[0]) - 1 or y < 0 or y > len(image) - 1:
        return 0
    a = image[int(y)][int(x)]
    b = image[round_up(y)][int(x)]
    c = image[int(y)][round_up(x)]
    d = image[round_up(y)][round_up(x)]
    pixel = a*(1-delta(x))*(1-delta(y))+b*delta(y)*(1-delta(x)) + \
        c*delta(x)*(1-delta(y))+d*delta(x)*delta(y)
    return round_num(pixel)

# create a new version of a picture in givven sizes


def resize(image: SingleChannelImage, new_height: int, new_width: int) -> SingleChannelImage:
    SingleChannelImage = list()
    for i in range(new_height):
        SingleChannelImage_row = list()

        for j in range(new_width):

            if i == 0 and j == 0:
                SingleChannelImage_row.append(image[0][0])
            elif i == 0 and j == new_width-1:
                SingleChannelImage_row.append(image[0][len(image[0])-1])
            elif i == new_height-1 and j == 0:
                SingleChannelImage_row.append(image[len(image)-1][0])
            elif i == new_height-1 and j == new_width-1:
                SingleChannelImage_row.append(
                    image[len(image)-1][len(image[0])-1])
            else:
                y = i*(float(len(image))/(new_height-1))
                x = j*(float(len(image[0]))/(new_width-1))
                SingleChannelImage_row.append(
                    bilinear_interpolation(image, y, x))
        SingleChannelImage.append(SingleChannelImage_row)
    return SingleChannelImage

# turn photo sideways


def rotate_90(image: Image, direction: str) -> Image:
    new_image = list()
    if direction == 'R':
        for j in range(len(image[0])):
            new_image_row = list()
            for i in range(len(image)-1, -1, -1):
                new_image_row.append(image[i][j])
            new_image.append(new_image_row)
    if direction == 'L':
        for j in range(len(image[0])-1, -1, -1):
            new_image_row = list()
            for i in range(len(image)):
                new_image_row.append(image[i][j])
            new_image.append(new_image_row)
    return new_image

# calculate the color brightness limit of a givven part of the picture


def threshold(block_size, image, row, column, c):
    sum = 0
    counter = 0
    for i in range(-(int(block_size/2)), int(block_size/2)+1):
        for j in range(-(int(block_size/2)), int(block_size/2)+1):
            if is_in_range(row+i, column+j, image):
                sum += image[i][j]
                counter += 1
            else:
                sum += image[row][column]
                counter += 1
    return sum/counter - c

# create a new version of a photo with highter contrast


def get_edges(image: SingleChannelImage, blur_size: int, block_size: int, c: float) -> SingleChannelImage:
    kernel = blur_kernel(blur_size)
    blurred_image = apply_kernel(image, kernel)
    new_image = list()
    for i in range(len(image)):
        new_image_row = list()
        for j in range(len(image[0])):
            if image[i][j] < threshold(block_size, blurred_image, i, j, c):
                new_image_row.append(0)
            else:
                new_image_row.append(255)
        new_image.append(new_image_row)
    return new_image

# create a new version of a B&W photo with different color blending


def quantize(image: SingleChannelImage, N: int) -> SingleChannelImage:
    quantized_image = list()
    for i in range(len(image)):
        quantized_image_row = list()
        for j in range(len(image[0])):
            new_pixel = round(math.floor(image[i][j]*(N/256))*255/(N-1))
            quantized_image_row.append(new_pixel)
        quantized_image.append(quantized_image_row)
    return quantized_image

# create a new version of a colorized photo with different color blending


def quantize_colored_image(image: ColoredImage, N: int) -> ColoredImage:
    quantized_image = list()
    for i in range(len(image)):
        quantized_image_row = list()
        for j in range(len(image[0])):
            quantized_pixel = list()
            for m in range(len(image[0][0])):
                quantized_pixel.append(
                    round(math.floor(image[i][j][m]*(N/256))*255/(N-1)))
            quantized_image_row.append(quantized_pixel)
        quantized_image.append(quantized_image_row)
    return quantized_image

# check either a photo is colored or not


def is_colorful(image):
    if type(image[0][0]) == int:
        return False
    else:
        return True

# check if num is double


def is_valid(num):
    if int(num) != float(num):
        return False
    if int(num) % 2 == 0:
        return False
    else:
        return True

 # check if num is positive


def is_pos(num):
    if int(num) != num:
        return False
    if num <= 1:
        return False
    else:
        return True


# chek if the height and width are valid to resize
def is_resizable(new_size):
    args = new_size.split(',')
    if len(args) != 2:
        return False
    if float(args[0]) != round_num(float(args[0])):
        return False
    if float(args[1]) != round_num(float(args[1])):
        return False
    if float(args[0]) <= 1:
        return False
    if float(args[1]) <= 1:
        return False
    else:
        return True

# check if its valid direction


def is_rotation(direction):
    if direction == 'L' or direction == 'R'\
            or direction == 'l' or direction == 'r':
        return True
    else:
        return False

# check if parameters are valid to edge


def is_edgable(ansr):
    args = ansr.split(',')
    if len(args) != 3:
        return False
    if float(args[2]) < 0:
        return False
    if float(args[0]) != round_num(float(args[0])):
        return False
    if float(args[0]) < 0:
        return False
    if float(args[0]) % 2 == 0:
        return False
    if float(args[1]) != round_num(float(args[1])):
        return False
    if float(args[1]) < 0:
        return False
    if float(args[1]) % 2 == 0:
        return False
    else:
        return True


def analyze(new_size):
    if (is_resizable(new_size)):
        args = new_size.split(',')
        return int(args[0]), int(args[1])


def get_args(ansr):
    if (is_edgable(ansr)):
        args = ansr.split(',')
        return int(args[0]), int(args[1]), int(args[2])


def black_and_white(image):
    if not is_colorful(image):
        print(ERROR_MSG_2)
    else:
        image = RGB2grayscale(image)
        return image


def blurr(image):
    kernel_size = input(REQUEST_KARNEL)
    if is_valid(float(kernel_size)):
        kernel = blur_kernel(int(kernel_size))
        if is_colorful(image):
            image = separate_channels(image)
            for i in range(len(image)):
                image[i] = apply_kernel(image[i], kernel)
            image = combine_channels(image)
        else:
            image = apply_kernel(image, kernel)
        return image
    else:
        print(ERROR_MSG_3)


def resize_image(image):
    print(REQUEST_SIZES)
    new_size = input()
    if is_resizable(new_size):
        height, width = analyze(new_size)
        if is_colorful(image):
            rows = separate_channels(image)
            res = list()
            for row in rows:
                res.append(resize(row, height, width))
            image = combine_channels(res)
        else:
            image = resize(image, height, width)
        return image
    else:
        print(ERROR_MSG_3)


def turn_image(image):
    print(REQUEST_DIRECTION)
    direction = input()
    if is_rotation(direction):
        image = rotate_90(image, direction)
        return image
    else:
        print(ERROR_MSG_3)


def mark_edges(image):
    print(REQUEST_EDGES_PARAMETERS)
    params = input()
    if is_edgable(params):
        blur, block, c = get_args(params)
        if is_colorful(image):
            image = RGB2grayscale(image)
            image = get_edges(image, blur, block, c)
        else:
            image = get_edges(image, blur, block, c)
        return image
    else:
        print(ERROR_MSG_3)


def quantisize_image(image):
    print(REQUEST_QUNTIZATION_COLOR)
    quant = int(input())
    if is_pos(quant):
        if is_colorful(image):
            image = separate_channels(image)
            res = list()
            for i in range(len(image)):
                res.append(quantize(image[i], quant))
            image = combine_channels(res)
        else:
            image = quantize(image)
        return image
    else:
        print(ERROR_MSG_3)


def save_inage(image):
    print(REQUEST_PATH)
    path = input()
    image_editor_helper.save_image(image, path)


if __name__ == "__main__":
    args = sys.argv
    if len(args) <= 0:
        print(ERROR_MSG_1)
    else:
        image_path = args[1]
    image = image_editor_helper.load_image(image_path)
    new_image = image  # use a copy to not mess with original files
    print(WELCOM)
    while 1:
        print(MENUE)
        request = int(input())
        if request == 1:
            new_image = black_and_white(new_image)
            continue
        elif request == 2:
            new_image = blurr(new_image)
            continue
        elif request == 3:
            new_image = resize_image(new_image)
            continue
        elif request == 4:
            new_image = turn_image(new_image)
            continue
        elif request == 5:
            new_image = mark_edges(new_image)
            continue
        elif request == 6:
            new_image = quantisize_image(new_image)
            continue
        elif request == 7:
            image_editor_helper.show_image(new_image)
            continue
        elif request == 8:
            save_image(new_image)
            break
        else:
            print(ERROR_MSG_3)
            continue
