import cv2
import numpy as np

"""Padding """
def padding(image, border_width):
    padded_image = cv2.copyMakeBorder(
        image,
        border_width,
        border_width,
        border_width,
        border_width,
        cv2.BORDER_REFLECT
    )
    return padded_image

"""Cropping"""
def crop(image, x_0, x_1, y_0, y_1):
    cropped_image = image[y_0:y_1, x_0:x_1]
    return cropped_image

"""resize"""
def resize(image, width, height):
    resized_image = cv2.resize(image, (width, height))
    return resized_image

"""Manual copy"""
def copy(image, emptyPictureArray):
    height, width, channels = image.shape

    for y in range(height):
        for x in range(width):
            for c in range(channels):
                emptyPictureArray[y, x, c] = image[y, x, c]

    return emptyPictureArray

"""Grayscale"""
def grayscale(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

"""HSV"""
def hsv(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return hsv

"""Color shifting"""
def hue_shifted(image, emptyPictureArray, hue):
    height, width, channels = image.shape

    for y in range(height):
        for x in range(width):
            for c in range(channels):
                emptyPictureArray[y, x, c] = (int(image[y, x, c]) + hue) % 256

    return emptyPictureArray

"""Smoothing"""
def smoothing(image):
    smoothed_image = cv2.GaussianBlur(image, ksize=(15, 15), sigmaX=0, borderType=cv2.BORDER_REFLECT)
    return smoothed_image

"""Rotation"""
def rotate(image, rotation_angle):
    """if rotate = 90, else rotate = 180"""
    if rotation_angle == 90:
        rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    else:
        rotated_image = cv2.rotate(image, cv2.ROTATE_180)

    return rotated_image

if __name__ == "__main__":
    image = cv2.imread("iris-1.jpg")
    padded_image = padding(image, border_width=100)

    cv2.imwrite("padded_image.png", padded_image)

    height, width = image.shape[0:2]

    x_0 = 200
    y_0 = 200
    x_1 = width - 130
    y_1 = height - 130

    cropped_image = crop(image, x_0, x_1, y_0, y_1)
    cv2.imwrite("cropped_image.png", cropped_image)

    resized_image = resize(image, width=200, height=200)
    cv2.imwrite("resized_image.png", resized_image)

    height, width, channels = image.shape
    emptyPictureArray = np.zeros((height, width, 3), dtype=np.uint8)
    copied_image = copy(image, emptyPictureArray)
    cv2.imwrite("copied_image.png", copied_image)

    grayscaled_image = grayscale(image)
    cv2.imwrite("grayscaled_image.png", grayscaled_image)

    hsv_image = hsv(image)
    cv2.imwrite("hsv_image.png", hsv_image)

    emptyPictureArray_hue = np.zeros((height, width, 3), dtype=np.uint8)
    hue_shifted_image = hue_shifted(image, emptyPictureArray_hue, hue=50)
    cv2.imwrite("hue_shifted_image.png", hue_shifted_image)

    smoothed_image = smoothing(image)
    cv2.imwrite("smoothed_image.png", smoothed_image)

    rotated_image90 = rotate(image, 90)
    cv2.imwrite("rotated_image90.png", rotated_image90)
    rotated_image180 = rotate(image, 180)
    cv2.imwrite("rotated_image180.png", rotated_image180)