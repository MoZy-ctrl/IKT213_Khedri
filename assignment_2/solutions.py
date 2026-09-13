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


