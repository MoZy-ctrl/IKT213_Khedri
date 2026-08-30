import numpy as np
import cv2

def print_image_information(image):
    height, width, channels = image.shape
    size = image.size
    dtype = image.dtype

    print("Image Height:", height)
    print("Image Width:", width)
    print("Image Channels:", channels)
    print("Image Size:", size)
    print("Image Type:", dtype)

def main():
    image = cv2.imread("iris-1.jpg")
    print_image_information(image)

if __name__ == "__main__":
    main()