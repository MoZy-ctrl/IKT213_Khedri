import numpy as np
import cv2
from numpy.ma.core import resize

image = cv2.imread("lambo.png")

"""Sobel edge detection"""
def sobel_edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, ksize=(3, 3), sigmaX=0)
    sobel_image = cv2.Sobel(blurred, cv2.CV_64F, 1, 1, ksize=1)
    return sobel_image

"""Canny edge detection"""
def canny_edge_detection(image, threshold_1, threshold_2):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, ksize=(3, 3), sigmaX=0)
    canny_image = cv2.Canny(blurred, threshold_1, threshold_2)
    return canny_image

"""Template matching"""
def template_match(image, template):
    shapes_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    w, h = template_gray.shape[::-1]

    res = cv2.matchTemplate(shapes_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    threshold = 0.9
    loc = np.where(res >= threshold)

    output = image.copy()
    for pt in zip(*loc[::-1]):
        cv2.rectangle(output, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

    return output

"""Resizing"""
def resize(image, scale_factor: int, up_or_down: str):
    output_image = image.copy()

    for i in range(scale_factor):
        if up_or_down == "up":
            output_image = cv2.pyrUp(output_image)
        elif up_or_down == "down":
            output_image = cv2.pyrDown(output_image)

    return output_image

if __name__ == "__main__":
    sobel_image = sobel_edge_detection(image)
    cv2.imwrite("sobel_image.png", sobel_image)

    canny_image = canny_edge_detection(image, threshold_1=50, threshold_2=50)
    cv2.imwrite("canny_image.png", canny_image)

    shape_1 = cv2.imread("given_image/shapes-1.png")
    shapes_template = cv2.imread("given_image/shapes_template.jpg")

    matched_image = template_match(shape_1, shapes_template)
    cv2.imwrite("matched_image.png", matched_image)

    resized_up = resize(image, scale_factor=2, up_or_down="up")
    cv2.imwrite("resized_up.png", resized_up)

    resized_down = resize(image, scale_factor=2, up_or_down="down")
    cv2.imwrite("resized_down.png", resized_down)