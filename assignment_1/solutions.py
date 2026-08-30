import cv2

import os

def print_image_information(image):
    height, width, channels = image.shape
    size = image.size
    dtype = image.dtype

    print("Image Height:", height)
    print("Image Width:", width)
    print("Image Channels:", channels)
    print("Image Size:", size)
    print("Image Type:", dtype)

def save_camera_info():
    cam = cv2.VideoCapture(0)

    if not cam.isOpened():
        print("Cannot open camera")
        return

    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cam.get(cv2.CAP_PROP_FPS))

    os.makedirs("solutions", exist_ok=True)
    output_path = os.path.join("solutions", "camera_outputs.txt")
    with open(output_path, "w") as file:
        file.write(f"fps: {fps}\n")
        file.write(f"height: {frame_height}\n")
        file.write(f"width: {frame_width}\n")

    print(f"Camera info saved to {output_path}")

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        cv2.imshow("Camera", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

def main():
    image = cv2.imread("iris-1.jpg")
    print_image_information(image)

    save_camera_info()

if __name__ == "__main__":
    main()