import cv2 as cv
import base64
import numpy as np
import requests

ROBOFLOW_API_KEY = "wgQ2NUXoRmyebbiz1sib"
ROBOFLOW_MODEL = "shrimp-detection-m0gh7/1"  # eg xx-xxxx--#
ROBOFLOW_SIZE = 416

upload_url = "".join([
    "https://detect.roboflow.com/",
    ROBOFLOW_MODEL,
    "?access_token=",
    ROBOFLOW_API_KEY,
    "&format=image",
    "&stroke=5"
])

video = cv.VideoCapture(0)

# Check if camera opened successfully
if (video.isOpened()):
    print("successfully opening video stream or file")


# Infer via the Roboflow Infer API and return the result
def infer():
    # Get the current image from the webcam
    ret, frame = video.read()

    # Resize (while maintaining the aspect ratio)
    # to improve speed and save bandwidth
    height, width, channels = frame.shape
    scale = ROBOFLOW_SIZE / max(height, width)
    img = cv.resize(frame, (round(scale * width), round(scale * height)))

    # Encode image to base64 string
    retval, buffer = cv.imencode('.mp4', img)
    img_str = base64.b64encode(buffer)

    # Get prediction from Roboflow Infer API
    resp = requests.post(upload_url, data=img_str, headers={
        "Content-Type": "application/x-www-form-urlencoded"
    }, stream=True).raw

    # Parse result image
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv.imdecode(image, cv.IMREAD_COLOR)

    return image


# Main loop; infers sequentially until you press "q"
while video.isOpened():
    # On "q" keypress, exit
    key = cv.waitKey(0)
    if key == ord("q"):
        break

    # Synchronously get a prediction from the Roboflow Infer API
    image = infer()
    # And display the inference results
    cv.imshow('image', image)

# Release resources when finished
video.release()
cv.destroyAllWindows()