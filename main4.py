from flask import Flask, render_template, Response, jsonify
import json
import cv2
import base64
import numpy as np
import httpx
import time
import asyncio

app = Flask(__name__, template_folder='C:\\Users\\User\\PycharmProjects\\pythonProject6\\venv\\Lib\\site-packages\\flask\\templates')


# load config
with open('roboflow_config.json') as f:
    config = json.load(f)

    ROBOFLOW_API_KEY = config["ROBOFLOW_API_KEY"]
    ROBOFLOW_MODEL = config["ROBOFLOW_MODEL"]
    ROBOFLOW_SIZE = config["ROBOFLOW_SIZE"]

    FRAMERATE = config["FRAMERATE"]
    BUFFER = config["BUFFER"]

# Construct the Roboflow Infer URL
# (if running locally replace https://detect.roboflow.com/ with eg http://127.0.0.1:9001/)
upload_url = "".join([
    "https://detect.roboflow.com/",
    ROBOFLOW_MODEL,
    "?api_key=",
    ROBOFLOW_API_KEY,
    "&format=image", # Change to json if you want the prediction boxes, not the visualization
    "&stroke=3",
    "&labels=true"
])

# Get webcam interface via opencv-python
video = cv2.VideoCapture(0)

# Infer via the Roboflow Infer API and return the result
# Takes an httpx.AsyncClient as a parameter
async def infer(requests):
    # Get the current image from the webcam
    ret, img = video.read()

    # Resize (while maintaining the aspect ratio) to improve speed and save bandwidth
    height, width, channels = img.shape
    scale = ROBOFLOW_SIZE / max(height, width)
    img = cv2.resize(img, (round(scale * width), round(scale * height)))

    # Encode image to base64 string
    retval, buffer = cv2.imencode('.jpg', img)
    img_str = base64.b64encode(buffer)

    # Get prediction from Roboflow Infer API
    resp = await requests.post(upload_url, data=img_str, headers={
        "Content-Type": "application/x-www-form-urlencoded"
    })

    # Parse result image
    image = np.asarray(bytearray(resp.content), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    return image
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/start_inference')
def start_inference():
    async def generate():
        async with httpx.AsyncClient() as requests:
            last_frame = time.time()
            futures = []

            while 1:
                # Throttle to FRAMERATE fps and print actual frames per second achieved
                elapsed = time.time() - last_frame
                await asyncio.sleep(max(0, 1/FRAMERATE - elapsed))
                last_frame = time.time()

                # Enqueue the inference request and safe it to our buffer
                task = asyncio.create_task(infer(requests))
                futures.append(task)

                # Wait until our buffer is big enough before we start displaying results
                if len(futures) < BUFFER * FRAMERATE:
                    continue

                # Remove the first image from our buffer
                # wait for it to finish loading (if necessary)
                image = await futures.pop(0)
                # And display the inference results
                ret, buffer = cv2.imencode('.jpg', image)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_inference')
def stop_inference():
    video.release()
    cv2.destroyAllWindows()

    return "Inference stopped!"

if __name__ == '__main__':
    app.run()

