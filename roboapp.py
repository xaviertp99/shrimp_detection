import json
import asyncio
import cv2
import base64
import numpy as np
import httpx
import time
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox


class RoboflowApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Roboflow App")

        # Load config from file
        with open('oboflow_config.json') as f:
            config = json.load(f)

        self.roboflow_api_key = config["ROBOFLOW_API_KEY"]
        self.roboflow_model = config["ROBOFLOW_MODEL"]
        self.roboflow_size = config["ROBOFLOW_SIZE"]
        self.framerate = config["FRAMERATE"]
        self.buffer = config["BUFFER"]

        # Construct the Roboflow Infer URL
        # (if running locally replace https://detect.roboflow.com/ with eg http://127.0.0.1:9001/)
        self.upload_url = "".join([
            "https://detect.roboflow.com/",
            self.roboflow_model,
            "?api_key=",
            self.roboflow_api_key,
            "&format=image",  # Change to json if you want the prediction boxes, not the visualization
            "&stroke=3",
            "&labels=true"
        ])

        # Create GUI elements
        self.btn_start = ttk.Button(self.root, text="Start", command=self.start)
        self.btn_stop = ttk.Button(self.root, text="Stop", command=self.stop)
        self.btn_stop.config(state="disabled")

        self.table_headers = ["Class", "Confidence"]
        self.table = ttk.Treeview(self.root, columns=self.table_headers, show="headings")
        for header in self.table_headers:
            self.table.heading(header, text=header)

        # Layout GUI elements
        self.btn_start.pack(side=tk.LEFT, padx=5, pady=5)
        self.btn_stop.pack(side=tk.LEFT, padx=5, pady=5)
        self.table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Initialize variables
        self.video = None
        self.is_running = False
        self.last_frame = None
        self.futures = []

    def start(self):
        # Initialize webcam
        self.video = cv2.VideoCapture(0)

        # Start main loop
        self.is_running = True
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        asyncio.run(self.main_loop())

    async def main_loop(self):
        async with httpx.AsyncClient() as requests:
            while self.is_running:
                # On "q" keypress, exit
                if (cv2.waitKey(1) == ord('q')):
                    break

                # Throttle to FRAMERATE fps and print actual frames per second achieved
                if self.last_frame is not None:
                    elapsed = time.time() - self.last_frame
                    await asyncio.sleep(max(0, 1 / self.framerate - elapsed))
                    print((1 / (time.time() - self.last_frame)), " fps")
                self.last_frame = time.time()

                # Enqueue the inference request and safe it to our buffer
                task = asyncio.create_task(self.infer(requests))
                self.futures.append(task)

                # Wait until our buffer is big enough before we start displaying results
                if len(self.futures) < self.buffer * self.framerate:
                    continue

                # Remove the first image from our buffer
                done, self.futures = await asyncio.wait([self.futures[0]], timeout=0)
                task = list(done)[0]
                boxes = task.result()
                # Parse result image
                image = np.asarray(bytearray(resp.content), dtype="uint8")
                image = cv2.imdecode(image, cv2.IMREAD_COLOR)

                return image
                # Draw boxes and labels
                img = self.draw_boxes(image, boxes)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


                # Display image
                cv2.imshow(self.window_name, img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                cv2.destroyAllWindows()