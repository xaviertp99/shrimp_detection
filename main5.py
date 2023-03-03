import cv2
import torch
from flask import Flask, render_template, jsonify, request

app = Flask(__name__, template_folder='C:\\Users\\User\\PycharmProjects\\pythonProject6\\venv\\Lib\\site-packages\\flask\\templates')

# Cargar el modelo pre-entrenado para la detección de objetos
model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:\\Users\\User\\yolov5\\best.pt', force_reload=True)

# Definir las clases de objetos que se van a detectar
classes = ['GRANDE CCYCL', 'GRANDE CCYSCL', 'GRANDE SCCYCL', 'GRANDE SCCYSCL', 'MEDIANO CCYCL', 'MEDIANO CCYSCL', 'MEDIANO SCCYCL', 'MEDIANO SCCYSCL', 'PEQUENO CCYCL', 'PEQUENO CCYSCL', 'PEQUEÑO SCCYCL', 'PEQUENO SCCYSCL']

# Inicializar el contador para cada clase de objeto detectado
counts = {cls: 0 for cls in classes}

# Definir la posición y tamaño de cada sección en la ventana
section_positions = [(0, 0), (0, 320), (0, 640), (0, 960),
                     (240, 0), (240, 320), (240, 640), (240, 960),
                     (480, 0), (480, 320), (480, 640), (480, 960)]

# Ruta para la página principal
@app.route("/")
def index():
    return render_template("index.html")

# Ruta para la inferencia
@app.route("/infer", methods=["POST"])
def infer():
    # Captura un frame de la cámara
    ret, frame = cap.read()

    # Detecta objetos en el frame
    results = model(frame)

    # Cuenta la cantidad de objetos detectados de cada clase
    for cls in classes:
        counts[cls] = 0
    for obj in results.xyxy[0]:
        if obj[4] > 0.5:
            cls = obj[5]
            if cls in classes:
                counts[cls] += 1

    # Formatea la salida como una cadena de texto
    output = ""
    for cls in classes:
        output += f"{cls}: {counts[cls]}\n"

    # Retorna la salida como una respuesta JSON
    return jsonify(output)

# Ruta para detener la inferencia
@app.route("/stop", methods=["POST"])
def stop():
    # Detiene la cámara
    cap.release()

    # Formatea la salida como una cadena de texto
    output = ""
    for cls in classes:
        output += f"{cls}: {counts[cls]}\n"

    # Retorna la salida como una respuesta JSON
    return jsonify(output)

# Ruta para reiniciar la inferencia
@app.route("/reset", methods=["POST"])
def reset():
    # Reinicia los contadores
    for cls in classes:
        counts[cls] = 0

    # Retorna una respuesta vacía
    return ""

# Inicia la cámara
cap = cv2.VideoCapture(0)

if __name__ == "__main__":
    app.run(host="0.0.0.0")
