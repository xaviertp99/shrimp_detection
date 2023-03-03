import cv2
import torch
import tkinter as tk
from PIL import Image, ImageTk
from prettytable import PrettyTable

# Cargar el modelo pre-entrenado para la detección de objetos
model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:\\Users\\User\\yolov5\\best.pt', force_reload=True)

# Inicia la cámara
cap = cv2.VideoCapture('C:\\Users\\User\\Videos\\camaron2.mp4')

# Definir las clases de objetos que se van a detectar
classes = ['GRANDE CCYCL', 'GRANDE CCYSCL', 'GRANDE SCCYCL', 'GRANDE SCCYSCL', 'MEDIANO CCYCL', 'MEDIANO CCYSCL',
           'MEDIANO SCCYCL', 'MEDIANO SCCYSCL', 'PEQUENO CCYCL', 'PEQUENO CCYSCL', 'PEQUEÑO SCCYCL', 'PEQUENO SCCYSCL']

# Inicializar el contador para cada clase de objeto detectado
counts = {cls: 0 for cls in classes}

# Definir la posición y tamaño de cada sección en la ventana
section_positions = [(0, 0), (0, 320), (0, 640), (0, 960),
                     (240, 0), (240, 320), (240, 640), (240, 960),
                     (480, 0), (480, 320), (480, 640), (480, 960)]

# Crear la ventana de tkinter
root = tk.Tk()

# Cargar la imagen
image = Image.open("C:\\Users\\User\\OneDrive\\Imágenes\\Universidad-Politecnica-Salesiana.jpg")
# Convertir la imagen en un objeto PhotoImage
photo = ImageTk.PhotoImage(image)
# Crear el widget de imagen
image_widget = tk.Label(root, image=photo)
image_widget.pack()

# Crear los botones
button_start = tk.Button(root, text="REINICIO", command=lambda: cap.open('C:\\Users\\User\\Videos\\camaron2.mp4'), height=3, width=15)
button_start.pack(side=tk.LEFT, padx=20, pady=10)
button_stop = tk.Button(root, text="DETENGA", command=lambda: cap.release(), height=3, width=15)
button_stop.pack(side=tk.LEFT, padx=20, pady=10)
button_exit = tk.Button(root, text="SALIR", command=lambda: root.destroy(), height=3, width=15)
button_exit.pack(side=tk.LEFT, padx=20, pady=10)


while True:
    # Captura un frame de la cámara
    ret, frame = cap.read()
    root.update()
    # Crea una sección para mostrar las detecciones de camarones
    section_size = 200
    section_x = 20
    section_y = frame.shape[0] - section_size - 20
    section_frame = frame[section_y:(section_y + section_size), section_x:(section_x + section_size), :]

    # Detecta objetos en el frame utilizando el modelo
    results = model(frame)

    # Dibuja los cuadros de los objetos detectados en el frame y en la sección
    for result in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = result

        # Obtener la clase predicha
        predicted_class = classes[int(cls)]

        # Actualizar el contador para la clase correspondiente
        counts[predicted_class] += 1

        # Dibujar el cuadro del objeto detectado en el frame
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        # Dibujar el texto con la clase y la confianza en el frame
        label = f'{predicted_class}: {conf:.2f}'
        cv2.putText(frame, label, (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Dibujar el texto con la clase y la confianza en la sección
        section_label = f'{predicted_class.split()[0][:2]}: {conf:.2f}'
        section_x_pos = int((section_size - len(section_label) * 10) / 2)
        section_y_pos = int(20 + (int(cls) % 4) * 40)
        cv2.putText(section_frame, section_label, (section_x_pos, section_y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 2)

    # Dibujar el contador de objetos detectados en el frame
    count_text = ', '.join([f'{cls.split()[0][:2]}: {counts[cls]}' for cls in classes])
    cv2.putText(frame, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Muestra el frame con los cuadros de los objetos detectados
    cv2.imshow('Object Detection', frame)

    # Presiona 'q' para salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Imprime el conteo final
print(f"Se detectaron {counts} ")

# Libera la cámara y cierra todas las ventanas abiertas
cap.release()
cv2.destroyAllWindows()

# Crear la tabla
table = PrettyTable()
table.field_names = ["Clase", "Cantidad"]
table.align["Clase"] = "l"
table.align["Cantidad"] = "r"
table.padding_width = 1
table.hrules = True
table.vrules = True

# Agregar los datos de conteo a la tabla
for cls, count in counts.items():
    table.add_row([cls, count])

# Crear la ventana y el widget de texto
root = tk.Tk()
text_widget = tk.Text(root, height=10, width=30)
text_widget.pack()

# Agregar los valores de la tabla al widget de texto
text_widget.insert(tk.END, str(table))

# Iniciar el bucle de eventos de tkinter
root.mainloop()