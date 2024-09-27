import cv2
from ultralytics import YOLO
import requests
import os

# Cargar el modelo YOLOv8s (más preciso que nano)
model = YOLO('yolov10s.pt')

# Imprimir las etiquetas que reconoce el modelo
print("Etiquetas reconocidas por el modelo:", model.names)

# URL de la cámara IP
rtsp_url = "rtsp://admin:administrador123@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0"

# Abrir la captura de video desde la cámara IP
cap = cv2.VideoCapture(rtsp_url)

# Verificar si la cámara está abierta correctamente
if not cap.isOpened():
    print("Error al abrir la cámara IP")
    exit()

# Configuración del bot de Telegram
telegram_token = '7277805155:AAFkN5vw4-V66I2QPv2L7YmnC_Yqpb2OZD8'
chat_id = '1522836228'

# Variable para controlar si ya se ha enviado la alerta
alert_sent = False

# Función para enviar notificación y screenshot por Telegram
def send_telegram_alert(label, frame):
    global alert_sent
    if not alert_sent:
        screenshot_path = "screenshot.jpg"
        cv2.imwrite(screenshot_path, frame)  # Guardar la imagen capturada

        url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
        message = f"¡Alerta! Se ha detectado un {label}."

        # Enviar el mensaje
        data = {
            "chat_id": chat_id,
            "text": message
        }
        response = requests.post(url, data=data)

        # Enviar la imagen
        url_photo = f"https://api.telegram.org/bot{telegram_token}/sendPhoto"
        with open(screenshot_path, 'rb') as photo:
            files = {'photo': photo}
            data = {'chat_id': chat_id}
            response_photo = requests.post(url_photo, data=data, files=files)

        if response.status_code == 200 and response_photo.status_code == 200:
            print("Mensaje y captura enviados exitosamente a Telegram")
            alert_sent = True
        else:
            print(f"Error al enviar mensaje o captura: {response.status_code} - {response.text}")

        os.remove(screenshot_path)

# Bucle para la captura continua de video
while True:
    ret, frame = cap.read()

    if not ret:
        print("No se pudo recibir el frame. Finalizando...")
        break

    # Realizar la predicción con YOLOv8 y reducir la confianza a 0.2
    results = model.predict(frame, conf=0.2)

    # Dibujar las detecciones
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            cls = box.cls[0]
            conf = box.conf[0]
            label = model.names[int(cls)]

            # Filtrar solo armas (pistolas y cuchillos)
            if label in ['guns', 'knife', 'pistols']:  # Ajustar según las etiquetas correctas
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Enviar alerta a Telegram (solo una vez)
                send_telegram_alert(label, frame)

                # Imprimir alerta en la consola
                print(f"¡Alerta de {label} detectada con {conf:.2f} confianza!")

    # Mostrar el video con las detecciones
    cv2.imshow('Detección de Armas', frame)

    # Presionar 'q' para salir del bucle
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la captura de video y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
