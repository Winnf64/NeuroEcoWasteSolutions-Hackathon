# Importación de librerías
import os
import numpy as np
import time
import serial
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

categorias = ['vidrio', 'carton', 'metal', 'plastico']

# Importación del modelo
modelo = load_model('modelo.h5')

# Prueba con imagen
# Carga y procesamiento de la imagen
ruta_imagen = '4Clases/class_0/brown-glass1.jpg'
imagen = image.load_img(ruta_imagen, target_size=(32, 32))
imagen_arreglo = image.img_to_array(imagen)
imagen_arreglo = np.expand_dims(imagen_arreglo, axis=0)
imagen_arreglo /= 255.0

# Predicción
prediccion = modelo.predict(imagen_arreglo)
categoria = np.argmax(prediccion)

print("Probabilidades por categoría: ")
for categoria, probabilidad in enumerate(prediccion):
    print(f"{categorias[categoria]}: {probabilidad}")

print(f'\nLa categoría es: {categorias[categoria]}[{categoria}] con una probabilidad de {np.max(prediccion):.5f}')

# Visión por computadora

# Crear las carpetas de clasificación si no existen
for categoria in categorias:
    if not os.path.exists(categoria):
        os.makedirs(categoria)

# Obtener el tiempo actual (en milisegundos)
def tiempo():
    return int(round(time.time() * 1000))

# Conexión conexión con el puerto serial
COM = 'COM13'
BAUD = '115200'
ser = serial.Serial(COM, BAUD)

# Registro de imágenes
contador_imagenes = {categoria: 0 for categoria in categorias}

# Umbral de probabilidad máxima
umbral_probabilidad = 0.9

# Obtener el tiempo actual
ultima_clasificacion = tiempo()

# Inicializar la captura de video (0 para la cámara por defecto)
camara = cv2.VideoCapture(0)

# Proceso de clasificación
while True:

    # Comprobar si hay datos en espera
    if ser.in_waiting == 0:
        continue

    # Leer el dato del puerto serial
    dato = ser.readline().decode('utf-8').rstrip()
    print(dato)

    if dato != "1":
        continue

    # Conectar con la cámara
    retorno, frame = camara.read()

    # Verificar si el frame fue capturado correctamente
    if not retorno:
        break

    # Preprocesar el frame
    imagen = cv2.resize(frame, (32, 32))                # Ajustar el tamaño según lo que espera tu modelo
    imagen_arreglo = np.expand_dims(imagen, axis=0)     # Expandir las dimensiones
    imagen_arreglo = imagen_arreglo / 255.0             # Normalizar la imagen si es necesario

    # Comprobar si han pasado 5 segundos desde la última clasificación
    if (tiempo() - ultima_clasificacion) > 5000:

        # Hacer la predicción
        prediccion = modelo.predict(imagen_arreglo)
        categoria = np.argmax(prediccion)
        probabilidad_maxima = np.max(prediccion)
        ultima_clasificacion = tiempo()

        nombre_categoria = categorias[categoria]

        # Verificar si la probabilidad máxima supera el umbral
        if probabilidad_maxima >= umbral_probabilidad:
            # Guardar la imagen en la carpeta correspondiente
            contador_imagenes[nombre_categoria] += 1
            nombre_imagen = f"{nombre_categoria}_{contador_imagenes[nombre_categoria]}.png"
            ruta_imagen = os.path.join(nombre_categoria, nombre_imagen)
            # Guardar la imagen original (frame)
            cv2.imwrite(ruta_imagen, frame)

        ser.write(f'%{categoria};\n'.encode())

    cv2.putText(frame, f'Categoria: {categorias[categoria]}[{categoria}]', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Precisión: {probabilidad_maxima}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('NeuroEcoWaste Solutions', frame)
    print(f'Predicción: {prediccion}')

    # Romper el loop con la tecla 'q' o el num 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camara.release()
cv2.destroyAllWindows()
