#Importación de librerías
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Cargar el modelo preentrenado
modelo_base = load_model('cifar10.h5')

#Congelar capas para conservar características
for capa in modelo_base.layers[:-5]:
    capa.trainable = False

#Ajustar las últimas capas del modelo
clases = 4

x = modelo_base.layers[-2].output                                               #Conectar a la penúltima capa del modelo base
x = Dense(256, activation='relu', name='custom_dense_stl10')(x)                 #Nombre único para la primera capa densa
salida = Dense(clases, activation='softmax', name='custom_output_stl10')(x)     #Nombre único para la capa de salida
modelo = Model(inputs=modelo_base.input, outputs=salida)

#Cargar datos desde el directorio
ruta = '4Clases/'
imagenes = []
etiquetas = []

#Recorrer las clases dentro del directorio principal
for etiqueta in range(clases):
    ruta_clase = os.path.join(ruta, f'class_{etiqueta}')
    #Obtener lista de imágenes en cada clase
    nombres_imagenes = os.listdir(ruta_clase)
    #Completar las rutas de las imágenes
    rutas_imagenes = [os.path.join(ruta_clase, nombre_imagen) for nombre_imagen in nombres_imagenes if nombre_imagen.endswith('.jpg')]
    #Agregar a la lista de todas las imágenes y etiquetas
    imagenes.extend(rutas_imagenes)
    etiquetas.extend([etiqueta] * len(rutas_imagenes))

#Crear conjuntos de datos
tamano_lote = 32
altura = 32
ancho = 32

imagenes_entrenamiento, imagenes_prueba, etiquetas_entrenamiento, etiquetas_prueba = train_test_split(
    imagenes, etiquetas, test_size=0.3, random_state=123)

imagenes_validacion, imagenes_prueba, etiquetas_validacion, etiquetas_prueba = train_test_split(
    imagenes_prueba, etiquetas_prueba, test_size=0.5, random_state=123)

#Procesamiento de imágenes
def preprocess_image(ruta_imagen, etiqueta):
    imagen = tf.io.read_file(ruta_imagen)
    imagen = tf.image.decode_jpeg(imagen, channels=3)
    imagen = tf.image.resize(imagen, size=(altura, ancho))
    imagen = tf.cast(imagen, tf.float32) / 255.0  # Normalizar a [0,1]
    return imagen, etiqueta

#Crear datasets de entrenamiento, validación y prueba
conjunto_entrenamiento = tf.data.Dataset.from_tensor_slices((imagenes_entrenamiento, etiquetas_entrenamiento))
conjunto_entrenamiento = conjunto_entrenamiento.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
conjunto_entrenamiento = conjunto_entrenamiento.batch(tamano_lote).prefetch(tf.data.AUTOTUNE)

conjunto_validacion = tf.data.Dataset.from_tensor_slices((imagenes_validacion, etiquetas_validacion))
conjunto_validacion = conjunto_validacion.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
conjunto_validacion = conjunto_validacion.batch(tamano_lote).prefetch(tf.data.AUTOTUNE)

conjunto_prueba = tf.data.Dataset.from_tensor_slices((imagenes_prueba, etiquetas_prueba))
conjunto_prueba = conjunto_prueba.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
conjunto_prueba = conjunto_prueba.batch(tamano_lote).prefetch(tf.data.AUTOTUNE)

#Calcular la media y la desviación estándar de los datos de entrenamiento
media = 0
desviacion_estandar = 0
cantidad_lotes = 0

for imagenes, _ in conjunto_entrenamiento:
    media_lote = tf.reduce_mean(imagenes, axis=[0, 1, 2])
    desviacion_estandar_lote = tf.math.reduce_std(imagenes, axis=[0, 1, 2])
    media += media_lote
    desviacion_estandar += desviacion_estandar_lote
    cantidad_lotes += 1

media /= cantidad_lotes
desviacion_estandar /= cantidad_lotes

#Normalizar los datos usando z-score y codificar las etiquetas
def normalize_and_encode_img(imagen, etiqueta):
    imagen = (imagen - media) / (desviacion_estandar + 1e-7)
    etiqueta = tf.one_hot(etiqueta, clases)
    return imagen, etiqueta

conjunto_entrenamiento = conjunto_entrenamiento.map(normalize_and_encode_img)
conjunto_validacion = conjunto_validacion.map(normalize_and_encode_img)
conjunto_prueba = conjunto_prueba.map(normalize_and_encode_img)

#Compilar el modelo
modelo.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#Definir callbacks
detencion_temprana = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduccion_tasa_aprendizaje = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

#Entrenar el modelo
historial = modelo.fit(
    conjunto_entrenamiento,
    validation_data=conjunto_validacion,
    epochs=100,
    callbacks=[detencion_temprana, reduccion_tasa_aprendizaje]
)

#Evaluar el modelo
perdida, precision = modelo.evaluate(conjunto_prueba)
print(f"Pérdida: {perdida}")
print(f"Precisión: {precision}")

# Predicciones y etiquetas verdaderas para el conjunto de prueba
probabilidades = modelo.predict(conjunto_prueba)
predicciones = np.argmax(probabilidades, axis=1)
valores_reales = np.concatenate([etiqueta for _, etiqueta in conjunto_prueba], axis=0)
valores_reales = np.argmax(valores_reales, axis=1)  # Convertir y_true a su forma original

# Calcular precisión global del nuevo modelo usando sklearn
precision_global = accuracy_score(valores_reales, predicciones)
print("Precisión global del nuevo modelo:", precision_global)

#Guardar el modelo entrenado
modelo.save('modelo.h5')