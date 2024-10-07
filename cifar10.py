#Importación de librerías
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

#Cargar el dataset CIFAR-10
(imagenes_entrenamiento, etiquetas_entrenamiento), (imagenes_prueba, etiquetas_prueba) = cifar10.load_data()
print("Datos de entrenamiento: ", len(imagenes_entrenamiento))
print("Datos de validación: ", len(imagenes_prueba))

#Procesamiento de datos
print("\nValor original de pixel: ", imagenes_entrenamiento[0][0][0][0])
media = np.mean(imagenes_entrenamiento, axis=(0, 1, 2, 3))
desviacion_estandar = np.std(imagenes_entrenamiento, axis=(0, 1, 2, 3))
imagenes_entrenamiento = (imagenes_entrenamiento - media) / (desviacion_estandar + 1e-7)
imagenes_prueba = (imagenes_prueba - media) / (desviacion_estandar + 1e-7)
print("Valor procesado de pixel: ", imagenes_entrenamiento[0][0][0][0])

#Aumento de datos
aumento_datos = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
)
aumento_datos.fit(imagenes_entrenamiento)

#Convertir etiquetas a formato categórico
clases = 10

print("\nEtiqueta original: ", etiquetas_entrenamiento[0][0])
etiquetas_entrenamiento = tf.keras.utils.to_categorical(etiquetas_entrenamiento, clases)
etiquetas_prueba = tf.keras.utils.to_categorical(etiquetas_prueba, clases)
print("Etiqueta categórica: ", etiquetas_entrenamiento[0])

#Construcción del modelo
modelo = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4), input_shape=imagenes_entrenamiento.shape[1:]),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.2),

    Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.3),

    Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    BatchNormalization(),
    Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.4),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(clases, activation='softmax')
])

#Compilación del modelo
modelo.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#Configuración de callbacks
detencion_temprana = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduccion_tasa_aprendizaje = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

#Entrenar el modelo
tamano_lote = 64
historial = modelo.fit(aumento_datos.flow(imagenes_entrenamiento, etiquetas_entrenamiento, batch_size=tamano_lote),
                    steps_per_epoch=imagenes_entrenamiento.shape[0] // tamano_lote,
                    epochs=5,
                    validation_data=(imagenes_prueba, etiquetas_prueba),
                    callbacks=[detencion_temprana, reduccion_tasa_aprendizaje])

#Evaluación del modelo
perdida, precision = modelo.evaluate(imagenes_prueba, etiquetas_prueba)
print(f"Pérdida: {perdida}")
print(f"Precisión: {precision}")

#Guardar el modelo entrenado
modelo.save('cifar10.h5')

