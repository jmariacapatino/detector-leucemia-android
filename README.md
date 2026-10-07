# 🔬 Detector de Indicios de Leucemia Linfoblástica Aguda (LLA) — Android

[![Platform](https://img.shields.io/badge/Platform-Android-green.svg)](https://developer.android.com/)
[![Kotlin](https://img.shields.io/badge/Language-Kotlin-blue.svg)](https://kotlinlang.org/)
[![TensorFlow Lite](https://img.shields.io/badge/ML-TensorFlow%20Lite%20INT8-orange.svg)](https://www.tensorflow.org/lite)
[![OpenCV](https://img.shields.io/badge/Vision-OpenCV%204.12-red.svg)](https://opencv.org/)

Prototipo de aplicación móvil para la asistencia en el tamizaje preliminar de **Leucemia Linfoblástica Aguda (LLA)** a partir de imágenes microscópicas de frotis de sangre periférica. 

Desarrollado como proyecto de tesis de pregrado en **Ingeniería de Software — Universidad San Ignacio de Loyola (USIL)**.

---

##  Descripción del Proyecto

El sistema captura imágenes de muestras sanguíneas en tiempo real mediante **CameraX**, procesa los leucocitos a través de un pipeline automatizado con **OpenCV** y clasifica las células mediante una red neuronal convolucional (**MobileNetV2**) optimizada para inferencia en el borde (*Edge AI*) con **TensorFlow Lite**.

### Características Principales
- **Procesamiento de imagen en el dispositivo (*Edge Computing*):** Todo el análisis se realiza localmente sin requerir conexión a internet ni servidores externos.
- **Pipeline de Visión Computacional Robusto:** Normalización de contraste y variación de tinción entre laboratorios usando espacio de color LAB y ecualización adaptativa (CLAHE).
- **Segmentación celular:** Aislamiento del leucocito mediante filtrado morfológico y segmentación en espacio de color HSV (tinción Giemsa).
- **Inferencia Cuantizada:** Red MobileNetV2 cuantizada a enteros de 8 bits (**TFLite INT8**), permitiendo baja latencia y mínimo consumo energético en procesadores móviles ARM.
- **Monitoreo de Telemetría Térmica y Rendimiento:** Lectura en tiempo real de temperatura CPU y frecuencia de reloj vía `sysfs` para análisis de viabilidad térmica en despliegue clínico.

---

## Arquitectura del Pipeline de Visión e Inferencia

```text
[ Entrada de Cámara (CameraX / YUV_420_888) ]
                     │
                     ▼
       Conversión a Matriz OpenCV (BGR)
                     │
                     ▼
  1. Normalización de Color (Espacio LAB + CLAHE en canal L)
                     │
                     ▼
  2. Reducción de Ruido (Filtro Gaussiano 3x3)
                     │
                     ▼
  3. Segmentación de Tinción Giemsa (Rango Púrpura en HSV)
                     │
                     ▼
  4. Operaciones Morfológicas (Closing + Opening elíptico)
                     │
                     ▼
  5. Detección de Contorno y Extracción de Región de Interés (ROI)
                     │
                     ▼
  6. Redimensionamiento a 224x224 px y Normalización [0, 1]
                     │
                     ▼
[ Inferencia TFLite INT8 (MobileNetV2 Transfer Learning) ]
                     │
                     ▼
  Resultados: ALL (Blasto) vs. HEM (Normal) + Latencia + Temp CPU
```

---

## Tecnologías y Herramientas

- **Lenguaje:** Kotlin 1.9+
- **Framework Móvil:** Android SDK (Min SDK: 24, Target SDK: 34+)
- **Captura de Cámara:** Android Jetpack CameraX 1.3.0
- **Visión Computacional:** OpenCV Android SDK 4.12.0
- **Machine Learning Runtime:** TensorFlow Lite 2.16.1
- **Dataset de Entrenamiento:** ALL-IDB / C-NMC 2019 (*Leukemia Challenge Dataset*)
- **Entorno de Entrenamiento:** Python 3.10, TensorFlow/Keras, Google Colab GPU

---

## Requisitos e Instalación

### Requisitos Previos
- Android Studio Ladybug / Koala o superior
- JDK 17
- Dispositivo Android físico con cámara (API 24+)

### Pasos para compilar
1. Clonar el repositorio:
   ```bash
   git clone https://github.com/TU-USUARIO/detector-leucemia-android.git
   ```
2. Abrir el proyecto en **Android Studio**.
3. Sincronizar las dependencias con **Gradle**.
4. Colocar el modelo cuantizado en `app/src/main/assets/modelo_lla_int8.tflite` (si no está incluido).
5. Conectar tu dispositivo y hacer clic en **Run** (`Shift + F10`).

---

## Autor

**José Manuel Mariaca**  
- Carrera: Ingeniería de Software — Universidad San Ignacio de Loyola (USIL)  
- LinkedIn: [linkedin.com/in/josemariaca](https://www.linkedin.com/in/josemariaca/)  
- Email: jmariacapatino@gmail.com
