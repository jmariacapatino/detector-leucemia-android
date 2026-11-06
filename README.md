# Detector de Indicios de Leucemia Linfoblástica Aguda (LLA) en Android (MVP v2.0)

[![Platform](https://img.shields.io/badge/Platform-Android-green.svg)](https://developer.android.com/)
[![Kotlin](https://img.shields.io/badge/Language-Kotlin-blue.svg)](https://kotlinlang.org/)
[![TensorFlow Lite](https://img.shields.io/badge/Edge_AI-TFLite_INT8-orange.svg)](https://www.tensorflow.org/lite)
[![FastAPI](https://img.shields.io/badge/Cloud_API-FastAPI_FP32-009688.svg)](https://fastapi.tiangolo.com/)
[![OpenCV](https://img.shields.io/badge/Vision-OpenCV_4.12-red.svg)](https://opencv.org/)

Prototipo funcional (MVP v2.0) de aplicación móvil Android para el apoyo al cribado preliminar de **Leucemia Linfoblástica Aguda (LLA)** en imágenes microscópicas de frotis de sangre periférica, implementando una arquitectura experimental dual para evaluar comparativamente la inferencia local en el borde (*Edge AI*) frente a la inferencia remota cliente-servidor.

Desarrollado como parte del trabajo de investigación de pregrado en **Ingeniería de Software en la Universidad San Ignacio de Loyola (USIL)** por **José Manuel Mariaca Patiño**.

---

## Descripción General

El sistema permite capturar o evaluar láminas citológicas de células sanguíneas (linfoblastos malignos `ALL` frente a células hematopoyéticas sanas `HEM`) utilizando una red neuronal convolucional **EfficientNet-B0** entrenada mediante aprendizaje por transferencia sobre el conjunto de datos clínico público **C-NMC 2019** (*ISBI Challenge*).

Para responder a los objetivos de investigación sin introducir sesgos de interfaz o de hardware, la aplicación integra ambos paradigmas de despliegue (Variable Independiente $V_1$) dentro de un mismo cliente Android, manteniendo constantes el preprocesamiento con **OpenCV** y el recolector de telemetría de rendimiento (`UnifiedPerformanceModule`).

### Funcionalidades Principales del MVP v2.0

1. **Conmutador de Paradigma de Inferencia ($V_1$):**
   - **Modo Local (Edge AI - TFLite INT8):** Ejecuta el modelo cuantizado (`modelo_lla_int8.tflite`, $4.67\text{ MB}$) directamente en el procesador ARM del teléfono sin requerir conexión a internet ni transmitir datos fuera del dispositivo.
   - **Modo Remoto (Cliente-Servidor - API REST FP32):** Envía el recorte celular en memoria mediante una petición `HTTP POST` multiparte hacia el servidor **FastAPI / Uvicorn** (`server/servidor_remoto_fp32.py`), el cual ejecuta el modelo original en precisión completa de 32 bits (`FP32`). Una pulsación larga sobre el botón superior permite configurar la dirección IP del servidor en red local.
2. **Triple Flujo de Entrada de Muestras:**
   - **Muestra C-NMC (`[ Muestra C-NMC ]`):** Carga secuencialmente 20 imágenes clínicas reales de validación del dataset *C-NMC 2019* (10 casos `ALL` y 10 casos `HEM` intercalados en `app/src/main/assets/dataset_cnmc/`) y verifica automáticamente si la predicción coincide con la etiqueta real (`ACIERTO` / `ERROR`).
   - **Galería (`[ Galería ]`):** Permite analizar láminas almacenadas en la memoria del dispositivo mediante el selector nativo de Android.
   - **Cámara en Vivo (`[ Cámara ]`):** Flujo continuo con `CameraX` (`STRATEGY_KEEP_ONLY_LATEST`) para evaluar muestras directamente a través del ocular del microscopio.
3. **Módulo Unificado de Telemetría (`UnifiedPerformanceModule`):**
   - Registra de forma atómica tras cada inferencia los 4 indicadores técnicos de rendimiento del Instrumento 2: **Latencia total de respuesta ($\text{ms}$)**, **Consumo de memoria RAM nativa y Java Heap ($\text{MB}$)**, **Temperatura del dispositivo ($^\circ\text{C}$)** y **Frecuencia del procesador ($\text{MHz}$)**.
4. **Modo de Pantalla Completa Inmersiva:**
   - Ocultamiento automático de las barras de navegación del sistema (`WindowInsetsControllerCompat`) para evitar solapamientos con el panel inferior de controles y telemetría.

---

## Arquitectura del Flujo de Procesamiento

```text
[ Entrada: Cámara en Vivo (CameraX) | Muestra C-NMC (Assets) | Galería ]
                                   │
                                   ▼
         1. Conversión a Matriz OpenCV y Normalización CLAHE (LAB)
                                   │
                                   ▼
         2. Suavizado Gaussiano (3x3) y Segmentación HSV (Tinción Giemsa)
                                   │
                                   ▼
         3. Operaciones Morfológicas y Extracción de Región de Interés (ROI)
                                   │
                                   ▼
         4. Redimensionamiento Bilineal a 224x224 px (RGB)
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
     [ MODO LOCAL: Edge AI ]       [ MODO REMOTO: API REST ]
     Intérprete TFLite INT8        Petición HTTP POST /predict
     en dispositivo móvil          hacia servidor FastAPI FP32
                    │                             │
                    └──────────────┬──────────────┘
                                   ▼
         5. Clasificación (ALL vs HEM) + UnifiedPerformanceModule
            (Latencia ms | RAM MB | Temperatura °C | CPU MHz)
```

---

## Estructura del Proyecto

- `app/src/main/java/com/example/detectorpatrones/`
  - `MainActivity.kt`: Controlador principal de interfaz, modo inmersivo, selector de paradigma Local/Remoto y flujos de entrada (Cámara, Galería y Muestra C-NMC).
  - `LlaAnalyzer.kt`: Pipeline de preprocesamiento con OpenCV, inferencia local con TensorFlow Lite INT8, cliente HTTP para inferencia remota FP32 y clase `UnifiedPerformanceModule`.
- `app/src/main/res/layout/activity_main.xml`: Diseño de interfaz oscura con visor de cámara/muestra, tarjeta de diagnóstico y panel de control.
- `app/src/main/assets/`
  - `modelo_lla_int8.tflite`: Modelo `EfficientNet-B0` cuantizado a 8 bits (`INT8`).
  - `dataset_cnmc/`: 20 imágenes citológicas reales de validación del dataset *C-NMC 2019*.
- `server/servidor_remoto_fp32.py`: Servidor backend en Python con **FastAPI** y **TensorFlow** para ejecutar el paradigma de inferencia remota en precisión `FP32`.

---

## Requisitos e Instalación

### 1. Aplicación Android (Cliente Móvil)
- **Requisitos:** Android Studio Ladybug o superior, JDK 17, dispositivo físico Android (API 24+).
- **Compilación:**
  1. Clonar el repositorio:
     ```bash
     git clone https://github.com/jmariacapatino/detector-leucemia-android.git
     ```
  2. Abrir el proyecto en Android Studio y sincronizar Gradle.
  3. Ejecutar en el dispositivo físico (`Run 'app'`) o generar el APK mediante `./gradlew assembleDebug`.

### 2. Servidor de Inferencia Remota (Paradigma Cliente-Servidor FP32)
- **Requisitos:** Python 3.10+, TensorFlow, FastAPI, Uvicorn, Pillow, NumPy, Python-Multipart.
- **Ejecución en red local:**
  ```bash
  pip install fastapi uvicorn tensorflow pillow numpy python-multipart
  python server/servidor_remoto_fp32.py
  ```
- En la aplicación Android, mantener presionado el botón superior de modo para ingresar la dirección IP local de la computadora (`http://<IP_LOCAL>:8000/predict`).

---

## Autor

**José Manuel Mariaca Patiño**  
- Carrera: Ingeniería de Software — Universidad San Ignacio de Loyola (USIL), Lima, Perú  
- Contacto: jmariacapatino@gmail.com
