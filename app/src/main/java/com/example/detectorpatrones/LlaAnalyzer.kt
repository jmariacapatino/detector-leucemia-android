package com.example.detectorpatrones

import android.content.Context
import android.util.Log
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import org.tensorflow.lite.Interpreter
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

// ============================================================
//  Resultado de una inferencia individual
// ============================================================
/**
 * Encapsula el resultado completo de procesar un frame de cámara:
 * probabilidades de ambas clases, latencia, etiqueta ganadora
 * y métricas térmicas del dispositivo.
 */
data class InferenceResult(
    /** Probabilidad (0-1) de que sea un blasto leucémico (clase ALL) */
    val allProbability: Float,
    /** Probabilidad (0-1) de que sea un leucocito normal (clase HEM) */
    val hemProbability: Float,
    /** Tiempo de inferencia TFLite en milisegundos */
    val latencyMs: Long,
    /** Etiqueta final de la clase con mayor probabilidad: "ALL" o "HEM" */
    val label: String,
    /** Temperatura CPU en grados Celsius (0.0 si el archivo sysfs no es accesible) */
    val cpuTempC: Double,
    /** Frecuencia actual del núcleo CPU en MHz (0.0 si no es accesible) */
    val cpuFreqMhz: Double
)

// ============================================================
//  ThermalMonitor: lee temperatura y frecuencia CPU
//  desde la interfaz sysfs de Linux/Android
// ============================================================
/**
 * Lee métricas de hardware del sistema de archivos virtual sysfs.
 * No requiere permisos especiales en la mayoría de dispositivos Android.
 */
object ThermalMonitor {

    private const val TAG = "ThermalMonitor"

    // Posibles rutas de temperatura (diferentes fabricantes)
    private val TEMP_PATHS = arrayOf(
        "/sys/class/thermal/thermal_zone0/temp",
        "/sys/class/thermal/thermal_zone1/temp",
        "/sys/devices/virtual/thermal/thermal_zone0/temp"
    )

    // Posibles rutas de frecuencia CPU (núcleo 0)
    private val FREQ_PATHS = arrayOf(
        "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq",
        "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_cur_freq",
        "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq"
    )

    /**
     * Lee temperatura CPU en grados Celsius.
     * sysfs reporta miligrados Celsius → dividir entre 1000.
     * Si ninguna ruta es accesible, devuelve 0.0.
     */
    fun readCpuTemperatureCelsius(): Double {
        for (path in TEMP_PATHS) {
            try {
                val raw = java.io.File(path).readText().trim().toLong()
                return raw / 1000.0
            } catch (e: Exception) {
                // Ignorar y probar la siguiente ruta
            }
        }
        Log.w(TAG, "No se pudo leer temperatura CPU en ninguna ruta conocida")
        return 0.0
    }

    /**
     * Lee frecuencia CPU en MHz.
     * sysfs reporta kHz → dividir entre 1000.
     * Si ninguna ruta es accesible, devuelve 0.0.
     */
    fun readCpuFrequencyMhz(): Double {
        for (path in FREQ_PATHS) {
            try {
                val raw = java.io.File(path).readText().trim().toLong()
                return raw / 1000.0
            } catch (e: Exception) {
                // Ignorar y probar la siguiente ruta
            }
        }
        Log.w(TAG, "No se pudo leer frecuencia CPU en ninguna ruta conocida")
        return 0.0
    }
}

// ============================================================
//  LlaAnalyzer: implementa ImageAnalysis.Analyzer de CameraX
//  Pipeline: YUV→BGR → CLAHE → Blur → HSV segmentación →
//            Morfología → ROI → Resize → TFLite INT8
// ============================================================
class LlaAnalyzer(
    private val context: Context,
    /** Callback invocado en el hilo del analizador con el resultado de inferencia */
    private val onResult: (InferenceResult) -> Unit,
    /** Callback invocado cuando el archivo .tflite no se encuentra en assets */
    private val onModelMissing: () -> Unit
) : ImageAnalysis.Analyzer {

    companion object {
        private const val TAG = "LlaAnalyzer"

        // Nombre del archivo del modelo en la carpeta assets/
        private const val MODEL_FILENAME = "modelo_lla_int8.tflite"

        // Dimensiones de entrada esperadas por el modelo (224×224)
        private const val INPUT_SIZE = 224

        // Índices de clase en el vector de salida del modelo
        private const val CLASS_ALL = 0  // Blasto leucémico (Acute Lymphoblastic Leukemia)
        private const val CLASS_HEM = 1  // Hemoglobina / Leucocito normal

        // ---- Parámetros CLAHE iguales al pipeline de entrenamiento (Colab) ----
        private const val CLAHE_CLIP_LIMIT = 2.0
        private val CLAHE_TILE_SIZE = Size(8.0, 8.0)

        // ---- Rango HSV del color púrpura/violeta (tinción Giemsa/Wright) ----
        // Los blastos leucémicos aparecen en tonos púrpura/azul oscuro
        private val HSV_LOWER = Scalar(120.0, 30.0, 30.0)   // H_min=120, S_min=30, V_min=30
        private val HSV_UPPER = Scalar(170.0, 255.0, 255.0) // H_max=170, S_max=255, V_max=255

        // ---- Tamaño del elemento estructurante para morfología ----
        private val MORPH_KERNEL_SIZE = Size(5.0, 5.0)

        // ---- Padding del 20% alrededor del bounding box del contorno ----
        private const val ROI_PADDING_PERCENT = 0.20
    }

    // Intérprete TFLite; null mientras el modelo no esté disponible
    private var interpreter: Interpreter? = null

    // Buffer de entrada reutilizable: 224×224×3 bytes (UINT8, BGR)
    private val inputBuffer: ByteBuffer = ByteBuffer
        .allocateDirect(INPUT_SIZE * INPUT_SIZE * 3)
        .order(ByteOrder.nativeOrder())

    // Buffer de salida para modelo quantizado INT8 (2 clases)
    private val outputBuffer: ByteBuffer = ByteBuffer
        .allocateDirect(2) // 2 bytes, una por clase
        .order(ByteOrder.nativeOrder())

    init {
        // Intentar cargar el modelo al instanciar el analizador
        loadModel()
    }

    // ----------------------------------------------------------
    //  Carga del modelo TFLite desde la carpeta assets/
    // ----------------------------------------------------------
    private fun loadModel() {
        try {
            val modelBuffer: MappedByteBuffer = loadModelFile()
            val options = Interpreter.Options().apply {
                setNumThreads(4)  // Usar 4 hilos para la inferencia
            }
            interpreter = Interpreter(modelBuffer, options)
            Log.i(TAG, "Modelo '$MODEL_FILENAME' cargado correctamente")
        } catch (e: IOException) {
            // El .tflite aún no existe en assets → notificar a la UI
            Log.w(TAG, "Modelo no encontrado en assets/: $MODEL_FILENAME — ${e.message}")
            interpreter = null
            onModelMissing()
        } catch (e: Exception) {
            Log.e(TAG, "Error inesperado al cargar modelo: ${e.message}", e)
            interpreter = null
            onModelMissing()
        }
    }

    /**
     * Abre el archivo del modelo como MappedByteBuffer para evitar
     * copiar el modelo completo en el heap de Java.
     */
    private fun loadModelFile(): MappedByteBuffer {
        val assetFd = context.assets.openFd(MODEL_FILENAME)
        val inputStream = FileInputStream(assetFd.fileDescriptor)
        val channel: FileChannel = inputStream.channel
        return channel.map(
            FileChannel.MapMode.READ_ONLY,
            assetFd.startOffset,
            assetFd.declaredLength
        )
    }

    // ----------------------------------------------------------
    //  analyze(): punto de entrada del Analyzer de CameraX
    //  Llamado en el executor del ImageAnalysis use case
    // ----------------------------------------------------------
    override fun analyze(imageProxy: ImageProxy) {
        // Si el modelo no está listo, descartar el frame inmediatamente
        if (interpreter == null) {
            imageProxy.close()
            return
        }

        val image = imageProxy.image
        if (image == null) {
            imageProxy.close()
            return
        }

        // Paso 0: Convertir frame YUV_420_888 a Mat BGR de OpenCV
        val bgrMat = yuv420ToBgrMat(imageProxy)

        try {
            // Pasos 1-5: Pipeline de preprocesamiento OpenCV
            val processedMat = preprocessFrame(bgrMat)

            // Llenar el buffer de entrada UINT8
            fillInputBuffer(processedMat)
            processedMat.release()

            // ---- Inferencia TFLite con medición de latencia ----
            // Se captura en val local para evitar el smart-cast concurrente sobre la var
            val startTime = System.currentTimeMillis()
            val tflite = interpreter ?: return
            tflite.run(inputBuffer, outputBuffer)
            val latencyMs = System.currentTimeMillis() - startTime

            // ---- Decodificar salida cuantizada INT8 ----
            // Los bytes están en rango 0-255 (unsigned); Kotlin ByteArray es signed,
            // por eso usamos toUByte() antes de convertir a Float.
            // Leer los dos bytes de salida como valores sin signo (0‑255)
            outputBuffer.rewind()
            val rawAll = outputBuffer.get().toUByte().toFloat()
            val rawHem = outputBuffer.get().toUByte().toFloat()
            val sumRaw = rawAll + rawHem

            // Normalizar para obtener probabilidades que sumen 1
            val probAll = if (sumRaw > 0f) rawAll / sumRaw else 0.5f
            val probHem = if (sumRaw > 0f) rawHem / sumRaw else 0.5f

            // Etiqueta de la clase ganadora
            val label = if (probAll >= probHem) "ALL" else "HEM"

            // ---- Leer métricas térmicas del dispositivo ----
            val cpuTemp = ThermalMonitor.readCpuTemperatureCelsius()
            val cpuFreq = ThermalMonitor.readCpuFrequencyMhz()

            // ---- Notificar resultado a través del callback ----
            onResult(
                InferenceResult(
                    allProbability = probAll,
                    hemProbability = probHem,
                    latencyMs = latencyMs,
                    label = label,
                    cpuTempC = cpuTemp,
                    cpuFreqMhz = cpuFreq
                )
            )

        } catch (e: Exception) {
            Log.e(TAG, "Error durante el análisis del frame: ${e.message}", e)
        } finally {
            // Siempre liberar la Mat y cerrar el frame para no bloquear la cámara
            bgrMat.release()
            imageProxy.close()
        }
    }

    // ----------------------------------------------------------
    //  Conversión YUV_420_888 → Mat BGR
    //  Se leen los planos Y, U, V directamente del ImageProxy
    //  y se construye un buffer NV21 compatible con OpenCV.
    // ----------------------------------------------------------
    private fun yuv420ToBgrMat(imageProxy: ImageProxy): Mat {
        val image = imageProxy.image!!
        val width = image.width
        val height = image.height

        // Obtener los tres planos del formato YUV_420_888
        val planeY = image.planes[0]  // Luminancia
        val planeU = image.planes[1]  // Crominancia U (Cb)
        val planeV = image.planes[2]  // Crominancia V (Cr)

        val bufY = planeY.buffer
        val bufU = planeU.buffer
        val bufV = planeV.buffer

        val ySize = bufY.remaining()
        val uSize = bufU.remaining()
        val vSize = bufV.remaining()

        // Buffer NV21: plano Y completo, luego V y U intercalados
        val nv21 = ByteArray(ySize + uSize + vSize)

        // Copiar plano Y
        bufY.get(nv21, 0, ySize)

        // Leer U y V en arrays separados para intercalar correctamente
        val vArr = ByteArray(vSize)
        val uArr = ByteArray(uSize)
        bufV.get(vArr)
        bufU.get(uArr)

        // Intercalar V,U (formato NV21 = plano Y + plano interleaved V,U)
        var idx = ySize
        val chromaCount = minOf(vSize, uSize)
        for (i in 0 until chromaCount) {
            nv21[idx++] = vArr[i]
            if (idx < nv21.size) nv21[idx++] = uArr[i]
        }

        // Crear Mat con dimensiones NV21 (height * 3/2, width) en escala de grises
        val nv21Mat = Mat(height + height / 2, width, CvType.CV_8UC1)
        nv21Mat.put(0, 0, nv21)

        // Convertir NV21 → BGR (formato de trabajo de OpenCV)
        val bgrMat = Mat()
        Imgproc.cvtColor(nv21Mat, bgrMat, Imgproc.COLOR_YUV2BGR_NV21)
        nv21Mat.release()

        return bgrMat
    }

    // ----------------------------------------------------------
    //  Pipeline de preprocesamiento OpenCV
    //  Mismo orden que el Colab de entrenamiento del modelo
    // ----------------------------------------------------------
    private fun preprocessFrame(bgrMat: Mat): Mat {

        // =====================================================
        // PASO 1: Normalización de color con CLAHE en LAB
        //         Mejora el contraste de la tinción Giemsa
        // =====================================================
        val labMat = Mat()
        Imgproc.cvtColor(bgrMat, labMat, Imgproc.COLOR_BGR2Lab)

        val labChannels = ArrayList<Mat>(3)
        Core.split(labMat, labChannels)

        // CLAHE sobre el canal L* (luminancia); los canales a* y b* no cambian
        val clahe = Imgproc.createCLAHE(CLAHE_CLIP_LIMIT, CLAHE_TILE_SIZE)
        clahe.apply(labChannels[0], labChannels[0])

        Core.merge(labChannels, labMat)
        val normalizedBGR = Mat()
        Imgproc.cvtColor(labMat, normalizedBGR, Imgproc.COLOR_Lab2BGR)

        labMat.release()
        labChannels.forEach { it.release() }

        // =====================================================
        // PASO 2: Suavizado Gaussiano con kernel 3×3
        //         Reduce ruido de alta frecuencia
        // =====================================================
        val blurredMat = Mat()
        Imgproc.GaussianBlur(normalizedBGR, blurredMat, Size(3.0, 3.0), 0.0)
        normalizedBGR.release()

        // =====================================================
        // PASO 3: Segmentación HSV — rango de color púrpura
        //         Aísla el contenido nuclear teñido con Giemsa
        // =====================================================
        val hsvMat = Mat()
        Imgproc.cvtColor(blurredMat, hsvMat, Imgproc.COLOR_BGR2HSV)

        val mask = Mat()
        Core.inRange(hsvMat, HSV_LOWER, HSV_UPPER, mask)
        hsvMat.release()

        // =====================================================
        // PASO 4a: Morfología — CLOSE luego OPEN
        //          CLOSE: rellena huecos dentro de la célula
        //          OPEN: elimina artefactos pequeños del fondo
        // =====================================================
        val kernel = Imgproc.getStructuringElement(
            Imgproc.MORPH_ELLIPSE,
            MORPH_KERNEL_SIZE
        )

        val closedMask = Mat()
        Imgproc.morphologyEx(mask, closedMask, Imgproc.MORPH_CLOSE, kernel)
        mask.release()

        val openedMask = Mat()
        Imgproc.morphologyEx(closedMask, openedMask, Imgproc.MORPH_OPEN, kernel)
        closedMask.release()
        kernel.release()

        // =====================================================
        // PASO 4b: Detectar contorno más grande (célula principal)
        // =====================================================
        val contours = ArrayList<MatOfPoint>()
        val hierarchy = Mat()
        Imgproc.findContours(
            openedMask,
            contours,
            hierarchy,
            Imgproc.RETR_EXTERNAL,
            Imgproc.CHAIN_APPROX_SIMPLE
        )
        openedMask.release()
        hierarchy.release()

        // Calcular ROI con 20% de padding alrededor del contorno más grande
        val roi: Rect = if (contours.isNotEmpty()) {
            val largest = contours.maxByOrNull { Imgproc.contourArea(it) }!!
            val bbox = Imgproc.boundingRect(largest)

            val padX = (bbox.width * ROI_PADDING_PERCENT).toInt()
            val padY = (bbox.height * ROI_PADDING_PERCENT).toInt()

            val x1 = maxOf(0, bbox.x - padX)
            val y1 = maxOf(0, bbox.y - padY)
            val x2 = minOf(blurredMat.cols(), bbox.x + bbox.width + padX)
            val y2 = minOf(blurredMat.rows(), bbox.y + bbox.height + padY)

            Rect(x1, y1, x2 - x1, y2 - y1)
        } else {
            // Sin contorno detectado → imagen completa como fallback
            Log.d(TAG, "Sin contorno válido; se usa la imagen completa como ROI")
            Rect(0, 0, blurredMat.cols(), blurredMat.rows())
        }

        contours.forEach { it.release() }

        // Recortar región de interés de la imagen normalizada+suavizada
        val roiMat = blurredMat.submat(roi).clone()  // clone para independizar de blurredMat
        blurredMat.release()

        // =====================================================
        // PASO 5: Redimensionar a 224×224 (entrada del modelo)
        // =====================================================
        val resizedMat = Mat()
        Imgproc.resize(
            roiMat,
            resizedMat,
            Size(INPUT_SIZE.toDouble(), INPUT_SIZE.toDouble()),
            0.0, 0.0,
            Imgproc.INTER_LINEAR
        )
        roiMat.release()

        return resizedMat
    }

    // ----------------------------------------------------------
    //  Llenar ByteBuffer UINT8 a partir de Mat BGR 224×224
    //  El modelo fue entrenado con valores en rango [0, 255]
    // ----------------------------------------------------------
    private fun fillInputBuffer(mat: Mat) {
        inputBuffer.rewind()

        // Extraer todos los píxeles como array de bytes: [B, G, R, B, G, R, ...]
        val pixels = ByteArray(INPUT_SIZE * INPUT_SIZE * 3)
        mat.get(0, 0, pixels)

        // Copiar directo al buffer (sin normalización flotante — el modelo es UINT8)
        inputBuffer.put(pixels)
        inputBuffer.rewind()
    }

    /**
     * Liberar el intérprete TFLite cuando el analizador ya no se necesite.
     * Llamar desde onDestroy() de la Activity.
     */
    fun release() {
        interpreter?.close()
        interpreter = null
        Log.i(TAG, "Intérprete TFLite liberado correctamente")
    }
}
