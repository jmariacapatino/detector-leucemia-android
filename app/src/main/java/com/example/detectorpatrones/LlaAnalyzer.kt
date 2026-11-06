package com.example.detectorpatrones

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.graphics.Bitmap
import android.os.BatteryManager
import android.os.Debug
import android.util.Log
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import org.json.JSONObject
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import org.tensorflow.lite.Interpreter
import java.io.ByteArrayOutputStream
import java.io.DataOutputStream
import java.io.File
import java.io.FileInputStream
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

// ============================================================
//  Resultado unificado para los 2 Instrumentos de Recolección:
//  1. Matriz de registro de precisión diagnóstica
//  2. Matriz de registro de rendimiento técnico
// ============================================================
data class InferenceResult(
    val paradigmName: String,
    val sampleId: String,
    val groundTruthLabel: String,
    val allProbability: Float,
    val hemProbability: Float,
    val label: String,
    val latencyMs: Long,
    val ramUsageMb: Double,
    val cpuTempC: Double,
    val cpuFreqMhz: Double,
    val thermalThrottling: Boolean = false,
    val errorMessage: String? = null
)

// ============================================================
//  UnifiedPerformanceModule:
//  Módulo unificado en Kotlin para el Instrumento 2
//  (Matriz de registro de rendimiento técnico):
//  Latencia (ms), Memoria RAM (MB), Temperatura (°C) y Frecuencia CPU (MHz)
// ============================================================
object UnifiedPerformanceModule {

    private const val TAG = "UnifiedPerfModule"
    private const val THERMAL_THROTTLING_THRESHOLD_C = 45.0

    private val TEMP_PATHS = arrayOf(
        "/sys/class/thermal/thermal_zone0/temp",
        "/sys/class/thermal/thermal_zone1/temp",
        "/sys/class/thermal/thermal_zone2/temp",
        "/sys/class/thermal/thermal_zone3/temp",
        "/sys/devices/virtual/thermal/thermal_zone0/temp"
    )

    fun startTimerNs(): Long = System.nanoTime()

    fun measureLatencyMs(startTimeNs: Long): Long {
        return ((System.nanoTime() - startTimeNs) / 1_000_000L).coerceAtLeast(1L)
    }

    fun readRamUsageMb(): Double {
        return try {
            val runtime = Runtime.getRuntime()
            val javaHeapBytes = runtime.totalMemory() - runtime.freeMemory()
            val nativeHeapBytes = Debug.getNativeHeapAllocatedSize()
            (javaHeapBytes + nativeHeapBytes) / (1024.0 * 1024.0)
        } catch (e: Exception) {
            0.0
        }
    }

    fun readTemperatureCelsius(context: Context): Double {
        for (path in TEMP_PATHS) {
            try {
                val f = File(path)
                if (f.canRead()) {
                    val raw = f.readText().trim().toDouble()
                    val tempC = if (raw > 1000.0) raw / 1000.0 else if (raw > 100.0) raw / 10.0 else raw
                    if (tempC in 15.0..95.0) {
                        return tempC
                    }
                }
            } catch (_: Exception) {
            }
        }
        return try {
            val intent = context.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
            val tempTenths = intent?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, -1) ?: -1
            if (tempTenths > 0) tempTenths / 10.0 else 0.0
        } catch (e: Exception) {
            Log.w(TAG, "No se pudo leer sensor térmico: ${e.message}")
            0.0
        }
    }

    fun readCpuFrequencyMhz(): Double {
        for (core in 0..7) {
            val paths = arrayOf(
                "/sys/devices/system/cpu/cpu$core/cpufreq/scaling_cur_freq",
                "/sys/devices/system/cpu/cpu$core/cpufreq/cpuinfo_cur_freq",
                "/sys/devices/system/cpu/cpu$core/cpufreq/cpuinfo_max_freq"
            )
            for (path in paths) {
                try {
                    val f = File(path)
                    if (f.canRead()) {
                        val rawKhz = f.readText().trim().toLong()
                        if (rawKhz > 0) {
                            return rawKhz / 1000.0
                        }
                    }
                } catch (_: Exception) {
                }
            }
        }
        return 0.0
    }

    fun isThermalThrottling(tempC: Double): Boolean {
        return tempC >= THERMAL_THROTTLING_THRESHOLD_C
    }
}

// ============================================================
//  LlaAnalyzer: Soporta AMBOS paradigmas de despliegue:
//   - Modo Local: Preprocesamiento OpenCV + TensorFlow Lite INT8
//   - Modo Remoto: Cliente HTTP Multipart -> FastAPI REST FP32
//  Y soporta tanto frames de Cámara en Vivo como imágenes estáticas
//  del dataset C-NMC 2019 o de la Galería del teléfono.
// ============================================================
class LlaAnalyzer(
    private val context: Context,
    private val onResult: (InferenceResult) -> Unit,
    private val onModelMissing: () -> Unit
) : ImageAnalysis.Analyzer {

    companion object {
        private const val TAG = "LlaAnalyzer"
        private const val MODEL_FILENAME = "modelo_lla_int8.tflite"
        private const val INPUT_SIZE = 224

        private const val CLAHE_CLIP_LIMIT = 2.0
        private val CLAHE_TILE_SIZE = Size(8.0, 8.0)

        // Rango HSV de célula completa (núcleo + citoplasma sobre fondo oscuro)
        private val HSV_LOWER = Scalar(0.0, 15.0, 20.0)
        private val HSV_UPPER = Scalar(180.0, 255.0, 255.0)

        private val MORPH_KERNEL_SIZE = Size(5.0, 5.0)
        private const val ROI_PADDING_PERCENT = 0.20
    }

    @Volatile
    var isPausedForStaticImage: Boolean = false

    @Volatile
    var useRemoteParadigm: Boolean = false

    @Volatile
    var remoteServerUrl: String = "http://192.168.18.10:8000/predict"

    private var interpreter: Interpreter? = null
    private val inferenceLock = Any()

    private val inputBuffer: ByteBuffer = ByteBuffer
        .allocateDirect(INPUT_SIZE * INPUT_SIZE * 3)
        .order(ByteOrder.nativeOrder())

    private val outputBuffer: ByteBuffer = ByteBuffer
        .allocateDirect(2)
        .order(ByteOrder.nativeOrder())

    private var lastRemoteFrameTimeMs = 0L

    init {
        loadModel()
    }

    private fun loadModel() {
        try {
            val modelBuffer: MappedByteBuffer = loadModelFile()
            val options = Interpreter.Options().apply {
                setNumThreads(4)
            }
            interpreter = Interpreter(modelBuffer, options)
            Log.i(TAG, "Modelo '$MODEL_FILENAME' cargado correctamente")
        } catch (e: IOException) {
            Log.w(TAG, "Modelo no encontrado en assets/: $MODEL_FILENAME")
            interpreter = null
            onModelMissing()
        } catch (e: Exception) {
            Log.e(TAG, "Error inesperado al cargar modelo: ${e.message}", e)
            interpreter = null
            onModelMissing()
        }
    }

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

    override fun analyze(imageProxy: ImageProxy) {
        if (isPausedForStaticImage) {
            imageProxy.close()
            return
        }

        if (!useRemoteParadigm && interpreter == null) {
            imageProxy.close()
            return
        }

        try {
            // En Modo Remoto por cámara en vivo, limitar a ~2 peticiones/seg para no saturar red
            if (useRemoteParadigm) {
                val now = System.currentTimeMillis()
                if (now - lastRemoteFrameTimeMs < 500L) {
                    imageProxy.close()
                    return
                }
                lastRemoteFrameTimeMs = now
            }

            val bitmap = imageProxy.toBitmap()
            val result = evaluateBitmapInternal(
                bitmap = bitmap,
                sampleId = "CAMARA_VIVO",
                groundTruth = "-"
            )
            if (result != null && !isPausedForStaticImage) {
                onResult(result)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error analizando frame de cámara: ${e.message}", e)
        } finally {
            imageProxy.close()
        }
    }

    /**
     * Evalúa cualquier imagen estática (Muestra C-NMC 2019 o imagen de Galería)
     * bajo el paradigma actualmente seleccionado (Modo Local INT8 o Modo Remoto API REST FP32).
     */
    fun evaluateStaticBitmap(
        bitmap: Bitmap,
        sampleId: String,
        groundTruth: String
    ): InferenceResult? {
        return evaluateBitmapInternal(bitmap, sampleId, groundTruth)
    }

    private fun evaluateBitmapInternal(
        bitmap: Bitmap,
        sampleId: String,
        groundTruth: String
    ): InferenceResult? {
        return if (useRemoteParadigm) {
            runRemoteApiRestInference(bitmap, sampleId, groundTruth)
        } else {
            runLocalTfliteInt8Inference(bitmap, sampleId, groundTruth)
        }
    }

    private fun runLocalTfliteInt8Inference(
        bitmap: Bitmap,
        sampleId: String,
        groundTruth: String
    ): InferenceResult? {
        synchronized(inferenceLock) {
            val tflite = interpreter ?: return null
            val rgbaMat = Mat()
            val bgrMat = Mat()
            try {
                val startTimeNs = UnifiedPerformanceModule.startTimerNs()

                Utils.bitmapToMat(bitmap, rgbaMat)
                Imgproc.cvtColor(rgbaMat, bgrMat, Imgproc.COLOR_RGBA2BGR)

                val processedRgbMat = preprocessFrame(bgrMat)
                fillInputBuffer(processedRgbMat)
                processedRgbMat.release()

                // CRÍTICO: Rebobinar TANTO inputBuffer como outputBuffer antes de cada inferencia
                inputBuffer.rewind()
                outputBuffer.rewind()

                tflite.run(inputBuffer, outputBuffer)

                val latencyMs = UnifiedPerformanceModule.measureLatencyMs(startTimeNs)
                val ramMb = UnifiedPerformanceModule.readRamUsageMb()
                val cpuTemp = UnifiedPerformanceModule.readTemperatureCelsius(context)
                val cpuFreq = UnifiedPerformanceModule.readCpuFrequencyMhz()
                val throttling = UnifiedPerformanceModule.isThermalThrottling(cpuTemp)

                outputBuffer.rewind()
                val rawAll = outputBuffer.get().toUByte().toFloat()
                val rawHem = outputBuffer.get().toUByte().toFloat()
                val sumRaw = rawAll + rawHem

                val probAll = if (sumRaw > 0f) rawAll / sumRaw else 0.5f
                val probHem = if (sumRaw > 0f) rawHem / sumRaw else 0.5f
                val label = if (probAll >= probHem) "ALL" else "HEM"

                return InferenceResult(
                    paradigmName = "Modo_Local_INT8",
                    sampleId = sampleId,
                    groundTruthLabel = groundTruth,
                    allProbability = probAll,
                    hemProbability = probHem,
                    label = label,
                    latencyMs = latencyMs,
                    ramUsageMb = ramMb,
                    cpuTempC = cpuTemp,
                    cpuFreqMhz = cpuFreq,
                    thermalThrottling = throttling
                )
            } catch (e: Exception) {
                Log.e(TAG, "Error en inferencia local INT8: ${e.message}", e)
                return null
            } finally {
                rgbaMat.release()
                bgrMat.release()
            }
        }
    }

    private fun runRemoteApiRestInference(
        bitmap: Bitmap,
        sampleId: String,
        groundTruth: String
    ): InferenceResult {
        val startTimeNs = UnifiedPerformanceModule.startTimerNs()
        val boundary = "----LLABoundary" + System.currentTimeMillis()
        var conn: HttpURLConnection? = null
        try {
            val baos = ByteArrayOutputStream()
            bitmap.compress(Bitmap.CompressFormat.PNG, 100, baos)
            val imageBytes = baos.toByteArray()

            val url = URL(remoteServerUrl)
            conn = (url.openConnection() as HttpURLConnection).apply {
                connectTimeout = 5000
                readTimeout = 8000
                doInput = true
                doOutput = true
                useCaches = false
                requestMethod = "POST"
                setRequestProperty("Content-Type", "multipart/form-data; boundary=$boundary")
            }

            DataOutputStream(conn.outputStream).use { dos ->
                dos.writeBytes("--$boundary\r\n")
                dos.writeBytes("Content-Disposition: form-data; name=\"file\"; filename=\"$sampleId.png\"\r\n")
                dos.writeBytes("Content-Type: image/png\r\n\r\n")
                dos.write(imageBytes)
                dos.writeBytes("\r\n--$boundary--\r\n")
                dos.flush()
            }

            val code = conn.responseCode
            if (code == HttpURLConnection.HTTP_OK) {
                val respText = conn.inputStream.bufferedReader().use { it.readText() }
                val json = JSONObject(respText)
                val probAll = json.optDouble("allProbability", 0.5).toFloat()
                val probHem = json.optDouble("hemProbability", 0.5).toFloat()
                val label = json.optString("label", if (probAll >= probHem) "ALL" else "HEM")

                val latencyMs = UnifiedPerformanceModule.measureLatencyMs(startTimeNs)
                val ramMb = UnifiedPerformanceModule.readRamUsageMb()
                val cpuTemp = UnifiedPerformanceModule.readTemperatureCelsius(context)
                val cpuFreq = UnifiedPerformanceModule.readCpuFrequencyMhz()
                val throttling = UnifiedPerformanceModule.isThermalThrottling(cpuTemp)

                return InferenceResult(
                    paradigmName = "Modo_Remoto_FP32",
                    sampleId = sampleId,
                    groundTruthLabel = groundTruth,
                    allProbability = probAll,
                    hemProbability = probHem,
                    label = label,
                    latencyMs = latencyMs,
                    ramUsageMb = ramMb,
                    cpuTempC = cpuTemp,
                    cpuFreqMhz = cpuFreq,
                    thermalThrottling = throttling
                )
            } else {
                throw IOException("HTTP $code")
            }
        } catch (e: Exception) {
            val latencyMs = UnifiedPerformanceModule.measureLatencyMs(startTimeNs)
            val ramMb = UnifiedPerformanceModule.readRamUsageMb()
            val cpuTemp = UnifiedPerformanceModule.readTemperatureCelsius(context)
            val cpuFreq = UnifiedPerformanceModule.readCpuFrequencyMhz()
            return InferenceResult(
                paradigmName = "Modo_Remoto_FP32",
                sampleId = sampleId,
                groundTruthLabel = groundTruth,
                allProbability = 0f,
                hemProbability = 0f,
                label = "ERROR_RED",
                latencyMs = latencyMs,
                ramUsageMb = ramMb,
                cpuTempC = cpuTemp,
                cpuFreqMhz = cpuFreq,
                thermalThrottling = false,
                errorMessage = "Sin conexión con $remoteServerUrl (${e.javaClass.simpleName})"
            )
        } finally {
            conn?.disconnect()
        }
    }

    private fun preprocessFrame(bgrMat: Mat): Mat {
        val labMat = Mat()
        Imgproc.cvtColor(bgrMat, labMat, Imgproc.COLOR_BGR2Lab)

        val labChannels = ArrayList<Mat>(3)
        Core.split(labMat, labChannels)

        val clahe = Imgproc.createCLAHE(CLAHE_CLIP_LIMIT, CLAHE_TILE_SIZE)
        clahe.apply(labChannels[0], labChannels[0])

        Core.merge(labChannels, labMat)
        val normalizedBGR = Mat()
        Imgproc.cvtColor(labMat, normalizedBGR, Imgproc.COLOR_Lab2BGR)

        labMat.release()
        labChannels.forEach { it.release() }

        val blurredMat = Mat()
        Imgproc.GaussianBlur(normalizedBGR, blurredMat, Size(3.0, 3.0), 0.0)
        normalizedBGR.release()

        val hsvMat = Mat()
        Imgproc.cvtColor(blurredMat, hsvMat, Imgproc.COLOR_BGR2HSV)

        val mask = Mat()
        Core.inRange(hsvMat, HSV_LOWER, HSV_UPPER, mask)
        hsvMat.release()

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
            Rect(0, 0, blurredMat.cols(), blurredMat.rows())
        }

        contours.forEach { it.release() }

        val roiMat = blurredMat.submat(roi).clone()
        blurredMat.release()

        val resizedMat = Mat()
        Imgproc.resize(
            roiMat,
            resizedMat,
            Size(INPUT_SIZE.toDouble(), INPUT_SIZE.toDouble()),
            0.0, 0.0,
            Imgproc.INTER_LINEAR
        )
        roiMat.release()

        val rgbMat = Mat()
        Imgproc.cvtColor(resizedMat, rgbMat, Imgproc.COLOR_BGR2RGB)
        resizedMat.release()

        return rgbMat
    }

    private fun fillInputBuffer(mat: Mat) {
        inputBuffer.rewind()
        val pixels = ByteArray(INPUT_SIZE * INPUT_SIZE * 3)
        mat.get(0, 0, pixels)
        inputBuffer.put(pixels)
        inputBuffer.rewind()
    }

    fun release() {
        interpreter?.close()
        interpreter = null
        Log.i(TAG, "Intérprete TFLite liberado correctamente")
    }
}
