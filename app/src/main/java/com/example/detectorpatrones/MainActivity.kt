package com.example.detectorpatrones

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.EditText
import android.widget.ImageView
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
import org.opencv.android.OpenCVLoader
import java.io.File
import java.io.FileWriter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * Prototipo unificado para evaluar los 2 niveles de la Variable Independiente:
 *   - Modo Local: Inferencia en dispositivo Android con TensorFlow Lite INT8 + OpenCV (Offline)
 *   - Modo Remoto: Procesamiento cliente-servidor vía API REST FastAPI FP32 (Online)
 *
 * Incluye además 3 modos de entrada para pruebas y exposición:
 *   1. Muestras reales integradas del dataset C-NMC 2019 (20 células ALL y HEM de validación)
 *   2. Carga de imágenes desde la Galería del teléfono
 *   3. Cámara en vivo en tiempo real
 */
class MainActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "MainActivity"
        private const val REQUEST_CODE_CAMERA = 1001
        private const val CNMC_ASSETS_FOLDER = "dataset_cnmc"
    }

    // ---- Vistas del layout ----
    private lateinit var previewView: PreviewView
    private lateinit var ivStaticPreview: ImageView
    private lateinit var tvSampleInfo: TextView
    private lateinit var btnToggleParadigm: Button
    private lateinit var btnNextCnmcSample: Button
    private lateinit var btnPickGallery: Button
    private lateinit var btnLiveCamera: Button
    private lateinit var tvResult: TextView
    private lateinit var tvModelStatus: TextView
    private lateinit var tvLatencyTemp: TextView
    private lateinit var progressAll: ProgressBar
    private lateinit var progressHem: ProgressBar
    private lateinit var tvAllLabel: TextView
    private lateinit var tvHemLabel: TextView
    private lateinit var overlayPanel: View

    // ---- CameraX y Ejecutores ----
    private lateinit var cameraExecutor: ExecutorService
    private var analyzer: LlaAnalyzer? = null

    // ---- Estado de muestras C-NMC 2019 e imagen actual ----
    private var cnmcAssetFiles: List<String> = emptyList()
    private var currentCnmcIndex: Int = -1
    private var currentStaticBitmap: Bitmap? = null
    private var currentSampleId: String = "CAMARA_VIVO"
    private var currentGroundTruth: String = "-"

    // ---- Archivos CSV de los 2 Instrumentos ----
    private var modelReady = false
    private var sampleCounter = 0
    private var diagMatrixFile: File? = null
    private var perfMatrixFile: File? = null
    private var telemetryFile: File? = null
    private val dateFormat = SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS", Locale.US)

    // Selector de imagen de la Galería del celular
    private val pickImageLauncher = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        if (uri != null) {
            loadGalleryImage(uri)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Activar modo inmersivo (oculta los 3 botones inferiores de navegación de Android)
        enableImmersiveFullscreen()

        bindViews()

        if (OpenCVLoader.initLocal()) {
            Log.i(TAG, "OpenCV inicializado correctamente")
        } else {
            Log.e(TAG, "Error al inicializar OpenCV")
            Toast.makeText(this, getString(R.string.opencv_error), Toast.LENGTH_LONG).show()
        }

        cameraExecutor = Executors.newSingleThreadExecutor()

        // Cargar lista de las 20 imágenes reales C-NMC 2019 incluidas en el APK
        cnmcAssetFiles = try {
            assets.list(CNMC_ASSETS_FOLDER)?.sorted() ?: emptyList()
        } catch (e: Exception) {
            emptyList()
        }

        initInstrumentMatricesCsv()
        setupButtons()

        if (hasCameraPermission()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.CAMERA),
                REQUEST_CODE_CAMERA
            )
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        analyzer?.release()
        cameraExecutor.shutdown()
    }

    override fun onWindowFocusChanged(hasFocus: Boolean) {
        super.onWindowFocusChanged(hasFocus)
        if (hasFocus) {
            enableImmersiveFullscreen()
        }
    }

    /**
     * Oculta la barra de navegación inferior (triángulo, círculo, cuadrado)
     * y la barra de estado superior en modo inmersivo (se muestran solo si deslizas el borde).
     */
    private fun enableImmersiveFullscreen() {
        WindowCompat.setDecorFitsSystemWindows(window, false)
        val controller = WindowInsetsControllerCompat(window, window.decorView)
        controller.hide(WindowInsetsCompat.Type.systemBars())
        controller.systemBarsBehavior =
            WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
    }

    private fun bindViews() {
        previewView       = findViewById(R.id.previewView)
        ivStaticPreview   = findViewById(R.id.ivStaticPreview)
        tvSampleInfo      = findViewById(R.id.tvSampleInfo)
        btnToggleParadigm = findViewById(R.id.btnToggleParadigm)
        btnNextCnmcSample = findViewById(R.id.btnNextCnmcSample)
        btnPickGallery    = findViewById(R.id.btnPickGallery)
        btnLiveCamera     = findViewById(R.id.btnLiveCamera)
        tvResult          = findViewById(R.id.tvResult)
        tvModelStatus     = findViewById(R.id.tvModelStatus)
        tvLatencyTemp     = findViewById(R.id.tvLatencyTemp)
        progressAll       = findViewById(R.id.progressAll)
        progressHem       = findViewById(R.id.progressHem)
        tvAllLabel        = findViewById(R.id.tvAllLabel)
        tvHemLabel        = findViewById(R.id.tvHemLabel)
        overlayPanel      = findViewById(R.id.overlayPanel)
    }

    private fun setupButtons() {
        // 1. Alternar entre Modo Local (TFLite INT8) y Modo Remoto (API REST FP32)
        btnToggleParadigm.setOnClickListener {
            val lla = analyzer ?: return@setOnClickListener
            lla.useRemoteParadigm = !lla.useRemoteParadigm
            updateParadigmUi(lla.useRemoteParadigm)

            // Si hay una muestra C-NMC o de Galería en pantalla, re-evaluarla de inmediato con el nuevo paradigma
            val bmp = currentStaticBitmap
            if (lla.isPausedForStaticImage && bmp != null) {
                evaluateStaticImageAsync(bmp, currentSampleId, currentGroundTruth)
            }
        }

        // Mantener presionado el botón de paradigma permite editar la IP del servidor FastAPI
        btnToggleParadigm.setOnLongClickListener {
            showServerUrlDialog()
            true
        }

        // 2. Probar siguiente muestra real del dataset C-NMC 2019 incluida en el APK
        btnNextCnmcSample.setOnClickListener {
            loadNextCnmcDatasetSample()
        }

        // 3. Cargar cualquier imagen desde la Galería del celular
        btnPickGallery.setOnClickListener {
            pickImageLauncher.launch("image/*")
        }

        // 4. Volver a Cámara en Vivo en tiempo real
        btnLiveCamera.setOnClickListener {
            switchToLiveCameraMode()
        }
    }

    private fun updateParadigmUi(isRemote: Boolean) {
        if (isRemote) {
            btnToggleParadigm.text = "VI: MODO REMOTO (API REST FP32)"
            btnToggleParadigm.setBackgroundColor(0xFF7C3AED.toInt())
            tvModelStatus.text = "● API REST FP32"
            Toast.makeText(
                this,
                "Modo Remoto activo (Mantén presionado el botón morado si deseas cambiar la IP del servidor)",
                Toast.LENGTH_SHORT
            ).show()
        } else {
            btnToggleParadigm.text = "VI: MODO LOCAL (TFLite INT8)"
            btnToggleParadigm.setBackgroundColor(0xFF1E40AF.toInt())
            tvModelStatus.text = getString(R.string.model_ready)
        }
    }

    private fun showServerUrlDialog() {
        val lla = analyzer ?: return
        val input = EditText(this).apply {
            setText(lla.remoteServerUrl)
            setSelection(text.length)
        }
        AlertDialog.Builder(this)
            .setTitle("Endpoint Servidor Remoto (FastAPI FP32)")
            .setMessage("Ingresa la URL del servidor FastAPI en tu laptop (ej. http://192.168.18.10:8000/predict):")
            .setView(input)
            .setPositiveButton("Guardar") { _, _ ->
                val newUrl = input.text.toString().trim()
                if (newUrl.startsWith("http")) {
                    lla.remoteServerUrl = newUrl
                    Toast.makeText(this, "URL actualizada: $newUrl", Toast.LENGTH_SHORT).show()
                    val bmp = currentStaticBitmap
                    if (lla.isPausedForStaticImage && lla.useRemoteParadigm && bmp != null) {
                        evaluateStaticImageAsync(bmp, currentSampleId, currentGroundTruth)
                    }
                }
            }
            .setNegativeButton("Cancelar", null)
            .show()
    }

    private fun loadNextCnmcDatasetSample() {
        if (cnmcAssetFiles.isEmpty()) {
            Toast.makeText(this, "No se encontraron muestras C-NMC en assets", Toast.LENGTH_SHORT).show()
            return
        }
        val lla = analyzer ?: return
        lla.isPausedForStaticImage = true

        currentCnmcIndex = (currentCnmcIndex + 1) % cnmcAssetFiles.size
        val fileName = cnmcAssetFiles[currentCnmcIndex]
        val groundTruth = if (fileName.contains("_ALL_")) "ALL" else "HEM"

        try {
            assets.open("$CNMC_ASSETS_FOLDER/$fileName").use { stream ->
                val bmp = BitmapFactory.decodeStream(stream)
                if (bmp != null) {
                    currentStaticBitmap = bmp
                    currentSampleId = fileName
                    currentGroundTruth = groundTruth

                    ivStaticPreview.setImageBitmap(bmp)
                    ivStaticPreview.visibility = View.VISIBLE

                    val gtDesc = if (groundTruth == "ALL") "ALL (Blasto leucémico)" else "HEM (Célula normal)"
                    tvSampleInfo.text = String.format(
                        Locale.US,
                        "Muestra C-NMC #%02d/%02d | Etiqueta real: %s",
                        currentCnmcIndex + 1,
                        cnmcAssetFiles.size,
                        gtDesc
                    )

                    evaluateStaticImageAsync(bmp, fileName, groundTruth)
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error cargando muestra C-NMC: ${e.message}", e)
        }
    }

    private fun loadGalleryImage(uri: Uri) {
        val lla = analyzer ?: return
        lla.isPausedForStaticImage = true
        try {
            contentResolver.openInputStream(uri)?.use { stream ->
                val bmp = BitmapFactory.decodeStream(stream)
                if (bmp != null) {
                    currentStaticBitmap = bmp
                    currentSampleId = "GALERIA_${System.currentTimeMillis() % 10000}"
                    currentGroundTruth = "GALERIA"

                    ivStaticPreview.setImageBitmap(bmp)
                    ivStaticPreview.visibility = View.VISIBLE
                    tvSampleInfo.text = "Imagen cargada desde la Galería del dispositivo"

                    evaluateStaticImageAsync(bmp, currentSampleId, currentGroundTruth)
                }
            }
        } catch (e: Exception) {
            Toast.makeText(this, "No se pudo abrir la imagen seleccionada", Toast.LENGTH_SHORT).show()
        }
    }

    private fun switchToLiveCameraMode() {
        val lla = analyzer ?: return
        currentStaticBitmap = null
        currentSampleId = "CAMARA_VIVO"
        currentGroundTruth = "-"
        ivStaticPreview.visibility = View.GONE
        tvSampleInfo.text = "Modo Cámara en Vivo ( inferencia continua en tiempo real )"
        lla.isPausedForStaticImage = false
    }

    private fun evaluateStaticImageAsync(bmp: Bitmap, sampleId: String, groundTruth: String) {
        val lla = analyzer ?: return
        cameraExecutor.execute {
            val result = lla.evaluateStaticBitmap(bmp, sampleId, groundTruth)
            if (result != null) {
                updateUiWithResult(result)
            }
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_CAMERA) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                startCamera()
            } else {
                Toast.makeText(
                    this,
                    getString(R.string.camera_permission_denied),
                    Toast.LENGTH_LONG
                ).show()
            }
        }
    }

    private fun hasCameraPermission(): Boolean {
        return ContextCompat.checkSelfPermission(
            this,
            Manifest.permission.CAMERA
        ) == PackageManager.PERMISSION_GRANTED
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }

            analyzer = LlaAnalyzer(
                context = this,
                onResult = { result -> updateUiWithResult(result) },
                onModelMissing = { showModelPendingState() }
            )

            val imageAnalysis = ImageAnalysis.Builder()
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor, analyzer!!)
                }

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this,
                    cameraSelector,
                    preview,
                    imageAnalysis
                )
                Log.i(TAG, "CameraX vinculado correctamente")
            } catch (e: Exception) {
                Log.e(TAG, "Error al vincular CameraX: ${e.message}", e)
                Toast.makeText(
                    this,
                    getString(R.string.camera_error),
                    Toast.LENGTH_LONG
                ).show()
            }

        }, ContextCompat.getMainExecutor(this))
    }

    private fun updateUiWithResult(result: InferenceResult) {
        runOnUiThread {
            if (result.errorMessage != null) {
                tvResult.text = "SERVIDOR REMOTO OFF"
                tvResult.setTextColor(ContextCompat.getColor(this, R.color.colorBlastoDetected))
                tvLatencyTemp.text = result.errorMessage
                return@runOnUiThread
            }

            sampleCounter++

            // Verificar si coincide con la etiqueta clínica real cuando se prueba una muestra C-NMC 2019
            val matchSuffix = if (result.groundTruthLabel == "ALL" || result.groundTruthLabel == "HEM") {
                if (result.label == result.groundTruthLabel) " ( ACIERTO ✓ )" else " ( ERROR ✗ )"
            } else {
                ""
            }

            if (result.label == "ALL") {
                tvResult.text = getString(R.string.result_blasto) + matchSuffix
                tvResult.setTextColor(ContextCompat.getColor(this, R.color.colorBlastoDetected))
            } else {
                tvResult.text = getString(R.string.result_normal) + matchSuffix
                tvResult.setTextColor(ContextCompat.getColor(this, R.color.colorNormal))
            }

            val allPercent = (result.allProbability * 100).toInt().coerceIn(0, 100)
            val hemPercent = (result.hemProbability * 100).toInt().coerceIn(0, 100)
            progressAll.progress = allPercent
            progressHem.progress = hemPercent

            tvAllLabel.text = String.format(Locale.US, "ALL: %.1f%%", result.allProbability * 100f)
            tvHemLabel.text = String.format(Locale.US, "HEM: %.1f%%", result.hemProbability * 100f)

            val latencyText = "Lat: ${result.latencyMs}ms"
            val ramText = "RAM: %.1fMB".format(Locale.US, result.ramUsageMb)
            val tempText = if (result.cpuTempC > 0.0) {
                "Temp: %.1f°C".format(Locale.US, result.cpuTempC)
            } else {
                "Temp: N/D"
            }
            val freqText = if (result.cpuFreqMhz > 0.0) {
                "CPU: %.0fMHz".format(Locale.US, result.cpuFreqMhz)
            } else {
                "CPU: ARM"
            }
            tvLatencyTemp.text = "$latencyText | $ramText | $tempText | $freqText"

            appendInstrumentRows(sampleCounter, result)

            if (!modelReady) {
                modelReady = true
                tvModelStatus.text = getString(R.string.model_ready)
                tvModelStatus.setTextColor(
                    ContextCompat.getColor(this, R.color.colorNormal)
                )
            }
        }
    }

    private fun showModelPendingState() {
        runOnUiThread {
            modelReady = false
            tvModelStatus.text = getString(R.string.model_pending)
            tvModelStatus.setTextColor(
                ContextCompat.getColor(this, R.color.colorBlastoDetected)
            )
            tvResult.text = getString(R.string.waiting_model)
            tvResult.setTextColor(
                ContextCompat.getColor(this, R.color.colorTextSecondary)
            )
        }
    }

    private fun initInstrumentMatricesCsv() {
        try {
            val dir = getExternalFilesDir(null) ?: filesDir

            // Instrumento 1: Matriz de registro de precisión diagnóstica
            val fDiag = File(dir, "matriz_precision_diagnostica.csv")
            if (!fDiag.exists()) {
                FileWriter(fDiag, true).use { writer ->
                    writer.append("nro_muestra,id_imagen,etiqueta_real,paradigma,prob_all,prob_hem,clase_predicha\n")
                }
            }
            diagMatrixFile = fDiag

            // Instrumento 2: Matriz de registro de rendimiento técnico
            val fPerf = File(dir, "matriz_rendimiento_tecnico.csv")
            if (!fPerf.exists()) {
                FileWriter(fPerf, true).use { writer ->
                    writer.append("nro_muestra,marca_temporal,id_imagen,paradigma,latencia_ms,memoria_ram_mb,temperatura_cpu_c,frecuencia_cpu_mhz,thermal_throttling\n")
                }
            }
            perfMatrixFile = fPerf

            // Consolidado general
            val fAll = File(dir, "telemetria_edge_int8.csv")
            if (!fAll.exists()) {
                FileWriter(fAll, true).use { writer ->
                    writer.append("timestamp,paradigma,sample_id,ground_truth,predicted_label,prob_all,prob_hem,latency_ms,ram_mb,cpu_temp_c,cpu_freq_mhz,thermal_throttling\n")
                }
            }
            telemetryFile = fAll
        } catch (e: Exception) {
            Log.e(TAG, "No se pudieron inicializar las matrices CSV: ${e.message}")
        }
    }

    private fun appendInstrumentRows(sampleNum: Int, result: InferenceResult) {
        val ts = dateFormat.format(Date())
        val throttlingStr = if (result.thermalThrottling) "Si" else "No"

        diagMatrixFile?.let { file ->
            try {
                val row = String.format(
                    Locale.US,
                    "%04d,%s,%s,%s,%.4f,%.4f,%s\n",
                    sampleNum,
                    result.sampleId,
                    result.groundTruthLabel,
                    result.paradigmName,
                    result.allProbability,
                    result.hemProbability,
                    result.label
                )
                FileWriter(file, true).use { it.append(row) }
            } catch (e: Exception) {
                Log.w(TAG, "Error en matriz_precision_diagnostica.csv: ${e.message}")
            }
        }

        perfMatrixFile?.let { file ->
            try {
                val row = String.format(
                    Locale.US,
                    "%04d,%s,%s,%s,%d,%.2f,%.2f,%.1f,%s\n",
                    sampleNum,
                    ts,
                    result.sampleId,
                    result.paradigmName,
                    result.latencyMs,
                    result.ramUsageMb,
                    result.cpuTempC,
                    result.cpuFreqMhz,
                    throttlingStr
                )
                FileWriter(file, true).use { it.append(row) }
            } catch (e: Exception) {
                Log.w(TAG, "Error en matriz_rendimiento_tecnico.csv: ${e.message}")
            }
        }

        telemetryFile?.let { file ->
            try {
                val row = String.format(
                    Locale.US,
                    "%s,%s,%s,%s,%s,%.4f,%.4f,%d,%.2f,%.2f,%.1f,%s\n",
                    ts,
                    result.paradigmName,
                    result.sampleId,
                    result.groundTruthLabel,
                    result.label,
                    result.allProbability,
                    result.hemProbability,
                    result.latencyMs,
                    result.ramUsageMb,
                    result.cpuTempC,
                    result.cpuFreqMhz,
                    throttlingStr
                )
                FileWriter(file, true).use { it.append(row) }
            } catch (e: Exception) {
                Log.w(TAG, "Error en telemetria_edge_int8.csv: ${e.message}")
            }
        }
    }
}
