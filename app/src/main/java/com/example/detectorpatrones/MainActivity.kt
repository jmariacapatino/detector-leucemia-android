package com.example.detectorpatrones

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import org.opencv.android.OpenCVLoader
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * Actividad principal de la aplicación de detección de LLA.
 * Flujo general:
 *   1. Inicializar OpenCV
 *   2. Solicitar permiso de cámara
 *   3. Configurar CameraX (Preview + ImageAnalysis)
 *   4. Pasar frames a LlaAnalyzer para inferencia TFLite
 *   5. Actualizar la UI con resultados en el hilo principal
 */
class MainActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "MainActivity"
        private const val REQUEST_CODE_CAMERA = 1001
    }

    // ---- Vistas del layout ----
    private lateinit var previewView: PreviewView
    private lateinit var tvResult: TextView
    private lateinit var tvModelStatus: TextView
    private lateinit var tvLatencyTemp: TextView
    private lateinit var progressAll: ProgressBar
    private lateinit var progressHem: ProgressBar
    private lateinit var tvAllLabel: TextView
    private lateinit var tvHemLabel: TextView
    private lateinit var overlayPanel: View

    // ---- CameraX ----
    private lateinit var cameraExecutor: ExecutorService
    private var analyzer: LlaAnalyzer? = null

    // ---- Estado ----
    private var modelReady = false

    // ----------------------------------------------------------
    //  Ciclo de vida
    // ----------------------------------------------------------
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Enlazar vistas
        bindViews()

        // Inicializar OpenCV de forma estática
        if (OpenCVLoader.initLocal()) {
            Log.i(TAG, "OpenCV inicializado correctamente")
        } else {
            Log.e(TAG, "Error al inicializar OpenCV")
            Toast.makeText(this, getString(R.string.opencv_error), Toast.LENGTH_LONG).show()
        }

        // Executor dedicado para el analizador de imágenes (hilo en background)
        cameraExecutor = Executors.newSingleThreadExecutor()

        // Comprobar y solicitar permiso de cámara
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
        // Liberar el intérprete TFLite y detener el executor
        analyzer?.release()
        cameraExecutor.shutdown()
    }

    // ----------------------------------------------------------
    //  Enlace de vistas del layout XML
    // ----------------------------------------------------------
    private fun bindViews() {
        previewView   = findViewById(R.id.previewView)
        tvResult      = findViewById(R.id.tvResult)
        tvModelStatus = findViewById(R.id.tvModelStatus)
        tvLatencyTemp = findViewById(R.id.tvLatencyTemp)
        progressAll   = findViewById(R.id.progressAll)
        progressHem   = findViewById(R.id.progressHem)
        tvAllLabel    = findViewById(R.id.tvAllLabel)
        tvHemLabel    = findViewById(R.id.tvHemLabel)
        overlayPanel  = findViewById(R.id.overlayPanel)
    }

    // ----------------------------------------------------------
    //  Manejo de respuesta al permiso de cámara
    // ----------------------------------------------------------
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

    // ----------------------------------------------------------
    //  Verificar permiso de cámara
    // ----------------------------------------------------------
    private fun hasCameraPermission(): Boolean {
        return ContextCompat.checkSelfPermission(
            this,
            Manifest.permission.CAMERA
        ) == PackageManager.PERMISSION_GRANTED
    }

    // ----------------------------------------------------------
    //  Inicializar CameraX con Preview + ImageAnalysis
    // ----------------------------------------------------------
    private fun startCamera() {
        // Obtener el futuro del proveedor de cámara
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            // ---- Caso de uso Preview: muestra el visor en tiempo real ----
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }

            // ---- Instanciar el analizador LLA con sus callbacks ----
            analyzer = LlaAnalyzer(
                context = this,
                onResult = { result -> updateUiWithResult(result) },
                onModelMissing = { showModelPendingState() }
            )

            // ---- Caso de uso ImageAnalysis ----
            val imageAnalysis = ImageAnalysis.Builder()
                // Formato RGBA_8888 es el más compatible para procesamiento manual de píxeles
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                // Descartar frames anteriores si el analizador no terminó aún
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor, analyzer!!)
                }

            // Usar siempre la cámara trasera
            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                // Desvincular cualquier uso previo antes de reasignar
                cameraProvider.unbindAll()

                // Vincular Preview + ImageAnalysis al ciclo de vida de esta Activity
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

    // ----------------------------------------------------------
    //  Actualizar la interfaz con el resultado de la inferencia
    //  SIEMPRE se ejecuta en el hilo principal (Main thread)
    // ----------------------------------------------------------
    private fun updateUiWithResult(result: InferenceResult) {
        runOnUiThread {
            // ---- Texto principal de resultado ----
            if (result.label == "ALL") {
                // Blasto detectado → texto rojo de alerta
                tvResult.text = getString(R.string.result_blasto)
                tvResult.setTextColor(ContextCompat.getColor(this, R.color.colorBlastoDetected))
            } else {
                // Leucocito normal → texto verde
                tvResult.text = getString(R.string.result_normal)
                tvResult.setTextColor(ContextCompat.getColor(this, R.color.colorNormal))
            }

            // ---- Barras de probabilidad (0–100) ----
            val allPercent = (result.allProbability * 100).toInt().coerceIn(0, 100)
            val hemPercent = (result.hemProbability * 100).toInt().coerceIn(0, 100)
            progressAll.progress = allPercent
            progressHem.progress = hemPercent

            // ---- Etiquetas con porcentaje exacto ----
            tvAllLabel.text = getString(R.string.label_all_prob, allPercent)
            tvHemLabel.text = getString(R.string.label_hem_prob, hemPercent)

            // ---- Latencia y temperatura ----
            val latencyText = "Latencia: ${result.latencyMs}ms"
            val tempText = if (result.cpuTempC > 0.0) {
                "CPU Temp: %.1f°C".format(result.cpuTempC)
            } else {
                "CPU Temp: N/D"
            }
            tvLatencyTemp.text = "$latencyText | $tempText"

            // ---- Estado del modelo (listo) ----
            if (!modelReady) {
                modelReady = true
                tvModelStatus.text = getString(R.string.model_ready)
                tvModelStatus.setTextColor(
                    ContextCompat.getColor(this, R.color.colorNormal)
                )
            }
        }
    }

    // ----------------------------------------------------------
    //  Mostrar estado "modelo pendiente" cuando el .tflite
    //  no se encuentra en assets/
    // ----------------------------------------------------------
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
            Log.w(TAG, "Modelo TFLite no disponible; mostrando estado pendiente")
        }
    }
}
