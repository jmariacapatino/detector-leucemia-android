package com.example.detectorpatrones

import androidx.annotation.OptIn
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer

class PatronAnalyzer(private val tfliteInterpreter: Interpreter) : ImageAnalysis.Analyzer {

    // Aca metodo se ejecuta automáticamente por cada frame de la cámara
    @OptIn(ExperimentalGetImage::class)
    override fun analyze(imageProxy: ImageProxy) {
        val mediaImage = imageProxy.image
        if (mediaImage != null) {


            // Aca conviertes el 'mediaImage' (formato YUV) a una matriz 'Mat' de OpenCV (formato RGB)
            var frameOpenCV = convertirImageProxyAMat(imageProxy)

            // Pre-procesamiento con OpenCV
            val frameLimpio = Mat()
            Imgproc.cvtColor(frameOpenCV, frameLimpio, Imgproc.COLOR_RGB2GRAY)
            // Redimensionar al tamaño de la Red Neuronal (ej. 224x224)

            // Imgproc.resize(frameLimpio, frameLimpio, Size(224.0, 224.0))

            // Preparar datos para TensorFlow Lite
            // TFLite no lee 'Mat' de OpenCV directamente. para eso el 'ByteBuffer' o 'TensorImage'.
            val inputBuffer = convertirMatATensor(frameLimpio)

            // El cerebro
            // Creamos un arreglo vacío para recibir la respuesta del modelo (prob verdadera y prob falsa)
            val outputArray = Array(1) { FloatArray(2) }

            tfliteInterpreter.run(inputBuffer, outputArray)

            // Aca se lee el resultado y enviarlo a la pantalla
            val probabilidad = outputArray[0][1] // Suponiendo que el índice 1 es "Patrón Encontrado"
            if (probabilidad > 0.80) { // Si hay más del 80% de seguridad
                println("¡Patrón detectado con éxito!")
                // Aquí mas adelante llamar a una función para actualizar la interfaz gráfica
            }


            frameOpenCV.release()
            frameLimpio.release()
        }


        imageProxy.close()
    }


    private fun convertirImageProxyAMat(image: ImageProxy): Mat {
        // Lógica para convertir formato Android a matriz matemática OpenCV

        return Mat()
    }

    private fun convertirMatATensor(mat: Mat): ByteBuffer {
        // Lógica para convertir matriz OpenCV al formato ligero que pide TFLite

        return ByteBuffer.allocate(0)
    }
}