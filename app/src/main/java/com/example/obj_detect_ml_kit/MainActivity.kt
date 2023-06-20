package com.example.obj_detect_ml_kit_custom

import android.content.ContentValues.TAG
import android.graphics.ImageDecoder.ImageInfo
import android.net.Uri
import android.os.Bundle
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import android.graphics.*
import android.util.Log
import com.google.mlkit.common.model.LocalModel
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.objects.DetectedObject
import com.google.mlkit.vision.objects.ObjectDetection
import com.google.mlkit.vision.objects.custom.CustomObjectDetectorOptions


class MainActivity : AppCompatActivity() {

    private lateinit var loadImageButton: Button
    private lateinit var imageView: ImageView
    private lateinit var selectedImage: Bitmap
    private lateinit var resultTextView: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        imageView = findViewById(R.id.image_view)
        resultTextView = findViewById(R.id.result_textview)
        loadImageButton = findViewById(R.id.load_image_button)

        loadImageButton.setOnClickListener {
            pickImage.launch("image/*")
        }

        imageView.setOnClickListener {
            runMLKITObjectDetection(selectedImage)
        }

    }
    private val pickImage =
        registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
            uri?.let {
                "To detect objects, click the image".also { resultTextView.text = it } // Clear text
                val source = ImageDecoder.createSource(contentResolver, uri)
                selectedImage = ImageDecoder.decodeBitmap(source)
                //ImageDecoder.decodeBitmap(source)
                { imageDecoder: ImageDecoder, imageInfo: ImageInfo?, source1: ImageDecoder.Source? ->
                    imageDecoder.isMutableRequired = true
                }
                imageView.setImageBitmap(selectedImage)
            }
        }

    // ML Kit Object Detection function
    private fun runMLKITObjectDetection(bitmap: Bitmap) {
        //val inputBitmap = addDimensionToBitmap(bitmap) // Add extra dimension to input bitmap
        //val image = InputImage.fromBitmap(inputBitmap, 0)
        val image = InputImage.fromBitmap(bitmap, 0)

        //val image = InputImage.fromBitmap( Bitmap.createBitmap(IntArray(100 * 100), 100, 100, Bitmap.Config.ARGB_8888),0)

        val localModel = LocalModel.Builder()
            .setAssetFilePath("object_labeler.tflite")
            // or .setAbsoluteFilePath(absolute file path to model file)
            // or .setUri(URI to model file)
            .build()

    // Multiple object detection in static images
        val customObjectDetectorOptions =
            CustomObjectDetectorOptions.Builder(localModel)
                .setDetectorMode(CustomObjectDetectorOptions.SINGLE_IMAGE_MODE)
                .enableMultipleObjects()
                .enableClassification()
                .setClassificationConfidenceThreshold(0.5f)
                .setMaxPerObjectLabelCount(3)
                .build()

        val objectDetector =
            ObjectDetection.getClient(customObjectDetectorOptions)
        objectDetector.process(image)

        objectDetector.process(image).addOnSuccessListener { results ->
            print_results(results)

            // Parse ML Kit's DetectedObject and create corresponding visualization data
            val detectedObjects = results.map {
                var text = "Unknown"        // Default text

                // Display the top confident detection result if it exists
                if (it.labels.isNotEmpty()) {
                    val firstLabel = it.labels.first()
                    text = "${firstLabel.text}, ${firstLabel.confidence.times(100).toInt()}%"
                }
                dataClassBoxText(it.boundingBox, text)
            }

            // Draw the detection result on the input bitmap
            val visualizedResult = drawBoxTextDetections(bitmap, detectedObjects)

            // Show the detection result on the app screen
            imageView.setImageBitmap(visualizedResult)
        }.addOnFailureListener {
            Log.d(TAG, "On failure \n \n ")
            Log.e(TAG, it.message.toString())
        }
    }

    private fun addDimensionToBitmap(bitmap: Bitmap): Bitmap {
        val resultBitmap = Bitmap.createBitmap(1, 300, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(resultBitmap)
        canvas.drawBitmap(bitmap, null, RectF(0f, 0f, 300f, 300f), null)
        return resultBitmap
    }

    // Draw bounding boxes with object names around detected objects

    private fun drawBoxTextDetections(
        bitmap: Bitmap,
        detectionResults: List<dataClassBoxText>
    ): Bitmap {
        val outputBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(outputBitmap)
        val pen = Paint()
        pen.textAlign = Paint.Align.LEFT

        detectionResults.forEach {
            // draw bounding box
            pen.color = Color.GREEN
            pen.strokeWidth = 1.5F
            pen.style = Paint.Style.STROKE
            val box = it.box
            canvas.drawRect(box, pen)

            val tagSize = Rect(0, 0, 0, 0)

            // calculate the right font size
            pen.style = Paint.Style.FILL_AND_STROKE
            pen.color = Color.BLUE
            pen.strokeWidth = 1.5F
            pen.textSize = 80F
            pen.getTextBounds(it.text, 0, it.text.length, tagSize)
            val fontSize: Float = pen.textSize * box.width() / tagSize.width()

            // adjust the font size so texts are inside the bounding box
            if (fontSize < pen.textSize) pen.textSize = fontSize

            var margin = (box.width() - tagSize.width()) / 2.0F
            if (margin < 0F) margin = 0F
            canvas.drawText(
                it.text, box.left + margin,
                box.top + tagSize.height().times(1F), pen
            )
        }
        return outputBitmap
    }

    private fun print_results(detectedObjects: List<DetectedObject>) {
        detectedObjects.forEachIndexed { index, detectedObject ->
            val box = detectedObject.boundingBox

            Log.d(TAG, "Detected object: $index")
            Log.d(TAG, " trackingId: ${detectedObject.trackingId}")
            Log.d(TAG, " boundingBox: (${box.left}, ${box.top}) - (${box.right},${box.bottom})")
            detectedObject.labels.forEach {
                Log.d(TAG, " categories: ${it.text}")
                Log.d(TAG, " confidence: ${it.confidence}")
            }
        }
    }
}

// Data class to store detection results for visualization
data class dataClassBoxText(val box: Rect, val text: String)


