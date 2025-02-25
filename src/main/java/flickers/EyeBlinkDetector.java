package flickers;

import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.features2d.Features2d;

public class EyeBlinkDetector {
    private CascadeClassifier faceCascade;
    private CascadeClassifier eyeCascade;
    private double eyeAspectRatioThreshold = 0.3; // Threshold for determining blink
    private int consecutiveFrames = 3;            // Number of consecutive frames for a blink
    private int counter = 0;                      // Counter for consecutive frames
    private boolean blinking = false;             // Current blink state

    public EyeBlinkDetector() {
        // Initialize the classifiers
        faceCascade = new CascadeClassifier();
        eyeCascade = new CascadeClassifier();

        // Load the Haar cascade XML files (assumes these files are in your resources)
        faceCascade.load("haarcascade_frontalface_default.xml");
        eyeCascade.load("haarcascade_eye.xml");

        if (faceCascade.empty() || eyeCascade.empty()) {
            System.err.println("Error loading cascade classifiers");
        }
    }

    public boolean detectBlink(Mat frame) {
        if (frame.empty()) {
            return false;
        }

        // Convert to grayscale for better detection
        Mat grayFrame = new Mat();
        Imgproc.cvtColor(frame, grayFrame, Imgproc.COLOR_BGR2GRAY);
        Imgproc.equalizeHist(grayFrame, grayFrame);

        // Detect faces
        MatOfRect faces = new MatOfRect();
        faceCascade.detectMultiScale(grayFrame, faces, 1.1, 3, 0, new Size(30, 30), new Size());

        Rect[] facesArray = faces.toArray();
        if (facesArray.length > 0) {
            // We'll focus on the first face detected
            Rect faceRect = facesArray[0];

            // Create a region of interest for the face
            Mat faceROI = grayFrame.submat(faceRect);

            // Detect eyes within the face region
            MatOfRect eyes = new MatOfRect();
            eyeCascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0, new Size(20, 20), new Size());

            Rect[] eyesArray = eyes.toArray();

            // If we can't detect eyes, that might be a blink
            if (eyesArray.length == 0) {
                counter++;
                if (counter >= consecutiveFrames) {
                    blinking = true;
                }
            } else {
                // Process detected eyes to check for blink using EAR (Eye Aspect Ratio)
                double earValue = calculateEAR(eyesArray, faceROI);

                if (earValue < eyeAspectRatioThreshold) {
                    counter++;
                    if (counter >= consecutiveFrames) {
                        blinking = true;
                    }
                } else {
                    counter = 0;
                    blinking = false;
                }
            }
        } else {
            // Reset if no face detected
            counter = 0;
            blinking = false;
        }

        grayFrame.release();
        return blinking;
    }

    private double calculateEAR(Rect[] eyes, Mat faceROI) {
        // If we have at least one eye detected
        if (eyes.length > 0) {
            double sumEAR = 0.0;
            int count = 0;

            for (Rect eye : eyes) {
                // Extract eye region
                Mat eyeROI = faceROI.submat(eye);

                // Calculate eye height and width
                double eyeHeight = eye.height;
                double eyeWidth = eye.width;

                // Simple EAR approximation based on height/width ratio
                // A more accurate implementation would use eye landmarks
                double ear = eyeHeight / eyeWidth;

                sumEAR += ear;
                count++;

                eyeROI.release();
            }

            // Return average EAR for all detected eyes
            return count > 0 ? sumEAR / count : 1.0;
        }

        return 1.0; // Return high value if no eyes detected
    }

    // Additional method to reset the detector state if needed
    public void reset() {
        counter = 0;
        blinking = false;
    }
}




//
// public class EyeBlinkDetector {
//    public boolean detectBlink(Mat frame) {
//        // Placeholder for blink detection logic
//        // In a real implementation, this method would process the frame
//        // using an eye detection model and determine if a blink occurred.
//        return false;
//    }
//}

