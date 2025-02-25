package flickers;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

/**
 * The EyeBlinkDetector class is responsible for detecting eye blinks in video frames.
 * It uses OpenCV's CascadeClassifier to detect faces and eyes, and determines blinks
 * based on the absence of detected eyes for a certain number of consecutive frames.
 */
public class EyeBlinkDetector {
    private static final Logger logger = LogManager.getLogger(EyeBlinkDetector.class);

    private final CascadeClassifier faceCascade;
    private final CascadeClassifier eyeCascade;
    private double eyeAspectRatioThreshold = 0.3; // Threshold for determining blink
    private int consecutiveFrames = 3;            // Number of consecutive frames for a blink

    private boolean blinking = false;             // Current blink state
    private int framesWithoutEyes = 0;            // Counter for frames where no eyes are detected

    /**
     * Constructor for EyeBlinkDetector.
     * Initializes the face and eye cascade classifiers by loading the necessary XML files.
     */
    public EyeBlinkDetector() {
        // Initialize the classifiers
        faceCascade = new CascadeClassifier();
        eyeCascade = new CascadeClassifier();

        try {
            // Load the cascade classifier files from resources
            File faceCascadeFile = extractResource("haarcascade_frontalface_default.xml");
            File eyeCascadeFile = extractResource("haarcascade_eye.xml");

            // Load the cascade classifiers
            if (faceCascadeFile != null && eyeCascadeFile != null) {
                boolean faceLoaded = faceCascade.load(faceCascadeFile.getAbsolutePath());
                boolean eyeLoaded = eyeCascade.load(eyeCascadeFile.getAbsolutePath());

                if (!faceLoaded || !eyeLoaded) {
                    logger.error("Error loading cascade classifiers: Face loaded: {}, Eye loaded: {}", faceLoaded, eyeLoaded);
                }
            } else {
                logger.error("Could not extract cascade files from resources");
            }
        } catch (IOException e) {
            logger.error("Error loading cascade classifiers: {}", e.getMessage());
            e.printStackTrace();
        }
    }

    /**
     * Extracts a resource file to a temporary location for loading by OpenCV.
     *
     * @param resourceName The name of the resource file to extract.
     * @return A File object pointing to the extracted resource file.
     * @throws IOException If an error occurs during file extraction.
     */
    private File extractResource(String resourceName) throws IOException {
        // First, check if file exists in current directory
        File localFile = new File(resourceName);
        if (localFile.exists()) {
            return localFile;
        }

        // If not found, try to extract from resources
        InputStream is = getClass().getResourceAsStream("/" + resourceName);
        if (is == null) {
            is = getClass().getResourceAsStream(resourceName);
        }

        if (is == null) {
            logger.error("Could not find resource: {}", resourceName);
            return null;
        }

        // Create a temporary file
        File tempFile = File.createTempFile("cascade_", ".xml");
        tempFile.deleteOnExit();

        // Copy the resource to the temporary file
        try (FileOutputStream os = new FileOutputStream(tempFile)) {
            byte[] buffer = new byte[1024];
            int bytesRead;
            while ((bytesRead = is.read(buffer)) != -1) {
                os.write(buffer, 0, bytesRead);
            }
        }

        is.close();
        return tempFile;
    }

    /**
     * Detects blinks in the given video frame.
     *
     * @param frame The video frame to process.
     * @return True if a blink is detected, false otherwise.
     */
    public boolean detectBlink(Mat frame) {
        // Reset blink state at the beginning of each call
        boolean blinkDetected = false;

        if (frame.empty() || faceCascade.empty() || eyeCascade.empty()) {
            logger.error("Frame is empty or classifiers not loaded properly");
            return false;
        }

        // Convert to grayscale for better detection
        Mat grayFrame = new Mat();
        Imgproc.cvtColor(frame, grayFrame, Imgproc.COLOR_BGR2GRAY);
        Imgproc.equalizeHist(grayFrame, grayFrame);

        // Detect faces
        MatOfRect faces = new MatOfRect();
        try {
            faceCascade.detectMultiScale(
                    grayFrame,
                    faces,
                    1.1,
                    3,
                    0,
                    new Size(30, 30),
                    new Size()
            );
        } catch (Exception e) {
            logger.error("Error in face detection: {}", e.getMessage());
            grayFrame.release();
            return false;
        }

        Rect[] facesArray = faces.toArray();
        boolean eyesDetected = false;

        if (facesArray.length > 0) {
            // We'll focus on the first face detected
            Rect faceRect = facesArray[0];

            // Create a region of interest for the face
            Mat faceROI = grayFrame.submat(faceRect);

            // Detect eyes within the face region
            MatOfRect eyes = new MatOfRect();
            try {
                eyeCascade.detectMultiScale(
                        faceROI,
                        eyes,
                        1.1,
                        2,
                        0,
                        new Size(20, 20),
                        new Size()
                );
            } catch (Exception e) {
                logger.error("Error in eye detection: {}", e.getMessage());
                faceROI.release();
                grayFrame.release();
                return false;
            }

            Rect[] eyesArray = eyes.toArray();

            // If we can detect eyes, reset the frames without eyes counter
            if (eyesArray.length > 0) {
                framesWithoutEyes = 0;
                eyesDetected = true;
            } else {
                // Increment counter for frames without eyes
                framesWithoutEyes++;

                // Check if this could be a blink
                // Minimum frames without eyes to count as a blink
                int minFramesForBlink = 2;
                // Maximum frames without eyes to still count as a blink
                int maxFramesForBlink = 7;
                if (framesWithoutEyes >= minFramesForBlink && framesWithoutEyes <= maxFramesForBlink) {
                    if (!blinking) {
                        blinking = true;
                        blinkDetected = true;
                    }
                } else if (framesWithoutEyes > maxFramesForBlink) {
                    // Too many frames without eyes - probably not a blink but eyes closed or looking away
                    blinking = false;
                }
            }

            // If eyes are detected after a blink, reset the blink state
            if (eyesDetected && blinking) {
                blinking = false;
            }

            faceROI.release();
        } else {
            // No face detected, reset counters
            framesWithoutEyes = 0;
            blinking = false;
        }

        grayFrame.release();
        return blinkDetected;
    }

    /**
     * Resets the detector state.
     */
    public void reset() {
        blinking = false;
        framesWithoutEyes = 0;
    }

    /**
     * Gets the eye aspect ratio threshold.
     *
     * @return The eye aspect ratio threshold.
     */
    public double getEyeAspectRatioThreshold() {
        return eyeAspectRatioThreshold;
    }

    /**
     * Sets the eye aspect ratio threshold.
     *
     * @param eyeAspectRatioThreshold The new eye aspect ratio threshold.
     */
    public void setEyeAspectRatioThreshold(double eyeAspectRatioThreshold) {
        this.eyeAspectRatioThreshold = eyeAspectRatioThreshold;
    }

    /**
     * Gets the number of consecutive frames required for a blink.
     *
     * @return The number of consecutive frames.
     */
    public int getConsecutiveFrames() {
        return consecutiveFrames;
    }

    /**
     * Sets the number of consecutive frames required for a blink.
     *
     * @param consecutiveFrames The new number of consecutive frames.
     */
    public void setConsecutiveFrames(int consecutiveFrames) {
        this.consecutiveFrames = consecutiveFrames;
    }
}