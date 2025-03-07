package flickers;

import org.opencv.core.Mat;
import org.opencv.videoio.VideoCapture;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.opencv.videoio.Videoio;

public class BlinkDetector {

    private static final Logger logger = LogManager.getLogger(BlinkDetector.class);

    /**
     * The main method serves as the entry point for the BlinkDetector application.
     * It loads the OpenCV library, checks for the correct number of command-line arguments,
     * and calls the countBlinks method to process the video file and count the number of blinks.
     *
     * @param args Command-line arguments. Expects a single argument: the path to the video file.
     */
    public static void main(String[] args) {

        try {
            // Load the OpenCV library
            org.bytedeco.javacpp.Loader.load(org.bytedeco.opencv.opencv_java.class);
        } catch (Exception e) {
            logger.error("Error loading OpenCV library", e);
            return;
        }

        // Check if the correct number of command-line arguments is provided
        if (args.length != 1) {
            logger.info("Usage: java -jar BlinkDetector.jar <path-to-mpg-file>");
            return;
        }

        // Get the video file path from the command-line arguments
        String videoPath = args[0];
        // Count the number of blinks in the video file
        int blinkCount = countBlinks(videoPath);
        logger.info("file: {}  was examined", videoPath);
        logger.info("Number of blinks detected: {}", blinkCount);
    }

    /**
     * Counts the number of blinks in the given video file.
     * It opens the video file, reads frames in a loop, and uses the EyeBlinkDetector
     * to detect blinks in each frame. The total number of blinks is returned.
     *
     * @param videoPath The path to the video file.
     * @return The total number of blinks detected in the video file.
     */
    private static int countBlinks(String videoPath) {
        // Open the video file
        VideoCapture capture = new VideoCapture(videoPath);
        if (!capture.isOpened()) {
            logger.error("Error: Could not open video.");
            return 0;
        }

        // Get the total frame count of the video
        double frameCount = capture.get(Videoio.CAP_PROP_FRAME_COUNT);
        if (frameCount == 0) {
            logger.error("Error: Could not retrieve frame count.");
            return 0;
        }

        // Get the frame rate of the video
        double frameRate = capture.get(Videoio.CAP_PROP_FPS);
        if (frameRate == 0) {
            logger.error("Error: Could not retrieve frame rate.");
            return 0;
        }

        // Calculate the duration of the video in seconds
        double videoDuration = frameCount / frameRate;
        logger.info("Video duration: {} seconds (min: {} sec: {})",
                videoDuration, videoDuration/60, videoDuration%60);



        Mat frame = new Mat();
        int myFrameCount = 0;
        int blinkCount = 0;
        EyeBlinkDetector eyeBlinkDetector = new EyeBlinkDetector();

        // Read frames from the video file in a loop
        while (capture.read(frame)) {
            myFrameCount++;
            // Detect blinks in the current frame
            if (eyeBlinkDetector.detectBlink(frame)) {
                blinkCount++;
                logger.info("Blink detected at frame: {} min: {} sec: {}",
                        myFrameCount, myFrameCount/25/60, myFrameCount/25%60);
            }
        }

        logger.info("Video duration: {} seconds", videoDuration);
        logger.info("Frame rate: {}", frameRate);
        logger.info("Total frames processed: {}", myFrameCount);
        logger.info("Total blinks detected: {}", blinkCount);

        // Release the video capture
        capture.release();
        return blinkCount;
    }
}