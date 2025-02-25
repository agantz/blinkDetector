package flickers;
import org.opencv.core.Mat;
import org.opencv.videoio.VideoCapture;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

public class BlinkDetector {

    private static final Logger logger = LogManager.getLogger(BlinkDetector.class);

    public static void main(String[] args) {

        try {
            org.bytedeco.javacpp.Loader.load(org.bytedeco.opencv.opencv_java.class);
        } catch (Exception e) {
            logger.error("Error loading OpenCV library", e);
            return;
        }

        if (args.length != 1) {
            logger.info("Usage: java -jar BlinkDetector.jar <path-to-mpg-file>");
            return;
        }

        String videoPath = args[0];
        int blinkCount = countBlinks(videoPath);
        logger.info("Number of blinks detected: " + blinkCount);
    }

    private static int countBlinks(String videoPath) {
        VideoCapture capture = new VideoCapture(videoPath);
        if (!capture.isOpened()) {
            logger.error("Error: Could not open video.");
            return 0;
        }

        Mat frame = new Mat();
        int blinkCount = 0;
        EyeBlinkDetector eyeBlinkDetector = new EyeBlinkDetector();

        while (capture.read(frame)) {
            if (eyeBlinkDetector.detectBlink(frame)) {
                blinkCount++;
            }
        }

        capture.release();
        return blinkCount;
    }
}