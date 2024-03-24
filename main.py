from ultralytics import YOLO
import cv2

# Load YOLOv8 model
model = YOLO('best.pt')

# Path to the image
image_path = 'test.jpg'

while True:
    # Read image
    frame = cv2.imread(image_path)

    # Detect and track objects
    results = model.track(frame, persist=True)

    # Plot results
    frame_ = results[0].plot()

    # Visualize
    cv2.imshow('frame', frame_)
    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()
