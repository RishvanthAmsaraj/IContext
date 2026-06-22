import cv2
from ultralytics import YOLO
import numpy as np
import os
from collections import Counter

from context_analyzer import ContextAnalyzer

# --- Configuration ---
MODEL_PATH = 'yolov8n.pt'  # Nano version for faster inference
CONFIDENCE_THRESHOLD = 0.5
CAMERA_INDEX = 0  # Default built-in camera

# --- Initialization ---
analyzer = ContextAnalyzer()

# --- Initialization ---

def initialize_model(model_path):
    """Load the YOLOv8 model."""
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Downloading...")
        try:
            model = YOLO(model_path)
            print("Model downloaded and loaded successfully.")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Please ensure you have internet access and the 'ultralytics' package is installed.")
            return None
    else:
        print(f"Loading model from {model_path}...")
        try:
            model = YOLO(model_path)
            print("Model loaded successfully.")
            return model
        except Exception as e:
            print(f"Error loading model from file: {e}")
            return None

def detect_objects(model):
    """Perform object detection and context analysis using webcam feed."""
    if model is None:
        print("Model is not loaded. Exiting.")
        return

    # Open webcam - explicitly use AVFoundation backend for better macOS support
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        print(f"Error: Could not open webcam at index {CAMERA_INDEX}.")
        return

    print("Starting object detection. Press 'q' to quit.")

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Perform detection
        results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)

        # Process results and draw bounding boxes
        annotated_frame = frame.copy()
        detected_object_names = []

        for result in results:
            boxes = result.boxes  # Boxes object for bounding box outputs
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = box.conf[0].item()
                class_id = int(box.cls[0].item())
                class_name = model.names[class_id]
                detected_object_names.append(class_name)

                # Draw bounding box
                cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                # Prepare label text
                label = f"{class_name}: {confidence:.2f}"

                # Put label and confidence
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated_frame, (int(x1), int(y1) - h - 5), (int(x1) + w, int(y1)), (0, 255, 0), -1)
                cv2.putText(annotated_frame, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # --- Context Analysis ---
        # Count occurrences of each object
        object_counts = Counter(detected_object_names)
        unique_objects = list(object_counts.keys())

        if unique_objects:
            contexts = analyzer.analyze(unique_objects)
            context_text = " | ".join(contexts)

            # Display context at the top of the frame
            cv2.rectangle(annotated_frame, (0, 0), (annotated_frame.shape[1], 60), (0, 0, 0), -1)
            cv2.putText(annotated_frame, f"Context: {context_text}", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # Display object counts
            counts_text = ", ".join([f"{obj}: {count}" for obj, count in object_counts.items()])
            cv2.putText(annotated_frame, f"Detected: {counts_text}", (10, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
        else:
            cv2.rectangle(annotated_frame, (0, 0), (annotated_frame.shape[1], 30), (0, 0, 0), -1)
            cv2.putText(annotated_frame, "Context: No objects detected.", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Display the annotated frame
        cv2.imshow('IContext - Object Detection & Context', annotated_frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Exiting object detection.")

# --- Main Execution ---
if __name__ == "__main__":
    # Ensure the model directory exists if needed, or let YOLO handle downloads
    # For simplicity, YOLO will download to a default cache location if not found
    
    loaded_model = initialize_model(MODEL_PATH)
    detect_objects(loaded_model)
