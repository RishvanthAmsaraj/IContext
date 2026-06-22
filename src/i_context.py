"""
IContext: Advanced Context-Aware Object and Person Analysis

This script combines YOLOv8 for object detection and MediaPipe for detailed
person analysis (pose, hands, and face landmarks). It generates a dynamic
collective context based on detected objects and the presence/pose of people.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
from collections import Counter
import os
import argparse

# Import the context analyzer
from context_analyzer import ContextAnalyzer

# --- Configuration ---
MODEL_PATH = 'yolov8n.pt'
CONFIDENCE_THRESHOLD = 0.5
CAMERA_INDEX = 0

# --- Initialization ---
print("Initializing models...")
yolo_model = YOLO(MODEL_PATH)
analyzer = ContextAnalyzer()

# Lazy-load Florence-2 only when needed (it's heavy)
scene_describer = None

def get_scene_describer():
    global scene_describer
    if scene_describer is None:
        try:
            from scene_describer import SceneDescriber
            scene_describer = SceneDescriber()
        except Exception as e:
            print(f"Could not load Florence-2: {e}")
            scene_describer = False  # Mark as failed
    return scene_describer if scene_describer else None

# MediaPipe Initialization
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize MediaPipe solutions
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=4)
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

print("Models loaded successfully.")


# --- Interaction Helper Functions ---
import math

def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def get_pixel_coords(landmark, frame_w, frame_h):
    """Convert normalized landmark to pixel coordinates."""
    return (int(landmark.x * frame_w), int(landmark.y * frame_h))


def get_hand_center(hand_landmarks, frame_w, frame_h):
    """Get the center of the hand based on wrist and middle finger base."""
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    cx = int((wrist.x + middle_mcp.x) / 2 * frame_w)
    cy = int((wrist.y + middle_mcp.y) / 2 * frame_h)
    return (cx, cy)


def is_typing_gesture(hand_landmarks):
    """Detect if the hand is making a typing gesture (fingers curled)."""
    curled = 0
    finger_pairs = [
        (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_MCP),
        (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_MCP),
        (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_MCP),
        (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_MCP),
    ]
    for tip_id, base_id in finger_pairs:
        if hand_landmarks.landmark[tip_id].y > hand_landmarks.landmark[base_id].y:
            curled += 1
    return curled >= 3


def detect_interaction(hand_landmarks, object_box, nose_pos, frame_w, frame_h):
    """
    Detect if a hand is interacting with an object and classify the action.

    Returns:
        str: "phone_ear", "typing", "holding", or None
    """
    x1, y1, x2, y2 = object_box
    obj_center = ((x1 + x2) // 2, (y1 + y2) // 2)
    hand_center = get_hand_center(hand_landmarks, frame_w, frame_h)

    distance = calculate_distance(hand_center, obj_center)

    # Interaction threshold (pixels)
    if distance > 150:
        return None

    # Check if hand is near the head/ear (phone to ear gesture)
    if nose_pos:
        nose_to_hand = calculate_distance(hand_center, nose_pos)
        if nose_to_hand < 100:
            return "phone_ear"

    # Check for typing gesture
    if is_typing_gesture(hand_landmarks):
        return "typing"

    return "holding"

def analyze_person_state(pose_landmarks, hand_landmarks_list):
    """Determine a simple state/action of a detected person based on landmarks."""
    if not pose_landmarks:
        return "a person is visible"

    # Get key landmark positions
    try:
        # Check if hands are raised (e.g., waving or studying)
        left_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        nose = pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]

        # Check if hands are near the face/head (e.g., thinking, eating, or on phone)
        if (left_wrist.y < nose.y and right_wrist.y < nose.y):
            return "a person has their hands near their face"
        # Check if hands are extended forward (e.g., typing or holding something)
        elif (left_wrist.visibility > 0.5 and right_wrist.visibility > 0.5):
            return "a person appears to be interacting with something in front of them"
    except Exception:
        pass

    if hand_landmarks_list:
        return f"a person is visible with {len(hand_landmarks_list)} hand(s) detected"
    return "a person is visible"


def generate_dynamic_context(objects, person_states, has_face):
    """Generate a combined context based on objects, people, and poses."""
    context_parts = []

    # Analyze objects
    if objects:
        obj_contexts = analyzer.analyze(objects)
        context_parts.extend(obj_contexts)

    # Analyze people
    if person_states:
        if has_face:
            context_parts.append("A person's face is visible.")
        for state in person_states:
            context_parts.append(state.capitalize() + ".")

    if not context_parts:
        return "Analyzing scene..."

    # Combine and de-duplicate
    unique_contexts = list(dict.fromkeys(context_parts))
    return " | ".join(unique_contexts)


def process_frame(frame, tracked_objects_state, interaction_votes):
    """Process a single frame for objects and people with tracking."""
    frame_h, frame_w = frame.shape[:2]

    # 1. YOLO Object Detection with Tracking
    yolo_results = yolo_model.track(frame, conf=CONFIDENCE_THRESHOLD, verbose=False, persist=True)
    detected_objects = []
    person_boxes = []  # Store (x1, y1, x2, y2) for people
    # Store tracked object info: {track_id: (class_name, box)}
    tracked_boxes = {}

    for result in yolo_results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            class_id = int(box.cls[0].item())
            class_name = yolo_model.names[class_id]
            confidence = box.conf[0].item()
            detected_objects.append(class_name)

            # Get the tracking ID (if available)
            track_id = int(box.id[0].item()) if box.id is not None else None
            display_name = class_name
            if track_id is not None:
                # Try to re-identify this object if it's not already tracked
                if track_id not in tracked_objects_state:
                    try_reidentify(track_id, class_name, tracked_objects_state, interaction_votes)
                # Look up if this ID has been reclassified
                if track_id in tracked_objects_state:
                    display_name = tracked_objects_state[track_id]
                # Store for interaction analysis (exclude persons)
                if class_name != "person":
                    tracked_boxes[track_id] = (class_name, (x1, y1, x2, y2))
                label = f"ID:{track_id} {display_name}: {confidence:.2f}"
            else:
                label = f"{display_name}: {confidence:.2f}"

            # Draw bounding box
            color = (0, 255, 0) if class_name != "person" else (255, 0, 0)
            # Use a different color if reclassified
            if display_name != class_name:
                color = (0, 255, 255)  # Yellow for reclassified
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)  # Thicker box
            # Draw label with filled background for visibility
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
            cv2.putText(frame, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            if class_name == "person":
                person_boxes.append((x1, y1, x2, y2))

    # 2. MediaPipe Person Analysis (on the whole frame for simplicity)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False

    pose_results = pose.process(rgb_frame)
    hand_results = hands.process(rgb_frame)
    face_results = face_mesh.process(rgb_frame)

    rgb_frame.flags.writeable = True

    person_states = []
    has_face = False

    # Draw Pose Landmarks
    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )

    # Draw Hand Landmarks
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

    # Draw Face Mesh (simplified, just the tessellation)
    if face_results.multi_face_landmarks:
        has_face = True
        for face_landmarks in face_results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )

    # --- Hand-Object Interaction & Reclassification ---
    # Get nose position from pose for "phone to ear" detection
    nose_pos = None
    if pose_results.pose_landmarks:
        nose = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        nose_pos = get_pixel_coords(nose, frame_w, frame_h)

    # Check each hand against each tracked object
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            hand_center = get_hand_center(hand_landmarks, frame_w, frame_h)
            for track_id, (orig_name, obj_box) in tracked_boxes.items():
                action = detect_interaction(hand_landmarks, obj_box, nose_pos, frame_w, frame_h)
                if action:
                    # Initialize vote counter for this object
                    if track_id not in interaction_votes:
                        interaction_votes[track_id] = {"phone_ear": 0, "typing": 0, "holding": 0}
                    interaction_votes[track_id][action] += 1

                    # Draw interaction line between hand and object
                    obj_center = ((obj_box[0] + obj_box[2]) // 2, (obj_box[1] + obj_box[3]) // 2)
                    cv2.line(frame, hand_center, obj_center, (0, 165, 255), 2)
                    cv2.putText(frame, f"Action: {action}", (hand_center[0] + 10, hand_center[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1, cv2.LINE_AA)

                    # Check if we have enough votes to reclassify
                    votes = interaction_votes[track_id]
                    # Reclassify "remote" as "cell phone" if held to ear
                    if orig_name == "remote" and votes["phone_ear"] > 15:
                        tracked_objects_state[track_id] = "cell phone"
                        print(f"Reclassified ID:{track_id} from 'remote' to 'cell phone'")
                    # Reclassify "remote" as "keyboard" if typing
                    elif orig_name == "remote" and votes["typing"] > 20:
                        tracked_objects_state[track_id] = "keyboard"
                        print(f"Reclassified ID:{track_id} from 'remote' to 'keyboard'")
                    # Reclassify ambiguous items being held as "phone"
                    elif orig_name in ["remote", "cell phone"] and votes["phone_ear"] > 10:
                        tracked_objects_state[track_id] = "cell phone"

    # Decay old votes slightly to allow for changing behavior
    for track_id in interaction_votes:
        for action in interaction_votes[track_id]:
            interaction_votes[track_id][action] = max(0, interaction_votes[track_id][action] - 1)

    # 3. Generate Context
    # If we have pose landmarks, analyze the person's state
    if pose_results.pose_landmarks:
        state = analyze_person_state(
            pose_results.pose_landmarks,
            hand_results.multi_hand_landmarks
        )
        person_states.append(state)

    # Use the reclassified names for context generation
    display_objects = []
    for obj_name in detected_objects:
        # Check if this object has been reclassified
        found_reclassified = False
        for track_id, new_name in tracked_objects_state.items():
            if track_id in tracked_boxes and tracked_boxes[track_id][0] == obj_name:
                display_objects.append(new_name)
                found_reclassified = True
                break
        if not found_reclassified:
            display_objects.append(obj_name)

    unique_objects = list(set(display_objects))
    context = generate_dynamic_context(unique_objects, person_states, has_face)

    return frame, context, detected_objects, tracked_objects_state, interaction_votes


# --- Re-identification System ---
# When objects are briefly lost (e.g., occluded by hands), we keep their state
# in this buffer and try to re-match them when they reappear.
MAX_LOST_FRAMES = 30  # Keep lost objects for ~1 second at 30fps

# Global lost object buffer
# Format: {track_id: {"class_name": str, "votes": dict, "reclassified_name": str, "frames_lost": int}}
lost_objects_buffer = {}


def reconcile_tracking(current_ids, tracked_objects_state, interaction_votes):
    """
    Move lost objects to the buffer, and clean up old entries.
    """
    global lost_objects_buffer

    # 1. Find objects that are no longer tracked (lost since last frame)
    for track_id in list(tracked_objects_state.keys()):
        if track_id not in current_ids:
            # This object was lost — save its state to the buffer
            if track_id not in lost_objects_buffer:
                lost_objects_buffer[track_id] = {
                    "class_name": tracked_objects_state[track_id],
                    "votes": interaction_votes.get(track_id, {}).copy(),
                    "reclassified_name": tracked_objects_state[track_id],
                    "frames_lost": 0,
                }
            # Remove from active state (it will be re-added if re-identified)
            del tracked_objects_state[track_id]
            if track_id in interaction_votes:
                del interaction_votes[track_id]

    # 2. Increment frames_lost for all buffered objects
    for lost_id in list(lost_objects_buffer.keys()):
        lost_objects_buffer[lost_id]["frames_lost"] += 1
        # Remove if too old
        if lost_objects_buffer[lost_id]["frames_lost"] > MAX_LOST_FRAMES:
            del lost_objects_buffer[lost_id]

    return tracked_objects_state, interaction_votes


def try_reidentify(new_track_id, new_class_name, tracked_objects_state, interaction_votes):
    """
    Try to match a newly tracked object to a recently lost one.
    If matched, transfer the old state (votes, reclassification) to the new ID.
    """
    global lost_objects_buffer
    if not lost_objects_buffer:
        return

    best_match = None

    for lost_id, lost_data in list(lost_objects_buffer.items()):
        # Class must match for re-identification
        if lost_data["class_name"] != new_class_name:
            continue

        # Prefer the most recently lost object (lowest frames_lost)
        if best_match is None or lost_data["frames_lost"] < best_match[1]["frames_lost"]:
            best_match = (lost_id, lost_data)

    if best_match:
        lost_id, lost_data = best_match
        # Transfer state: votes and reclassification
        if lost_data["votes"]:
            interaction_votes[new_track_id] = lost_data["votes"].copy()
        if lost_data["reclassified_name"] != lost_data["class_name"]:
            tracked_objects_state[new_track_id] = lost_data["reclassified_name"]
            print(f"Re-identified ID:{new_track_id} as previously lost ID:{lost_id} "
                  f"({lost_data['class_name']} -> {lost_data['reclassified_name']})")
        # Remove from buffer
        del lost_objects_buffer[lost_id]


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="IContext: Context-Aware Object and Person Analysis"
    )
    parser.add_argument(
        "--source", type=str, default="webcam",
        choices=["webcam", "video", "image"],
        help="Input source: webcam, video file, or image file"
    )
    parser.add_argument(
        "--path", type=str, default=None,
        help="Path to video or image file (required for --source video or image)"
    )
    return parser.parse_args()


def main():
    """Main loop to capture and process video."""
    args = parse_args()

    # Handle different input sources
    if args.source == "image":
        if not args.path:
            print("Error: --path is required for image source")
            return
        print(f"Analyzing image: {args.path}")
        frame = cv2.imread(args.path)
        if frame is None:
            print(f"Error: Could not read image at {args.path}")
            return

        os.makedirs("results", exist_ok=True)
        tracked_objects_state = {}
        interaction_votes = {}

        processed_frame, context, objects, tracked_objects_state, interaction_votes = process_frame(
            frame, tracked_objects_state, interaction_votes
        )

        # Generate Florence-2 scene description if available
        florence_caption = None
        describer = get_scene_describer()
        if describer:
            try:
                print("Generating Florence-2 scene description...")
                florence_caption = describer.get_caption(frame, detailed=True)
                print(f"Florence-2: {florence_caption}")
            except Exception as e:
                print(f"Florence-2 error: {e}")
                florence_caption = None

        # Save the result
        out_path = "results/image_analysis.jpg"

        # Draw context overlay at the top (same as live mode)
        h, w = processed_frame.shape[:2]
        # Determine banner height based on context length
        banner_h = 60
        if florence_caption:
            banner_h = 110  # More space for VLM description
        cv2.rectangle(processed_frame, (0, 0), (w, banner_h), (0, 0, 0), -1)
        cv2.putText(processed_frame, "IContext Analysis", (10, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
        # Wrap context text
        y_offset = 38
        words = context.split(' ')
        line = ""
        for word in words:
            test_line = line + word + " "
            (tw, _), _ = cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            if tw > w - 20:
                cv2.putText(processed_frame, line, (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                line = word + " "
                y_offset += 14
            else:
                line = test_line
        cv2.putText(processed_frame, line, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        # Draw Florence-2 caption if available
        if florence_caption:
            y_offset = 60
            cv2.putText(processed_frame, "Florence-2:", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
            y_offset += 14
            words = florence_caption.split(' ')
            line = ""
            for word in words:
                test_line = line + word + " "
                (tw, _), _ = cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
                if tw > w - 20:
                    cv2.putText(processed_frame, line, (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 255), 1, cv2.LINE_AA)
                    line = word + " "
                    y_offset += 12
                else:
                    line = test_line
            cv2.putText(processed_frame, line, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 255), 1, cv2.LINE_AA)

        # Draw object counts at the bottom
        if objects:
            counts = Counter(objects)
            counts_text = ", ".join([f"{obj}: {count}" for obj, count in counts.items()])
            cv2.rectangle(processed_frame, (0, h - 25), (w, h), (0, 0, 0), -1)
            cv2.putText(processed_frame, f"Objects: {counts_text}", (10, h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imwrite(out_path, processed_frame)
        print(f"\n=== Context Analysis ===")
        print(f"Context: {context}")
        print(f"Objects detected: {objects}")
        if florence_caption:
            print(f"Florence-2 Description: {florence_caption}")
        print(f"Result saved to: {out_path}")
        return

    # Video or webcam source
    if args.source == "video":
        if not args.path:
            print("Error: --path is required for video source")
            return
        print(f"Opening video: {args.path}")
        cap = cv2.VideoCapture(args.path)
    else:
        print("Opening webcam...")
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_AVFOUNDATION)

    if not cap.isOpened():
        source_desc = args.path if args.source == "video" else f"index {CAMERA_INDEX}"
        print(f"Error: Could not open source: {source_desc}")
        return

    print("Starting IContext. Press 'q' to quit, 'c' to capture.")

    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)

    # Dictionary to track persistent object IDs and their (potentially reclassified) names
    # Format: {track_id: "display_name"}
    tracked_objects_state = {}
    # Track interaction votes per object ID for reclassification
    # Format: {track_id: {"phone_ear": count, "typing": count, "holding": count}}
    interaction_votes = {}

    # Florence-2 is slow on CPU (~3-5s per image), so throttle it
    # Update the VLM description every N frames (default: every 60 frames ~ 2 seconds)
    FLORENCE_INTERVAL = 60
    frame_count = 0
    florence_caption = None
    describer = get_scene_describer()  # Lazy load once
    if describer:
        print(f"Florence-2 enabled. Generating descriptions every {FLORENCE_INTERVAL} frames.")
    else:
        print("Florence-2 not available. Running in rule-based mode only.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        frame_count += 1

        # Reconcile tracking: move lost objects to buffer, clean up old entries
        current_ids = set(tracked_objects_state.keys())
        tracked_objects_state, interaction_votes = reconcile_tracking(
            current_ids, tracked_objects_state, interaction_votes
        )

        # Process the frame
        processed_frame, context, objects, tracked_objects_state, interaction_votes = process_frame(
            frame, tracked_objects_state, interaction_votes
        )

        # Periodically generate Florence-2 caption (throttled)
        if describer and frame_count % FLORENCE_INTERVAL == 0:
            try:
                florence_caption = describer.get_caption(processed_frame, detailed=True)
            except Exception as e:
                print(f"Florence-2 error: {e}")

        # Draw overlays (same as image mode)
        h, w = processed_frame.shape[:2]
        banner_h = 110 if florence_caption else 60
        cv2.rectangle(processed_frame, (0, 0), (w, banner_h), (0, 0, 0), -1)

        # Title
        cv2.putText(processed_frame, "IContext Live Analysis", (10, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

        # Wrap context text
        y_offset = 38
        words = context.split(' ')
        line = ""
        for word in words:
            test_line = line + word + " "
            (tw, _), _ = cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            if tw > w - 20:
                cv2.putText(processed_frame, line, (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                line = word + " "
                y_offset += 14
            else:
                line = test_line
        cv2.putText(processed_frame, line, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        # Draw Florence-2 caption if available
        if florence_caption:
            y_offset = 70
            cv2.putText(processed_frame, "Florence-2:", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
            y_offset += 14
            words = florence_caption.split(' ')
            line = ""
            for word in words:
                test_line = line + word + " "
                (tw, _), _ = cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
                if tw > w - 20:
                    cv2.putText(processed_frame, line, (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 255), 1, cv2.LINE_AA)
                    line = word + " "
                    y_offset += 12
                else:
                    line = test_line
            cv2.putText(processed_frame, line, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 255), 1, cv2.LINE_AA)

        # Object counts at the bottom
        if objects:
            counts = Counter(objects)
            counts_text = ", ".join([f"{obj}: {count}" for obj, count in counts.items()])
            cv2.rectangle(processed_frame, (0, h - 25), (w, h), (0, 0, 0), -1)
            cv2.putText(processed_frame, f"Objects: {counts_text}", (10, h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow('IContext - Advanced Scene Analysis', processed_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            # Save the frame
            timestamp = cv2.getTickCount()
            filename = f"results/capture_{timestamp}.jpg"
            cv2.imwrite(filename, processed_frame)
            print(f"Saved capture to {filename}")

    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    hands.close()
    face_mesh.close()
    print("Exiting IContext.")


if __name__ == "__main__":
    main()
