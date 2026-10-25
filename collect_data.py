import cv2
import numpy as np
import os
import mediapipe as mp
import sys
from utils.mediapipe_utils import mediapipe_detection, preprocess_keypoints, draw_styled_landmarks  # Updated to 'utils'

# Initialize Mediapipe
try:
    mp_holistic = mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1)
except Exception as e:
    print(f"Error initializing Mediapipe: {e}")
    sys.exit(1)

# Parameters
actions = ['blanc', 'hello', 'goodbye', 'yes', 'no', 'thank you', 'i love you']
SEQUENCE_LENGTH = 15
NUM_SEQUENCES = 50
DATA_PATH = './gesture_data'
TIMEOUT_SECONDS = 10
CAMERA_INDEX = 0

# Create directories
for action in actions:
    action_path = os.path.join(DATA_PATH, action)
    os.makedirs(action_path, exist_ok=True)

# Skip 'i love you' if data exists
skip_i_love_you = os.path.exists(os.path.join(DATA_PATH, 'i love you', 'seq_0.npy'))
if skip_i_love_you:
    print("Existing data found for 'i love you'. Skipping collection for this gesture.")
    actions = [a for a in actions if a != 'i love you']

# Initialize camera
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print(f"Error: Could not open webcam at index {CAMERA_INDEX}. Troubleshooting:")
    print("- Check if camera is connected and powered.")
    print("- Try other indices (0, 1, 2) by changing CAMERA_INDEX.")
    print("- Ensure no other apps are using the camera.")
    print("- Verify camera permissions in OS settings (Settings > Privacy > Camera on Windows).")
    print("- Update camera drivers.")
    print("- Reinstall opencv-python: pip install opencv-python")
    for i in range(3):
        test_cap = cv2.VideoCapture(i)
        if test_cap.isOpened():
            print(f"Success: Camera found at index {i}. Update CAMERA_INDEX to {i}.")
            test_cap.release()
    sys.exit(1)

try:
    for action in actions:
        for sequence in range(NUM_SEQUENCES):
            hand = 'right' if sequence < NUM_SEQUENCES // 2 else 'left'
            print(f"Collecting sequence {sequence + 1}/{NUM_SEQUENCES} for {action} ({hand} hand)")
            
            buffer = []
            start_time = None
            retries = 0
            max_retries = 3
            
            while len(buffer) < SEQUENCE_LENGTH and retries < max_retries:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to capture frame. Check camera connection.")
                    cap.release()
                    cv2.destroyAllWindows()
                    sys.exit(1)
                
                frame = cv2.flip(frame, 1)
                image, results = mediapipe_detection(frame, mp_holistic)
                draw_styled_landmarks(image, results)
                
                if start_time is None:
                    cv2.putText(image, f'Prepare for {action} ({hand} hand)', (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(image, 'Press "s" to start', (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow('Data Collection', image)
                    
                    if cv2.waitKey(1) & 0xFF == ord('s'):
                        start_time = cv2.getTickCount() / cv2.getTickFrequency()
                else:
                    elapsed = (cv2.getTickCount() / cv2.getTickFrequency()) - start_time
                    remaining = TIMEOUT_SECONDS - elapsed
                    if remaining <= 0:
                        print(f"Timeout reached for {action}, sequence {sequence + 1}. Retrying ({retries + 1}/{max_retries}).")
                        retries += 1
                        start_time = None
                        buffer = []
                        continue
                    
                    cv2.putText(image, f'Collecting {action} ({hand}): {len(buffer)}/{SEQUENCE_LENGTH}', 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(image, f'Time left: {remaining:.1f}s', 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    from utils.mediapipe_utils import extract_keypoints  # Updated to 'utils'
                    keypoints = extract_keypoints(results)
                    keypoints = preprocess_keypoints(keypoints)
                    
                    if keypoints.shape[0] != 292:
                        if keypoints.shape[0] > 292:
                            keypoints = keypoints[:292]  # Trim to 292
                            print(f"Trimmed keypoints to 292 for {action}, sequence {sequence + 1}.")
                        else:
                            print(f"Warning: Incorrect keypoint shape {keypoints.shape} for {action}, sequence {sequence + 1}. Expected 292.")
                            continue
                    
                    if action == 'blanc':
                        buffer.append(keypoints)
                    elif np.sum(keypoints[132:258]) != 0:  # Left hand (132:195) + Right hand (195:258)
                        buffer.append(keypoints)
                    else:
                        cv2.putText(image, 'No hand detected - Move hand into frame', 
                                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        print(f"No hand detected for {action}, sequence {sequence + 1}.")
                    
                    cv2.imshow('Data Collection', image)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    sys.exit(0)
            
            if len(buffer) == SEQUENCE_LENGTH:
                np.save(os.path.join(DATA_PATH, action, f'seq_{sequence}.npy'), np.array(buffer))
                print(f"Saved sequence {sequence + 1} for {action}")
            else:
                print(f"Failed to collect sequence {sequence + 1} for {action}: insufficient frames after {max_retries} retries.")
            
            cv2.waitKey(1000)

except Exception as e:
    print(f"Error during collection: {e}")
finally:
    cap.release()
    cv2.destroyAllWindows()
    mp_holistic.close()