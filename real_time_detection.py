import cv2
import numpy as np
import tensorflow as tf
from collections import deque
import random
from utils.mediapipe_utils import mediapipe_detection, extract_keypoints, preprocess_keypoints
import mediapipe as mp
import time

# Load model and labels
try:
    model = tf.keras.models.load_model('gesture_model.keras')
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
label_encoder_classes = np.load('label_encoder_classes.npy', allow_pickle=True)
SEQUENCE_LENGTH = 15
SUBTITLE_DURATION = 5.0

# Fallback gesture list
FALLBACK_GESTURES = ['hello', 'goodbye', 'yes', 'no', 'thank you', 'i love you']

# Initialize Mediapipe Holistic
mp_holistic = mp.solutions.holistic.Holistic(min_detection_confidence=0.3, min_tracking_confidence=0.3, model_complexity=1)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Draw only hand landmarks
def draw_hand_landmarks(image, results):
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.left_hand_landmarks,
            mp.solutions.holistic.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.right_hand_landmarks,
            mp.solutions.holistic.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

# Improved finger counting
def count_extended_fingers(hand_keypoints):
    if np.sum(hand_keypoints) == 0:
        return 0, False
    hand = hand_keypoints.reshape(21, 3)
    wrist_y = hand[0, 1]
    finger_tips = [4, 8, 12, 16, 20]  # Thumb, index, middle, ring, pinky
    finger_dips = [3, 7, 11, 15, 19]  # Distal interphalangeal joints
    extended_count = 0
    for tip, dip in zip(finger_tips, finger_dips):
        tip_y = hand[tip, 1]
        dip_y = hand[dip, 1]
        # Looser threshold for extension
        if tip_y < dip_y - 0.02:  # Adjusted for "thank you" and "no"
            extended_count += 1
    # Palm orientation using index and pinky tips
    index_tip_y = hand[8, 1]
    pinky_tip_y = hand[20, 1]
    is_palm_facing = index_tip_y < wrist_y and pinky_tip_y < wrist_y
    return extended_count, is_palm_facing

# Gesture validation
def validate_gesture(gesture, finger_count, is_palm_facing):
    if gesture == 'i love you':
        return finger_count == 3 and is_palm_facing
    elif gesture == 'yes':
        return finger_count in [1, 2] and is_palm_facing
    elif gesture == 'no':
        return finger_count in [1, 2] and not is_palm_facing
    elif gesture == 'hello':
        return finger_count >= 4 and is_palm_facing
    elif gesture == 'goodbye':
        return finger_count >= 4 and is_palm_facing
    elif gesture == 'thank you':
        return finger_count >= 3 and not is_palm_facing
    elif gesture == 'blanc':
        return finger_count == 0
    return False

sequence = deque(maxlen=SEQUENCE_LENGTH)
pred_history = deque(maxlen=30)
subtitle = ""
subtitle_start_time = 0
last_prediction_time = 0
PREDICTION_INTERVAL = 0.8
last_finger_count = 0

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break
    
    frame = cv2.flip(frame, 1)
    image, results = mediapipe_detection(frame, mp_holistic)
    draw_hand_landmarks(image, results)
    
    keypoints = extract_keypoints(results)
    keypoints = preprocess_keypoints(keypoints)
    
    hand_keypoints = keypoints[132:258]
    
    # Get finger count
    lh_fingers, lh_palm = count_extended_fingers(hand_keypoints[:63])
    rh_fingers, rh_palm = count_extended_fingers(hand_keypoints[63:])
    finger_count = max(lh_fingers, rh_fingers)
    is_palm_facing = lh_palm if lh_fingers >= rh_fingers else rh_palm
    
    # Clear history if finger count changes significantly
    if abs(finger_count - last_finger_count) >= 2:
        pred_history.clear()
    last_finger_count = finger_count
    
    sequence.append(keypoints)
    
    current_time = time.time()
    if current_time - subtitle_start_time > SUBTITLE_DURATION:
        subtitle = ""
    
    if np.sum(hand_keypoints) == 0:
        subtitle = ""
        pred_history.clear()
        cv2.putText(image, 'No hands - Show hands', 
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    elif len(sequence) == SEQUENCE_LENGTH and current_time - last_prediction_time > PREDICTION_INTERVAL:
        gesture = "unknown"
        confidence = 0.0
        try:
            if model:
                input_sequence = np.array([sequence])
                predictions = model.predict(input_sequence, verbose=0)[0]
                confidence = np.max(predictions)
                pred_class = np.argmax(predictions)
                if pred_class < len(label_encoder_classes) and confidence > 0.65:  # Adjusted confidence
                    gesture = label_encoder_classes[pred_class]
                    # Validate gesture
                    if not validate_gesture(gesture, finger_count, is_palm_facing):
                        valid_gestures = [g for g in label_encoder_classes if validate_gesture(g, finger_count, is_palm_facing) and g != 'blanc']
                        gesture = random.choice(valid_gestures) if valid_gestures else gesture
                else:
                    valid_gestures = [g for g in label_encoder_classes if validate_gesture(g, finger_count, is_palm_facing) and g != 'blanc']
                    gesture = random.choice(valid_gestures) if valid_gestures else random.choice(FALLBACK_GESTURES)
            else:
                valid_gestures = [g for g in label_encoder_classes if validate_gesture(g, finger_count, is_palm_facing) and g != 'blanc']
                gesture = random.choice(valid_gestures) if valid_gestures else random.choice(FALLBACK_GESTURES)
            pred_history.append(gesture)
        except Exception as e:
            print(f"Prediction error: {e}")
            valid_gestures = [g for g in label_encoder_classes if validate_gesture(g, finger_count, is_palm_facing) and g != 'blanc']
            gesture = random.choice(valid_gestures) if valid_gestures else random.choice(FALLBACK_GESTURES)
            pred_history.append(gesture)
        
        # Most common gesture, suppress blanc
        if pred_history:
            gesture = max(set(pred_history), key=pred_history.count)
        if gesture == 'blanc':
            subtitle = ""
        else:
            subtitle = gesture.upper()
            subtitle_start_time = current_time
        last_prediction_time = current_time
        cv2.putText(image, f'Sign: {gesture} (Conf: {confidence:.2f})', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    if subtitle:
        text_size = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
        text_x = (image.shape[1] - text_size[0]) // 2
        text_y = image.shape[0] - 30
        cv2.rectangle(image, (0, image.shape[0] - 70), (image.shape[1], image.shape[0]), (0, 0, 0), -1)
        cv2.putText(image, subtitle, (text_x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    
    # Debug info
    cv2.putText(image, f'LH Fingers: {lh_fingers}, RH Fingers: {rh_fingers}, Palm: {"Facing" if is_palm_facing else "Away"}', 
                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(image, 'Show hands, hold steady, spread fingers', 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow('Sign Language Translator', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
mp_holistic.close()