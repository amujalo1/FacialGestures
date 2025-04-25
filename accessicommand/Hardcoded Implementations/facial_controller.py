# facial_controller.py (Optimized Version)
import cv2
import mediapipe as mp
import pyautogui
import time
import math
# Removed: import numpy as np # Not used

# --- MediaPipe Face Mesh Setup ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# --- Camera and Screen Setup ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

screen_w, screen_h = pyautogui.size()

# --- Landmark Indices (Keep definitions as they are) ---
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
MOUTH_CORNER_INDICES = [61, 291]
MOUTH_VERTICAL_INDICES = [13, 14]
LEFT_EYEBROW_INDICES = [70, 63, 105, 66, 107]
RIGHT_EYEBROW_INDICES = [336, 296, 334, 293, 300]
FACE_OVAL_INDICES = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

# Combine all indices we want to draw
LANDMARKS_TO_DRAW = LEFT_EYE_INDICES + RIGHT_EYE_INDICES + MOUTH_CORNER_INDICES + MOUTH_VERTICAL_INDICES + LEFT_EYEBROW_INDICES + RIGHT_EYEBROW_INDICES

# --- State Variables ---
left_eye_closed_state = False
right_eye_closed_state = False
mouth_open_state = False
eyebrows_raised_state = False
head_tilt_left_state = False
head_tilt_right_state = False
both_eyes_closed_state = False # Initialize here

# --- Blink detection variables ---
left_eye_blinked = False # These are reset each frame anyway
right_eye_blinked = False
left_eye_previously_closed = False
right_eye_previously_closed = False
blink_cooldown = 0.3
last_left_blink_time = 0
last_right_blink_time = 0

# --- Key tracking variables ---
keys_currently_pressed = set()

# --- Thresholds ---
EAR_THRESHOLD = 0.20
MAR_THRESHOLD = 0.35
ERR_THRESHOLD = 1.34
BOTH_EYES_CLOSED_FRAMES = 2

HEAD_TILT_LEFT_MIN = -100
HEAD_TILT_LEFT_MAX = -160
HEAD_TILT_RIGHT_MIN = 100
HEAD_TILT_RIGHT_MAX = 160

CONSEC_FRAMES_BLINK = 2
CONSEC_FRAMES_MOUTH = 3
CONSEC_FRAMES_EYEBROW = 3
CONSEC_FRAMES_HEAD_TILT = 2

# --- Counters ---
left_blink_counter = 0
right_blink_counter = 0
mouth_open_counter = 0
eyebrow_raise_counter = 0
head_tilt_left_counter = 0
head_tilt_right_counter = 0
both_eyes_closed_counter = 0

# --- Calculation Functions (Keep as is) ---
def calculate_distance(p1, p2):
    # Using math.dist might be marginally faster if available (Python 3.8+)
    # return math.dist((p1.x, p1.y, p1.z), (p2.x, p2.y, p2.z))
    # But the current implementation is fine and compatible
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

def calculate_ear(eye_landmarks):
    try:
        # Check list length defensively? Optional. Assumes 6 landmarks are passed.
        v1 = calculate_distance(eye_landmarks[1], eye_landmarks[5])
        v2 = calculate_distance(eye_landmarks[2], eye_landmarks[4])
        h = calculate_distance(eye_landmarks[0], eye_landmarks[3])
        if h == 0: return 1.0 # Prevent division by zero, assume open
        ear = (v1 + v2) / (2.0 * h)
        return ear
    except IndexError:
        # This handles cases where landmarks might be missing temporarily
        return 1.0 # Assume open on error

def calculate_mar(landmarks, corner_indices, inner_vertical_indices):
    try:
        mouth_left_corner = landmarks[corner_indices[0]]
        mouth_right_corner = landmarks[corner_indices[1]]
        lip_upper_inner = landmarks[inner_vertical_indices[0]]
        lip_lower_inner = landmarks[inner_vertical_indices[1]]
        vertical_dist = calculate_distance(lip_upper_inner, lip_lower_inner)
        horizontal_dist = calculate_distance(mouth_left_corner, mouth_right_corner)
        if horizontal_dist == 0: return 0 # Prevent division by zero, assume closed
        mar = vertical_dist / horizontal_dist
        return mar
    except IndexError:
        return 0 # Assume closed on error

def calculate_err(landmarks, eyebrow_indices, eye_indices):
    """Calculate Eyebrow Raise Ratio (ERR)"""
    try:
        eyebrow_middle = landmarks[eyebrow_indices[2]]
        eyebrow_outer = landmarks[eyebrow_indices[4]]
        eye_top = landmarks[eye_indices[1]] # Using top of the correct eye index list
        vertical_dist = abs(eyebrow_middle.y - eye_top.y) # Use Y difference directly for verticality
        horizontal_dist = calculate_distance(eyebrow_middle, eyebrow_outer)
        if horizontal_dist == 0: return 0
        err = vertical_dist / horizontal_dist
        return err
    except IndexError:
        return 0

def calculate_head_tilt(landmarks, frame_width, frame_height):
    """Calculate head tilt angle in degrees"""
    try:
        chin = landmarks[152]
        forehead = landmarks[10]
        # Avoid redundant int conversions if only using for atan2
        dx = forehead.x - chin.x # Use normalized coordinates directly
        dy = forehead.y - chin.y
        # atan2(x, y) gives angle from positive Y axis, clockwise positive.
        # We want angle from vertical (Y axis), with right tilt positive.
        # Using atan2(dx, -dy) should give angle from negative Y axis (up), right positive.
        angle = math.degrees(math.atan2(dx, -dy)) # Adjust atan2 args for intuitive angle
        return angle
    except IndexError:
        return 0

# --- Action Functions (Keep as is) ---
def update_keys(actions_to_perform):
    """Update the keys being pressed based on the current actions."""
    global keys_currently_pressed
    # Iterate over a copy of the set to allow modification during iteration
    pressed_keys_copy = keys_currently_pressed.copy()

    # Keys to press
    for key, should_press in actions_to_perform.items():
        if should_press and key not in keys_currently_pressed:
            try:
                pyautogui.keyDown(key)
                keys_currently_pressed.add(key)
                print(f"Pressed: {key}")
            except Exception as e:
                 print(f"Error pressing key '{key}': {e}") # Add error logging

    # Keys to release
    for key in pressed_keys_copy:
        if not actions_to_perform.get(key, False): # If key is no longer True in actions
             try:
                 pyautogui.keyUp(key)
                 keys_currently_pressed.remove(key)
                 print(f"Released: {key}")
             except Exception as e:
                  print(f"Error releasing key '{key}': {e}") # Add error logging


def perform_shift_key_combo(key):
    """Perform a single shift+key press and release"""
    try:
        pyautogui.keyDown('shift')
        pyautogui.keyDown(key)
        time.sleep(0.05) # Delay might be necessary for OS/game registration
        pyautogui.keyUp(key)
        pyautogui.keyUp('shift')
        print(f"Single press: shift+{key}")
    except Exception as e:
        print(f"Error performing shift combo for '{key}': {e}") # Add error logging

def release_all_keys():
    """Release all keys that are currently pressed."""
    global keys_currently_pressed
    print("Releasing all held keys...")
    keys_to_release = list(keys_currently_pressed) # Create list before clearing
    for key in keys_to_release:
        try:
            pyautogui.keyUp(key)
            print(f"Released: {key}")
        except Exception as e:
            # Log error but continue trying to release others
            print(f"Error releasing key '{key}' during cleanup: {e}")
    keys_currently_pressed.clear()

# --- Print Initial Instructions (Keep as is) ---
print("Starting Facial Controller. Press 'q' to quit.")
# ... (rest of print statements) ...

# --- Main Loop ---
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("WARN: Failed to grab frame, retrying...")
            time.sleep(0.5) # Wait before trying again
            continue

        # --- Frame Preparation ---
        frame = cv2.flip(frame, 1)
        frame_height, frame_width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # --- MediaPipe Processing (Set non-writeable flag) ---
        rgb_frame.flags.writeable = False
        results = face_mesh.process(rgb_frame)
        rgb_frame.flags.writeable = True # Make writeable again (though often not strictly needed if only reading)

        # --- Convert back for drawing/display ---
        # It's generally okay to do this early if drawing happens later
        frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        # --- Initialize values for this frame ---
        ear_left_val = 1.0 # Default to open
        ear_right_val = 1.0
        mar_val = 0.0 # Default to closed
        err_left_val = 0.0
        err_right_val = 0.0
        avg_err = 0.0 # FIX: Initialize avg_err
        head_tilt_angle = 0.0 # FIX: Initialize head_tilt_angle
        current_time = time.time()

        # Reset blink flags for this frame
        left_eye_blinked = False
        right_eye_blinked = False

        # --- Actions Dictionary (Reset each frame) ---
        actions = {
            'a': False, 'd': False, 'j': False, 'k': False, 'space': False
        }

        # --- Process Landmarks if Face Detected ---
        if results.multi_face_landmarks:
            # Assume only one face
            landmarks = results.multi_face_landmarks[0].landmark

            # --- Calculations ---
            # Get points using list comprehensions (slightly more concise)
            person_left_eye_points = [landmarks[i] for i in LEFT_EYE_INDICES]
            person_right_eye_points = [landmarks[i] for i in RIGHT_EYE_INDICES]

            ear_left_val = calculate_ear(person_left_eye_points)   # Person's Left Eye (Screen Right)
            ear_right_val = calculate_ear(person_right_eye_points) # Person's Right Eye (Screen Left)

            # --- Update States and Counters ---

            # Left eye blink detection (person's right eye on screen left)
            is_left_currently_closed = ear_right_val < EAR_THRESHOLD
            if is_left_currently_closed:
                left_blink_counter += 1
            else:
                left_blink_counter = 0
            left_eye_closed_state = left_blink_counter >= CONSEC_FRAMES_BLINK

            # Detect blink transition
            if left_eye_closed_state and not left_eye_previously_closed and current_time - last_left_blink_time > blink_cooldown:
                left_eye_blinked = True
                last_left_blink_time = current_time
            left_eye_previously_closed = left_eye_closed_state # Update previous state

            # Right eye blink detection (person's left eye on screen right)
            is_right_currently_closed = ear_left_val < EAR_THRESHOLD
            if is_right_currently_closed:
                right_blink_counter += 1
            else:
                right_blink_counter = 0
            right_eye_closed_state = right_blink_counter >= CONSEC_FRAMES_BLINK

            # Detect blink transition
            if right_eye_closed_state and not right_eye_previously_closed and current_time - last_right_blink_time > blink_cooldown:
                right_eye_blinked = True
                last_right_blink_time = current_time
            right_eye_previously_closed = right_eye_closed_state # Update previous state


            # Mouth open detection
            mar_val = calculate_mar(landmarks, MOUTH_CORNER_INDICES, MOUTH_VERTICAL_INDICES)
            if mar_val > MAR_THRESHOLD:
                mouth_open_counter += 1
            else:
                mouth_open_counter = max(0, mouth_open_counter - 1) # Gradual decrease might feel better
            mouth_open_state = mouth_open_counter >= CONSEC_FRAMES_MOUTH

            # Both eyes closed detection
            if is_left_currently_closed and is_right_currently_closed:
                both_eyes_closed_counter += 1
            else:
                both_eyes_closed_counter = max(0, both_eyes_closed_counter - 1) # Gradual decrease
            both_eyes_closed_state = both_eyes_closed_counter >= BOTH_EYES_CLOSED_FRAMES

            # Eyebrow raise detection
            # Pass correct eye indices corresponding to the eyebrow side
            err_left_val = calculate_err(landmarks, LEFT_EYEBROW_INDICES, LEFT_EYE_INDICES)  # Left brow -> Left eye
            err_right_val = calculate_err(landmarks, RIGHT_EYEBROW_INDICES, RIGHT_EYE_INDICES) # Right brow -> Right eye
            avg_err = (err_left_val + err_right_val) / 2.0 # Use float division

            if avg_err > ERR_THRESHOLD:
                eyebrow_raise_counter += 1
            else:
                eyebrow_raise_counter = max(0, eyebrow_raise_counter - 1) # Gradual decrease
            eyebrows_raised_state = eyebrow_raise_counter >= CONSEC_FRAMES_EYEBROW

            # Head tilt detection
            head_tilt_angle = calculate_head_tilt(landmarks, frame_width, frame_height)
            is_tilt_left = HEAD_TILT_LEFT_MIN >= head_tilt_angle >= HEAD_TILT_LEFT_MAX
            is_tilt_right = HEAD_TILT_RIGHT_MIN <= head_tilt_angle <= HEAD_TILT_RIGHT_MAX

            if is_tilt_left:
                head_tilt_left_counter += 1
                head_tilt_right_counter = 0 # Reset opposite counter
            elif is_tilt_right:
                head_tilt_right_counter += 1
                head_tilt_left_counter = 0 # Reset opposite counter
            else: # Neither tilt range active
                head_tilt_left_counter = max(0, head_tilt_left_counter - 1) # Gradual decrease
                head_tilt_right_counter = max(0, head_tilt_right_counter - 1) # Gradual decrease

            head_tilt_left_state = head_tilt_left_counter >= CONSEC_FRAMES_HEAD_TILT
            head_tilt_right_state = head_tilt_right_counter >= CONSEC_FRAMES_HEAD_TILT

            # --- Determine Actions based on States ---

            # Momentary Blinks (handled separately as they are not sustained)
            if left_eye_blinked and not right_eye_blinked:
                perform_shift_key_combo('a') # Person's right eye blink
            elif right_eye_blinked and not left_eye_blinked:
                perform_shift_key_combo('d') # Person's left eye blink

            # Sustained Actions
            if mouth_open_state: actions['k'] = True
            if both_eyes_closed_state: actions['space'] = True
            if eyebrows_raised_state: actions['j'] = True
            if head_tilt_left_state: actions['a'] = True
            if head_tilt_right_state: actions['d'] = True


            # --- Visualization (Only if face detected) ---
            # Draw used landmarks
            for index in LANDMARKS_TO_DRAW:
                try:
                    point = landmarks[index]
                    # Denormalize coordinates
                    x = int(point.x * frame_width)
                    y = int(point.y * frame_height)
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1) # Green dots
                except IndexError:
                    pass # Landmark might be temporarily unavailable

            # Draw head tilt line
            try:
                chin = landmarks[152]
                forehead = landmarks[10]
                # Denormalize for drawing
                chin_x = int(chin.x * frame_width)
                chin_y = int(chin.y * frame_height)
                forehead_x = int(forehead.x * frame_width)
                forehead_y = int(forehead.y * frame_height)
                cv2.line(frame, (chin_x, chin_y), (forehead_x, forehead_y), (255, 0, 0), 2) # Blue line
            except IndexError:
                pass # Skip drawing if landmarks missing

        # --- End of Face Detected Block ---

        # --- Update Keys (Always run to handle releases) ---
        update_keys(actions)

        # --- Display Status Text (Always display, uses default values if no face) ---
        # Determine colors based on current state variables
        left_eye_color = (0, 0, 255) if left_eye_closed_state else (0, 255, 0)
        right_eye_color = (0, 0, 255) if right_eye_closed_state else (0, 255, 0)
        mouth_color = (0, 0, 255) if mouth_open_state else (0, 255, 0)
        eyebrow_color = (0, 0, 255) if eyebrows_raised_state else (0, 255, 0)
        head_tilt_left_color = (0, 0, 255) if head_tilt_left_state else (0, 255, 0)
        head_tilt_right_color = (0, 0, 255) if head_tilt_right_state else (0, 255, 0)
        both_eyes_color = (0, 0, 255) if both_eyes_closed_state else (0, 255, 0)

        cv2.putText(frame, f"L EYE: {ear_right_val:.2f} ({'Closed' if left_eye_closed_state else 'Open'})", (10, 30), # Display person's R eye status as L
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, left_eye_color, 2)
        cv2.putText(frame, f"R EYE: {ear_left_val:.2f} ({'Closed' if right_eye_closed_state else 'Open'})", (10, 60), # Display person's L eye status as R
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, right_eye_color, 2)
        cv2.putText(frame, f"MAR: {mar_val:.2f} ({'Open' if mouth_open_state else 'Closed'})", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, mouth_color, 2)
        cv2.putText(frame, f"ERR: {avg_err:.2f} ({'Raised' if eyebrows_raised_state else 'Normal'})", (10, 120), # avg_err is now initialized
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, eyebrow_color, 2)
        cv2.putText(frame, f"Head Tilt: {head_tilt_angle:.1f} ({'Left' if head_tilt_left_state else 'Right' if head_tilt_right_state else 'Center'})", (10, 150), # head_tilt_angle initialized
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, head_tilt_left_color if head_tilt_left_state else head_tilt_right_color if head_tilt_right_state else (0, 255, 0), 2)
        cv2.putText(frame, f"Both Eyes: {'Closed' if both_eyes_closed_state else 'Open'}", (10, 180),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, both_eyes_color, 2)

        active_keys_text = "Active Keys: " + ", ".join(sorted(keys_currently_pressed)) if keys_currently_pressed else "No keys active" # Sort for consistent display
        cv2.putText(frame, active_keys_text, (10, 210),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Blink feedback text remains the same

        # --- Display Frame ---
        cv2.imshow('Facial Gesture Controller', frame)

        # --- Exit Condition ---
        key = cv2.waitKey(1) & 0xFF # Use waitKey(1) for max responsiveness
        if key == ord('q'):
            print("'q' pressed, exiting.")
            break

except Exception as e:
    # Catch potential errors during the loop
    print(f"\n--- An Error Occurred in Main Loop ---")
    import traceback
    traceback.print_exc()
    print(f"Error details: {e}")
    print("--------------------------------------")

finally:
    # --- Cleanup ---
    print("\nCleaning up...")
    release_all_keys() # Ensure all keys are released on exit
    if cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()
    if 'face_mesh' in locals() and hasattr(face_mesh, 'close'):
        face_mesh.close() # Close MediaPipe resources if possible
    print("Facial Controller Finished.")