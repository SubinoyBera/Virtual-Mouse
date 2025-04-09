import numpy as np
import random
import cv2 as cv
import mediapipe as mp
from utils import *
from pynput.mouse import Button, Controller
mouse= Controller()

# Initializing mediapipe
mp_hands= mp.solutions.hands
hands= mp_hands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
)

def find_fingertip(processed_frame):
    # Extracts the coordinates of the index fingertip from the processed hand landmarks.
    if processed_frame.multi_hand_landmarks:
        hand_landmarks= processed_frame.multi_hand_landmarks[0]
        return hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    
    return None

def is_leftclick(landmarks_list, thumb_index_dist):
    # Detects if the left-click gesture is being performed.
    return(
        get_angle(landmarks_list[5], landmarks_list[6], landmarks_list[8])<50 and
        get_angle(landmarks_list[9], landmarks_list[10], landmarks_list[12])>90 and
        thumb_index_dist>50
    )

def is_rightclick(landmarks_list, thumb_index_dist):
    # Detects if the right-click gesture is being performed.
    return(
        get_angle(landmarks_list[5], landmarks_list[6], landmarks_list[8])>90 and
        get_angle(landmarks_list[9], landmarks_list[10], landmarks_list[12])<50 and
        thumb_index_dist>50
    )

def is_doubleclick(landmarks_list, thumb_index_dist):
    # Detects if the double-click gesture is being performed.
    return(
    get_angle(landmarks_list[5], landmarks_list[6], landmarks_list[8])<50 and
    get_angle(landmarks_list[9], landmarks_list[10], landmarks_list[12])<50 and
    thumb_index_dist>50
    )

def is_screenshot(landmarks_list, thumb_index_dist):
    # Detects if the screenshot gesture is being performed.
    return(
        get_angle(landmarks_list[5], landmarks_list[6], landmarks_list[8])<50 and
        get_angle(landmarks_list[9], landmarks_list[10], landmarks_list[12])<50 and
        thumb_index_dist<50
    )

def detect_gestures(frame, landmarks_list, processed_frame):
    """
    Detects hand gestures and performs corresponding mouse actions.

    Args:
        frame: The current video frame being processed.
        landmarks_list: A list of hand landmark coordinates.
        processed_frame: The processed frame data containing hand landmarks.
    """
    # Check if there are enough landmarks to perform gesture detection
    if len(landmarks_list)>=21:
        index_fingertip= find_fingertip(processed_frame)
        thumb_index_dist= get_distance([landmarks_list[4], landmarks_list[5]])
        indexfinger_angle_pos= get_angle(landmarks_list[5], landmarks_list[6], landmarks_list[8])
        # Move mouse pointer
        if thumb_index_dist<50 and indexfinger_angle_pos>90:
            move_mouse(index_fingertip)
        # Left click
        elif is_leftclick(landmarks_list, thumb_index_dist):
            mouse.press(Button.left)
            cv.putText(frame, "Left Click", (50,50), cv.FONT_HERSHEY_SCRIPT_SIMPLEX, 1,(255,255,255,0),2)
            mouse.release(Button.left)
        # Right click
        elif is_rightclick(landmarks_list, thumb_index_dist):
            mouse.press(Button.right)
            cv.putText(frame, "Right Click", (50,50), cv.FONT_HERSHEY_SCRIPT_SIMPLEX, 1,(255,255,255,0),2)
            mouse.release(Button.right)
        # Double click
        elif is_doubleclick(landmarks_list, thumb_index_dist):
            pyautogui.doubleClick()
            cv.putText(frame, "Double Click", (50,50), cv.FONT_HERSHEY_SCRIPT_SIMPLEX, 1,(255,255,255,0),2)
        # Screenshot
        elif is_screenshot(landmarks_list, thumb_index_dist):
            img= pyautogui.screenshot()
            label= random.randint(1, 1000)
            img.save(f"my_screenshot{label}.png")
            cv.putText(frame, "Screenshot Taken", (50,50), cv.FONT_HERSHEY_SCRIPT_SIMPLEX, 1,(255,255,255,0),2)

# Main function to capture video from the webcam, process hand gestures and perform mouse actions 
# based on detected gestures.
def main():
    cap= cv.VideoCapture(0)
    draw= mp.solutions.drawing_utils
    try:
        while cap.isOpened():
            ret,frame= cap.read()
            if not ret:
                break
            # Flip the frame horizontally to match the camera view
            frame= cv.flip(frame, 1)
            frameRGB= cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            # Process the frame to detect hands
            processed_frame= hands.process(frameRGB)

            # Initialize an empty list to store the hand landmarks
            landmarks_list=[]
            # If there are any hand landmarks, draw them on the frame
            if processed_frame.multi_hand_landmarks:
                hand_landmarks= processed_frame.multi_hand_landmarks[0]
                draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Iterate over the landmarks and store the coordinates in the list
                for lm in hand_landmarks.landmark:
                    landmarks_list.append((lm.x, lm.y))

            # Call the detect_gestures function to detect the current gesture
            detect_gestures(frame, landmarks_list, processed_frame)
            
            # Show the frame
            cv.imshow('Frame', frame)
            # Wait for the user to press a key
            if cv.waitKey(1) and 0xFF==ord('q'):
                break

    except Exception:
        raise

    finally:
        # Release the capture and close all windows
        cap.release()
        cv.destroyAllWindows()


if __name__=='__main__':
    main()