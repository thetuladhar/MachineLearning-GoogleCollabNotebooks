import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
import mediapipe as mp

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilitiesimport cv2

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_drawing_styles = mp.solutions.drawing_styles

def mediapipe_detection(image, model):
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = holistic.process(image)
    # Draw landmark annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results):
    # FACE TESSALATION(lines in face)
    # RIGHT HAND
    mp_drawing.draw_landmarks(
        image,
        results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(77, 89, 7), thickness=2, circle_radius=5),  # specks
        mp_drawing.DrawingSpec(color=(237, 248, 255), thickness=2, circle_radius=2)  # skeleton
    )
    # LEFT HAND
    mp_drawing.draw_landmarks(
        image,
        results.left_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(77, 89, 7), thickness=2, circle_radius=5),  # specks
        mp_drawing.DrawingSpec(color=(237, 248, 255), thickness=2, circle_radius=2)  # skeleton
    )
    # POSE
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(77, 89, 7), thickness=2, circle_radius=5),  # specks
        mp_drawing.DrawingSpec(color=(237, 248, 255), thickness=2, circle_radius=2)  # skeleton
    )

def extract_keypoints(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    pose = np.array([[res.x, res.y, res.z] for res in
                     results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 3)
    face = np.array([[res.x, res.y, res.z] for res in
                     results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    return np.concatenate([lh, rh, pose, face])

DATA_PATH = os.path.join('MP_AlphaP')
#actions = np.array(["Hello"])
# Actions that we try to detect

# Thirty videos worth of data
#no_sequences = 30
# Videos are going to be 24 frames in length
frame_count = 1 #usedToBe24
cap = None
while True:
    action = input("Enter word (Type exit to stop:)")
    if action == "exit":
        break
    start = int(input("Enter number for start sequences:"))

    stop = int(input("Enter number for end sequences:"))

    for sequence in range(start,stop+1,1):
            try:
                os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))  # Make directiories
            except:
                pass

    # For webcam input:
    cap = cv2.VideoCapture(1)

    with mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic:
            # Loop through sequences aka videos
            for sequence in range(start, stop + 1, 1):
                # Loop through video length aka sequence length
                for frame_num in range(frame_count):
                    # Read feed
                    ret, frame = cap.read()

                    # Make detections
                    image, results = mediapipe_detection(frame, holistic)

                    # Draw landmarks
                    draw_styled_landmarks(image, results)
                    #for res in results.pose_landmarks.landmark:
                        #print("Hello",res)

                    # NEW Apply wait logic

                    if sequence == stop:
                        cv2.putText(image, 'COMPLETED {}/{}'.format(sequence,stop), (120, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3, cv2.LINE_AA)
                    if sequence >= 0 and sequence < stop:
                        cv2.putText(image, 'COLLECTING...{}/{}'.format(sequence,stop), (120, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3, cv2.LINE_AA)
                        cv2.putText(image, 'Collecting frames for {}'.format(action), (30, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
                        # Show to screen
                        cv2.imshow('OpenCV Feed', image)
                        cv2.waitKey(10)

                    else:
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (30, 30),

                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)
                        # Show to screen
                        cv2.imshow('OpenCV Feed', image)


                    # NEW Export keypoints
                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                    np.save(npy_path, keypoints)

                    # Break gracefully

                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        exit()
                        #break

cap.release()
cv2.destroyAllWindows()
exit()