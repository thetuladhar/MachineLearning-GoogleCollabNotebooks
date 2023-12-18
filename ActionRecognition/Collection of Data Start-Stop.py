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
    '''mp_drawing.draw_landmarks(
        image,
        results.face_landmarks,
        mp_holistic.FACEMESH_TESSELATION,  # MODEL
        mp_drawing.DrawingSpec(color=(77, 89, 7), thickness=1, circle_radius=1),  # specks
        mp_drawing.DrawingSpec(color=(82, 39, 24), thickness=1, circle_radius=10)  # connections
    )
    # FACE CONTOURS
    mp_drawing.draw_landmarks(
        image,
        results.face_landmarks,
        mp_holistic.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2)
        # connection_drawing_spec = mp_drawing_styles.get_default_face_mesh_contours_style()
    )'''
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

DATA_PATH = os.path.join('MP_Data')
#actions = np.array(["Hello"])
# Actions that we try to detect

# Thirty videos worth of data
#no_sequences = 30
# Videos are going to be 20 frames in length
frame_count = 20

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
    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic:
            # Loop through sequences aka videos
            for sequence in range(start, stop + 1, 1):
                # Read feed


                for i in range(3,0,-1):
                    #print(i)
                    ret, frame = cap.read()

                    # Make detections
                    image, results = mediapipe_detection(frame, holistic)

                    # Draw landmarks
                    draw_styled_landmarks(image, results)
                    cv2.putText(image, 'GET READY {}'.format(i), (120, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(1000)

                # Loop through video length aka sequence length
                for frame_num in range(frame_count):
                    ret, frame = cap.read()

                    # Make detections
                    image, results = mediapipe_detection(frame, holistic)

                    # Draw landmarks
                    draw_styled_landmarks(image, results)
                    #for res in results.pose_landmarks.landmark:
                        #print("Hello",res)

                    # NEW Apply wait logic

                    if frame_num == frame_count-1 and sequence == stop:
                        cv2.putText(image, 'COMPLETED', (120, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3, cv2.LINE_AA)
                    elif frame_num == frame_count-1:
                        cv2.putText(image, 'DONE', (120, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3, cv2.LINE_AA)
                        #cv2.waitKey(4000)
                    if frame_num < frame_count-1:
                        cv2.putText(image, 'COLLECTING..', (120, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3, cv2.LINE_AA)
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (30, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
                        # Show to screen
                        cv2.imshow('OpenCV Feed', image)

                    else:
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (30, 30),

                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
                        # Show to screen
                        cv2.imshow('OpenCV Feed', image)
                        cv2.waitKey(1000)

                    # NEW Export keypoints
                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                    np.save(npy_path, keypoints)

                    # Break gracefully

                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        exit()

cap.release()
cv2.destroyAllWindows()
exit()