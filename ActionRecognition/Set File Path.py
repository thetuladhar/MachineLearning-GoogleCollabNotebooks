import cv2
import os
import time
import numpy as np
from matplotlib import pyplot as plt
import mediapipe as mp

mp_holistic = mp.solutions.holistic# Holistic model
mp_drawing = mp.solutions.drawing_utils

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_drawing_styles = mp.solutions.drawing_styles

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = holistic.process(image)

    # Draw landmark annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    #FACE TESSALATION(lines in face)
    mp_drawing.draw_landmarks(
        image,
        results.face_landmarks,
        mp_holistic.FACEMESH_TESSELATION,#MODEL
        mp_drawing.DrawingSpec(color=(77, 89, 7), thickness=1, circle_radius=1),#specks
        mp_drawing.DrawingSpec(color=(82, 39, 24), thickness=1, circle_radius=10)#connections
     )
    #FACE CONTOURS
    mp_drawing.draw_landmarks(
       image,
        results.face_landmarks,
        mp_holistic.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2)
        #connection_drawing_spec = mp_drawing_styles.get_default_face_mesh_contours_style()
    )
    #RIGHT HAND
    mp_drawing.draw_landmarks(
        image,
        results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(77, 89, 7), thickness=2, circle_radius=5),  # specks
        mp_drawing.DrawingSpec(color=(237, 248, 255), thickness=2, circle_radius=2)  # skeleton
    )
    #LEFT HAND
    mp_drawing.draw_landmarks(
        image,
        results.left_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(77, 89, 7), thickness=2, circle_radius=5),  # specks
        mp_drawing.DrawingSpec(color=(237, 248, 255), thickness=2, circle_radius=2)  # skeleton
    )
    #POSE
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(77, 89, 7), thickness=2, circle_radius=5),#specks
        mp_drawing.DrawingSpec(color=(237, 248, 255), thickness=2, circle_radius=2)#skeleton
    )
    try:
        #pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten()
        #    if results.pose_landmarks
        #    else np.zeros(132)#error handler
        #print(len(pose))

        def extract_keypoints(results):
            pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                             results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
            face = np.array([[res.x, res.y, res.z] for res in
                             results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(
                468 * 3)
            lh = np.array([[res.x, res.y, res.z] for res in
                           results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(
                21 * 3)
            rh = np.array([[res.x, res.y, res.z] for res in
                           results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
                21 * 3)
            return np.concatenate([pose, face, lh, rh])

        #SETTING UP FOLDERS FOR DATA COLLECTION

        # Path for exported data, numbpie.py arrays
        DATA_PATH = os.path.join('MP_Data')

        # Actions that we try to detect
        actions = np.array(['Hello','Thanks',"IloveYou","J","Z"])

        # Thirty videos worth of data
        no_sequences = 30
        # Videos are going to be 30 frames in length
        sequence_length = 30
    
        for action in actions:
            for sequence in range(no_sequences):
                try:
                    os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))#Make directiories
                except:
                    pass
    except:
        pass

    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))
    if cv2.waitKey(10) & 0xFF == ord('q'):
      break
cap.release()
cv2.destroyAllWindows()