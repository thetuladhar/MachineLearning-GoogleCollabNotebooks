# import the trained model

# Make sure Tensorflow is version 2.4.1.Model was trained in this verision so didnt want to risk any errors
# import tensorflow as tf
# print(tf.__version__)

# removing DS store object, annoying as heck on a MAC
# cd to the file path in terminal then type
# find . -name '.DS_Store' -type f -delete

# Import the trained model
from tensorflow import keras

model = keras.models.load_model('modelChildMeshWeight.h5')

# check the model structure
# print(model.summary())

# Dependencies
import cv2
import os
import numpy as np
import mediapipe as mp

mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilitiesimport cv2


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
    )'''
    # FACE CONTOURS
    '''mp_drawing.draw_landmarks(
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
    if results.pose_landmarks and results.right_hand_landmarks and (
            results.pose_landmarks.landmark[23].y < results.right_hand_landmarks.landmark[12].y):
        # print(results.pose_landmarks.landmark[23].y,'\t',results.right_hand_landmarks.landmark[12].y)
        lh = np.zeros(21 * 3)
        rh = np.zeros(21 * 3)
        pose = np.zeros(21 * 3)
        face = np.zeros(21 * 3)
    else:

        lh = np.array([[res.x, res.y, res.z] for res in
                       results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(
            21 * 3)
        rh = np.array([[res.x, res.y, res.z] for res in
                       results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
            21 * 3)
        pose = np.array([[res.x, res.y, res.z] for res in
                         results.pose_landmarks.landmark]).flatten() if results.pose_landmarks and results.right_hand_landmarks else np.zeros(
            33 * 3)
        face = np.array([[res.x, res.y, res.z] for res in
                         results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)

    return np.concatenate([lh, rh, pose, face])


# funtion to render Probabilities
# (BGR colors)

testMotion = [(150, 110, 35), (211, 178, 21)]
colors2 = testMotion * 2


def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)

        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (7, 255, 0), 2,
                    cv2.LINE_AA)
        if prob * 100 >= 90:
            cv2.putText(output_frame, str(round(prob * 100, 3)), (200 + 500, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 2, cv2.LINE_AA)

        '''cv2.putText(output_frame,str(res[np.argmax(res)]), (0+500, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)'''

    return output_frame


dirActiions = '/Users/Tuladhar/PycharmProjects/MediaPipe/ActionRecognition/MP_Child'
listActions = os.listdir(dirActiions)
listActions.sort()
# print(list)

actionsMotion = np.array(listActions)

# 1. New detection variables
sequenceMotion = []
sentenceMotion = []
threshold = 0.90


cap = cv2.VideoCapture(0)  # use 1 for external camera
# Set mediapipe model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        # print(results)

        # Draw landmarks
        draw_styled_landmarks(image, results)

        # 2. Prediction logic
        keypoints = extract_keypoints(results)

        # INITIATING EMPTY LISTS
        rel_R_Motion = []
        rel_L_Motion = []
        pos_rel_Motion = []
        # was for i in range(66, 126, 3):
        for i in range(1, 21, 1):
            # keypoint[0] - keypoint[2] are x,y,z coordinates of Landmark[0] base point on left hand

            # LEFT_BASE_X = 0
            # LEFT_BASE_Y = 1
            # LEFT_BASE_Z = 2

            # x2 = keypoints[LEFT_BASE_X] - keypoints[i * 3]
            # y2 = keypoints[LEFT_BASE_Y] - keypoints[(i * 3) + 1]
            # z2 = keypoints[LEFT_BASE_Z] - keypoints[(i * 3) + 2]

            # keypoint[63] - keypoint[65] are x,y,z coordinates of Landmark[0] base point on right hand
            RIGHT_BASE_X = 63
            RIGHT_BASE_Y = 64
            RIGHT_BASE_Z = 65

            x1 = keypoints[RIGHT_BASE_X] - keypoints[(i * 3) + RIGHT_BASE_X]
            y1 = keypoints[RIGHT_BASE_Y] - keypoints[(i * 3) + RIGHT_BASE_Y]
            z1 = keypoints[RIGHT_BASE_Z] - keypoints[(i * 3) + RIGHT_BASE_Z]

            # keypoints[126] - keypoints[128] are x,y,z coordinates of landmark[0] which is on nose for pose
            POSE_NOSE_X = 126
            POSE_NOSE_Y = 127
            POSE_NOSE_Z = 128

            # pos_x = keypoints[POSE_NOSE_X] - keypoints[(i * 3) + LEFT_BASE_X]
            # pos_y = keypoints[POSE_NOSE_Y] - keypoints[(i * 3) + LEFT_BASE_Y]
            # pos_z = keypoints[POSE_NOSE_Z] - keypoints[(i * 3) + LEFT_BASE_Z]

            pos_x2 = keypoints[POSE_NOSE_X] - keypoints[(i * 3) + RIGHT_BASE_X]
            pos_y2 = keypoints[POSE_NOSE_Y] - keypoints[(i * 3) + RIGHT_BASE_Y]
            pos_z2 = keypoints[POSE_NOSE_Z] - keypoints[(i * 3) + RIGHT_BASE_Z]

            rel_R_Motion.append([x1, y1, z1])
            # rel_L.append([x2, y2, z2])
            # (pos_x, pos_y, pos_z,)
            pos_rel_Motion.append([pos_x2, pos_y2, pos_z2])

        arrayR = np.array(rel_R_Motion).flatten()
        arrayL = np.array(rel_L_Motion).flatten()
        pos_array = np.array(pos_rel_Motion).flatten()


        # arrayL,
        keypoints2 = np.concatenate([arrayR, pos_array])

        # the child folder has array spliced from specified index
        # child_file = keypoints[:SLICE_INDEX]

        sequenceMotion.append(keypoints2)
        sequenceMotion = sequenceMotion[-20:]  # WAS 24

        res = model.predict(np.expand_dims(sequenceMotion, axis=0))[0]
        # print(res, "*****",np.argmax(res), "*****",actions[np.argmax(res)])
        # qprint(actions[np.argmax(res)],sign_counter)
        # 3. Viz logic
        if res[np.argmax(res)] > threshold:
            if len(sentenceMotion) > 0:
                if actionsMotion[np.argmax(res)] != sentenceMotion[-1]:
                    sentenceMotion.append(actionsMotion[np.argmax(res)])
            else:
                sentenceMotion.append(actionsMotion[np.argmax(res)])

        if len(sentenceMotion) > 5:
            sentenceMotion = sentenceMotion[-5:]
        # print(len(res))
        # Viz probabilities
        image = prob_viz(res, actionsMotion, image, colors2)

        # cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentenceMotion), (500, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3, cv2.LINE_AA)

        # Show to screen
        cv2.imshow('testRELAlpha', image)

        # Break gracefully
        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit()
            # break
cap.release()
cv2.destroyAllWindows()
