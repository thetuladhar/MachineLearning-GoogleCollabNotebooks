#import the trained model

#Make sure Tensorflow is version 2.4.1.Model was trained in this verision so didnt want to risk any errors
#import tensorflow as tf
#print(tf.__version__)

#Import the trained model
from tensorflow import keras
modelMotion = keras.models.load_model('modelMotionMesh7.h5')

#check the model structure
#print(model.summary())

#Dependencies
import cv2
import os
import numpy as np
import mediapipe as mp

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilitiesimport cv2

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
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    pose = np.array([[res.x, res.y, res.z] for res in
                     results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 3)
    face = np.array([[res.x, res.y, res.z] for res in
                     results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)

    return np.concatenate([lh, rh, pose, face])


#funtion to render Probabilities
#(BGR colors)
colors = [(150, 110, 35),
          (211, 178, 21),
          (0, 215, 255),
          (150, 110, 35),
          (211, 178, 21),
          (0, 215, 255),
          (211, 178, 21),
          (0, 215, 255),
          (150, 110, 35),
          (211, 178, 21),
          (0, 215, 255)]
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)
        if prob*100>=90:
            cv2.putText(output_frame, str(round(prob*100,3)), (0 + 500, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 2,cv2.LINE_AA)

        '''cv2.putText(output_frame,str(res[np.argmax(res)]), (0+500, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)'''

    return output_frame


dir='/Users/Tuladhar/PycharmProjects/MediaPipe/ActionRecognition/MP_Data_REL6'
list=os.listdir(dir)
list.sort()
#print(list)

actions = np.array(list)

# 1. New detection variables
sequence = []
sentence = []
threshold = 0.98

cap = cv2.VideoCapture(0)
# Set mediapipe model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        #print(results)

        # Draw landmarks
        draw_styled_landmarks(image, results)

        # 2. Prediction logic
        keypoints = extract_keypoints(results)

        SLICE_INDEX=225

        # INITIATING EMPTY LISTS
        rel_R = []
        rel_L = []

        RIGHT_BASE_X = 63
        RIGHT_BASE_Y = 64
        RIGHT_BASE_Z = 65

        LEFT_BASE_X = 0
        LEFT_BASE_Y = 1
        LEFT_BASE_Z = 2

        for j in range(0, 63, 3):
            MTlistR = []
            MTlistL = []
            for i in range(0, 21, 1):
                Main_x1 = j + RIGHT_BASE_X
                Main_y1 = j + RIGHT_BASE_Y
                Main_z1 = j + RIGHT_BASE_Z

                Main_x2 = j + LEFT_BASE_X
                Main_y2 = j + LEFT_BASE_Y
                Main_z2 = j + LEFT_BASE_Z

                if Main_x1 != ((i * 3) + RIGHT_BASE_X):
                    x1 = keypoints[Main_x1] - keypoints[(i * 3) + RIGHT_BASE_X]
                    y1 = keypoints[Main_y1] - keypoints[(i * 3) + RIGHT_BASE_Y]
                    z1 = keypoints[Main_z1] - keypoints[(i * 3) + RIGHT_BASE_Z]
                    MTlistR.append([x1, y1, z1])

                if Main_x2 != ((i * 3) + LEFT_BASE_X):
                    x2 = keypoints[Main_x2] - keypoints[(i * 3) + LEFT_BASE_X]
                    y2 = keypoints[Main_y2] - keypoints[(i * 3) + LEFT_BASE_Y]
                    z2 = keypoints[Main_z2] - keypoints[(i * 3) + LEFT_BASE_Z]
                    MTlistL.append([x2, y2, z2])

            rel_R.append(MTlistR)
            rel_L.append(MTlistL)
        arrayR = np.array(rel_R).flatten()
        arrayL = np.array(rel_L).flatten()

        keypoints2 = np.concatenate([arrayR, arrayR, arrayR, arrayL])

        sequence.append(keypoints2)
        sequence = sequence[-24:]

        if len(sequence) == 24:
            res = modelMotion.predict(np.expand_dims(sequence, axis=0))[0]
            #print(res, "*****",np.argmax(res), "*****",actions[np.argmax(res)])

            # 3. Viz logic
            if res[np.argmax(res)] > threshold:
                if len(sentence) > 0:
                    if actions[np.argmax(res)] != sentence[-1]:
                        sentence.append(actions[np.argmax(res)])
                else:
                    sentence.append(actions[np.argmax(res)])

            if len(sentence) > 5:
                sentence = sentence[-5:]
            #print(len(res))
            # Viz probabilities
            image = prob_viz(res, actions, image, colors)

        #cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3, cv2.LINE_AA)

        # Show to screen
        cv2.imshow('OpenCV Feed', image)

        # Break gracefully
        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit()
            #break
cap.release()
cv2.destroyAllWindows()