# Dependencies
import cv2
import os
import numpy as np
import mediapipe as mp
from tensorflow import keras
import string

ALPHA_PROB_THRESHOLD = 0.95
MOTION_PROB_THRESHOLD = 0.95

mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilitiesimport cv2

alphamodel = keras.models.load_model('modelAlphaMesh6.h5')
wordmodel = keras.models.load_model('modelChildMeshWeight.h5')

alphabet_list = list(string.ascii_uppercase)
word_list=["No","Sorry","ThankYou","Yes"]
motionsequence = []
alphasequence = []
def mediapipe_detection(image,model):
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model.process(image)
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


def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (7, 255, 0), 2,
                    cv2.LINE_AA)
        if prob * 100 >= 90:
            cv2.putText(output_frame, str(round(prob * 100, 3)), (0 + 500, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 2, cv2.LINE_AA)

        '''cv2.putText(output_frame,str(res[np.argmax(res)]), (0+500, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)'''

    return output_frame

def alphabetPredict(keypoints):
    # INITIATING EMPTY LISTS
    rel_R = []
    RIGHT_BASE_X = 63
    RIGHT_BASE_Y = 64
    RIGHT_BASE_Z = 65
    # was for i in range(66, 126, 3):
    for j in range(0, 63, 3):
        MTlist = []
        for i in range(0, 21, 1):

            Main_x = j + RIGHT_BASE_X
            Main_y = j + RIGHT_BASE_Y
            Main_z = j + RIGHT_BASE_Z

            if Main_x != ((i * 3) + RIGHT_BASE_X):
                x1 = keypoints[Main_x] - keypoints[(i * 3) + RIGHT_BASE_X]
                y1 = keypoints[Main_y] - keypoints[(i * 3) + RIGHT_BASE_Y]
                z1 = keypoints[Main_z] - keypoints[(i * 3) + RIGHT_BASE_Z]

                MTlist.append([x1, y1, z1])
        rel_R.append(MTlist)

    arrayR = np.array(rel_R).flatten()
    # print(arrayR[0:10],arrayR[11:20],arrayR[21:30])
    # print(arrayR[30:40], arrayR[41:50], arrayR[51:])

    # the child folder has array spliced from specified index
    # not using in the current code
    # child_file = parent_file[:SLICE_INDEX]

    keypoints2 = np.concatenate([arrayR])
    # the child folder has array spliced from specified index
    # child_file = keypoints[:SLICE_INDEX]

    alphasequence.append(keypoints2)
    sequence = alphasequence[-1:]  # WAS 24

    res = alphamodel.predict(np.expand_dims(sequence, axis=0))[0]
    prediction=alphabet_list[np.argmax(res)]
    print(prediction)



def wordPredict(keypoints):

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
    #arrayL = np.array(rel_L_Motion).flatten()
    pos_array = np.array(pos_rel_Motion).flatten()

    # arrayL,
    keypoints2 = np.concatenate([arrayR, pos_array])

    motionsequence.append(keypoints2)
    sequenceMotion = motionsequence[-20:]  # WAS 24

    res = wordmodel.predict(np.expand_dims(sequenceMotion, axis=0))[0]
    prediction=word_list[np.argmax(res)]

    if res[np.argmax(res)] > MOTION_PROB_THRESHOLD:
        print(prediction)

#initiate
f_count = 0
w_count = 0
side = ""
MODE_COUNTER_THRESHOLD = 20

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

        ThumbRHX=keypoints[69]
        PinkyRHX=keypoints[114]
        MiddleRHY=keypoints[91]
        WristRHY=keypoints[64]
        mode_counter= ""
        if side !="Fingers" and (MiddleRHY < WristRHY):
            if ThumbRHX < PinkyRHX:
                w_count =0
                if f_count < MODE_COUNTER_THRESHOLD:
                    f_count = f_count + 1
                    mode_counter = MODE_COUNTER_THRESHOLD - f_count
                elif f_count == MODE_COUNTER_THRESHOLD:
                    f_count = 0
                    side= "Fingers"
            else:
                f_count = 0
        elif side != "Words":
            if ThumbRHX > PinkyRHX:
                f_count =0
                if w_count < MODE_COUNTER_THRESHOLD:
                    w_count = w_count + 1
                    mode_counter = MODE_COUNTER_THRESHOLD - w_count
                elif w_count == MODE_COUNTER_THRESHOLD:
                    w_count=0
                    side = "Words"
            else:
                w_count=0

        if side=="Fingers":
            alphabetPredict(keypoints)
        elif side == "Words":
            wordPredict(keypoints)
        else:
            print("NOT FINGERS")


        # Show to screen
        cv2.imshow('testRELAlpha', image)

        # Break gracefully
        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit()
            # breakq
cap.release()
cv2.destroyAllWindows()

print("Hello")