import cv2
import os
import numpy as np
import mediapipe as mp

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilitiesimport cv2

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
    mp_drawing.draw_landmarks(
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
    )
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

def face_point(n):
    return np.array([[results.face_landmarks.landmark[n].x, results.face_landmarks.landmark[n].y,
                      results.face_landmarks.landmark[n].z]]).flatten() if results.face_landmarks and \
                                                                            results.face_landmarks.landmark[
                                                                                n] else np.zeros(1 * 3)
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    #ONE FOREHEAD POINT TEST
    face=face_point(10)
    #face = np.array([[results.face_landmarks.landmark[10].x, results.face_landmarks.landmark[10].y, results.face_landmarks.landmark[10].z]]).flatten() if results.face_landmarks and results.face_landmarks.landmark[10] else np.zeros(1*3)

    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.face_landmarks and results.face_landmarks.landmark[10] and results.right_hand_landmarks else np.zeros(21*3)
    print(results.pose_landmarks.landmark.visibility)
    return np.concatenate([pose,face,lh,rh])

DATA_PATH = os.path.join('MP_Data')
#actions = np.array(["Hello"])
# Actions that we try to detect

# Thirty videos worth of data
#no_sequences = 30
# Videos are going to be 30 frames in length
sequence_length = 30
cap = None
while True:
    action = input("Enter word (Type exit to stop:)")
    if action == "exit":
        break
    no_sequences = int(input("Enter number of sequences:"))
    #if no_sequences > 30:
    #    print("Sorry,For Beta purposes,Enter a number that is less than 31")
    #    continue
    for sequence in range(no_sequences):
            try:
                os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))  # Make directiories
            except:
                pass

    # For webcam input:
    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic:
            # Loop through sequences aka videos
            for sequence in range(no_sequences):
                # Loop through video length aka sequence length
                for frame_num in range(sequence_length):
                    # Read feed
                    ret, frame = cap.read()

                    # Make detections
                    image, results = mediapipe_detection(frame, holistic)

                    # Draw landmarks
                    draw_styled_landmarks(image, results)
                    #for res in results.pose_landmarks.landmark:
                        #print("Hello",res)

                    # NEW Apply wait logic
                    if frame_num == 0:
                        cv2.putText(image, 'STARTING COLLECTION', (120, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 4, cv2.LINE_AA)
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15, 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1, cv2.LINE_AA)
                        # Show to screen
                        cv2.imshow('OpenCV Feed', image)
                        cv2.waitKey(2000)
                    else:
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15, 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1, cv2.LINE_AA)
                        # Show to screen
                        cv2.imshow('OpenCV Feed', image)

                    # NEW Export keypoints
                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                    np.save(npy_path, keypoints)

                    # Break gracefully
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
if cap:
    cap.release()
    cv2.destroyAllWindows()