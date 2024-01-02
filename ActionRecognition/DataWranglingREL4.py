import os
import numpy as np
import math
#removing DS store object, annoying as heck on a MAC
#cd to the file path in terminal then type
#find . -name '.DS_Store' -type f -delete

#The below three values are customizable
parent_path='/Users/Tuladhar/PycharmProjects/MediaPipe/ActionRecognition/MP_Data'
child_path='/Users/Tuladhar/PycharmProjects/MediaPipe/ActionRecognition/MP_Data_REL5'

SLICE_INDEX = 225

signsList= os.listdir(parent_path)
signsList.sort()
#print(signsList)

for sign in signsList:
    sequenceLength = len(os.listdir(os.path.join(parent_path,sign)))
    for sequence in range(sequenceLength):
            os.makedirs(os.path.join(child_path, sign, str(sequence)))  # Make directories
            frameLength = len(os.listdir(os.path.join(parent_path, sign, str(sequence))))
            #print(frameLength)
            for frame_num in range(frameLength):
                parent_file = os.path.join(parent_path, sign, str(sequence), str(frame_num) + ".npy")
                #print(parent_file)
                #keypoints represents parent file data
                keypoints = np.load(f'{parent_file}')

                child_file_path = os.path.join(child_path, sign, str(sequence), str(frame_num))
                # the child folder has array spliced from specified index

                rel_R = []
                #was for i in range(66, 126, 3):
                for i in range(63,126,3):
                    x = keypoints[i]
                    y = keypoints[i+1]
                    z = keypoints[i+2]

                    rel_R.append([x, y, z])


                weight=7
                arrayR= np.array(rel_R* weight).flatten()

                #print(arrayR)
                #the child folder has array spliced from specified index
                child_file = keypoints[:SLICE_INDEX]

                concat_file=np.concatenate([child_file,arrayR])
                #Right hand weighted 8x TEST

                np.save(child_file_path, concat_file)
print("Complete! All the best in Training.")