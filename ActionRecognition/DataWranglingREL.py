import os
import numpy as np
import math
#removing DS store object, annoying as heck on a MAC
#cd to the file path in terminal then type
#find . -name '.DS_Store' -type f -delete

#The below three values are customizable
parent_path='/Users/Tuladhar/PycharmProjects/MediaPipe/ActionRecognition/MP_Data'
child_path='/Users/Tuladhar/PycharmProjects/MediaPipe/ActionRecognition/MP_Data_REL2'
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
                parent_file = np.load(f'{parent_file}')

                child_file_path = os.path.join(child_path, sign, str(sequence), str(frame_num))
                # the child folder has array spliced from specified index

                rel = []
                for i in range(66, 126, 3):
                    x = parent_file[63] - parent_file[i]
                    y = parent_file[64] - parent_file[i + 1]
                    z = parent_file[65] - parent_file[i + 2]
                    rel.append([x, y, z])

                array = np.array(rel).flatten()

                # the child folder has array spliced from specified index
                child_file = parent_file[:SLICE_INDEX]
                concat_file=np.concatenate([child_file,array])

                np.save(child_file_path, concat_file)
print("Complete! All the best in Training.")



#TO CHECK THE SPLICED DATASET
'''
for root, dirs, files in os.walk(parent):
    for filename in files:
        filename=os.path.join(root, filename)
        file=np.load(f'{filename}')
        print((file[:6]))'''