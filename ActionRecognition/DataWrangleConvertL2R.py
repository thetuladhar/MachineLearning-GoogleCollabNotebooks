import os
import numpy as np

#removing DS store object, annoying as heck on a MAC OS
#cd to the file path in terminal then type
#find . -name '.DS_Store' -type f -delete

#The below three values are customizable
parent_path='/Users/Tuladhar/PycharmProjects/MediaPipe/ActionRecognition/MP_AlphaP'
child_path='/Users/Tuladhar/PycharmProjects/MediaPipe/ActionRecognition/MP_AlphaMesh'
SLICE_INDEX = 225

signsList= os.listdir(parent_path)
signsList.sort()
#print(signsList)

for sign in signsList:
    #sequenceLength = len(os.listdir(os.path.join(parent_path,sign)))
    start=601
    stop=800
    for sequence in range(start,stop+1,1):
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

                #INITIATING EMPTY LISTS
                rel_L = []
                LEFT_BASE_X = 0
                LEFT_BASE_Y = 1
                LEFT_BASE_Z = 2

                #was for i in range(66, 126, 3):
                for j in range(0, 63, 3):
                    MTlist = []
                    for i in range(0, 21, 1):
                        Main_x = j + LEFT_BASE_X
                        Main_y = j + LEFT_BASE_Y
                        Main_z = j + LEFT_BASE_Z

                        if Main_x != ((i * 3) + LEFT_BASE_X):
                            x1 = keypoints[Main_x] - keypoints[(i * 3) + LEFT_BASE_X]
                            y1 = keypoints[Main_y] - keypoints[(i * 3) + LEFT_BASE_Y]
                            z1 = keypoints[Main_z] - keypoints[(i * 3) + LEFT_BASE_Z]

                            MTlist.append([x1, y1, z1])
                    rel_L.append(MTlist)

                arrayL= np.array(rel_L ).flatten()

                # the child folder has array spliced from specified index
                # not using in the current code
                #child_file = parent_file[:SLICE_INDEX]

                concat_file=np.concatenate([arrayL])

                np.save(child_file_path, concat_file)
print("Complete! All the best in Training.")