from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import numpy as np

DATA_PATH = os.path.join('MP_Data')
actions = np.array(['Hello'])
# Thirty videos worth of data
no_sequences = 30
# Videos are going to be 30 frames in length
sequence_length = 30

label_map={label:num for num,label in enumerate(actions)}

#print(label_map)

sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

print(np.array(sequences).shape)