import os
import numpy as np
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.utils import to_categorical

def preprocess():
    #Replace these hand signs with whichever others you want from the dataset folder
    actions = np.array(['Thank You','Hello','I Love You'])

    DATA_PATH = os.path.join("Dataset\BasicSigns") 

    #Dataset consists of 30 videos of 30 frames for each sign stored in a numpy array format
    no_sequences = 30
    sequence_length = 30
    start_folder = 30

    #label encoding
    label_map = {label:num for num, label in enumerate(actions)}
    label_map

    #import data from the dataset folder
    sequences, labels = [], []
    for action in actions:
        for sequence in np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int):
            window = []
            for frame_num in range(sequence_length):
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])

    sequences = np.array(sequences)

    y = to_categorical(labels).astype(int)
    y = np.array(y)

    #generate positive samples for each action  
    thanks_im = []
    hello_im = []
    iloveyou_im = []
    for i in range(len(labels)):
        if(labels[i]==0):
            thanks_im.append(sequences[i])
        elif(labels[i]==1):
            hello_im.append(sequences[i])
        else:
            iloveyou_im.append(sequences[i])

    import itertools  

    # Test images
    test_hello_im = hello_im[27:]
    test_thanks_im = thanks_im[27:]
    test_iloveyou_im = iloveyou_im[27:]

    # Read only 27 videos from each class for training
    hello_im = hello_im[:27]
    thanks_im = thanks_im[:27]
    iloveyou_im = iloveyou_im[:27]

    positive_hello = list(itertools.combinations(hello_im, 2))
    positive_thanks = list(itertools.combinations(thanks_im, 2))
    positive_iloveyou = list(itertools.combinations(iloveyou_im, 2))

    #generating negative samples for each action
    negative1 = itertools.product(hello_im,thanks_im)
    negative1 = list(negative1)

    negative2 = itertools.product(thanks_im,iloveyou_im)
    negative2 = list(negative2)

    negative3 = itertools.product(hello_im,iloveyou_im)
    negative3 = list(negative3)

    # Create pairs of images and set target label for them.
    signs_X1 = []
    signs_X2 = []
    signs_y = []
    positive_samples = positive_hello + positive_thanks + positive_iloveyou
    negative_samples = negative1 + negative2 + negative3


    for fname in positive_samples :
        signs_X1.append(fname[0])
        signs_X2.append(fname[1])
        signs_y.append(1)

    for fname in negative_samples :
        signs_X1.append(fname[0])
        signs_X2.append(fname[1])
        signs_y.append(0)

    signs_y = np.array(signs_y)
    signs_X1 = np.array(signs_X1)
    signs_X2 = np.array(signs_X2)


    signs_X1 = signs_X1.reshape((len(negative_samples) + len(positive_samples), 30, 1662))
    signs_X2 = signs_X2.reshape((len(negative_samples) + len(positive_samples), 30, 1662))

    return signs_X1, signs_X2, signs_y

