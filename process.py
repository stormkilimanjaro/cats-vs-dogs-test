import cv2
import sys
import re
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

#%matplotlib inline

TRAIN_DIR = 'train'
#TRAIN_DIR = 'test'
TEST_DIR = 'test'
IMG_SIZE = 50
LR = 1e-3
MODEL_NAME = 'dogs-vs-cats-convnet'

"""
Pre-processing
"""

def create_label(image_name):
    """
    Create an one-hot encoded vector from image name
    """
    word_label = image_name.split('.')[-2]
    word_label = re.findall(r"[A-Z][a-z]+", word_label)
    #print (word_label)
    if word_label[0] == 'Cat':
        #print ("Word label found: %s" % word_label)
        return np.array([1,0])
    elif word_label[0] == 'Dog':
        #print("Word label found: %s" % word_label)
        return np.array([0,1])
    else:
        print("Word label not found")


"""
Every image will be resized to 50 x 50 pixels and read as grayscale
"""
def create_train_data():
    training_data = []
    exceptions = 0
    exception_list = []

    for img in tqdm(os.listdir(TRAIN_DIR)):
       
        if img != ".DS_Store":
            path = os.path.join(TRAIN_DIR, img)
            try:
                img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
                training_data.append([np.array(img_data), create_label(img)])
            except KeyError:
                print ("Key Error caused by %s" %(img))
            except:
                print("Exception: {}".format(img))
                exceptions = exceptions+1
                exception_list.append(img)


    '''
    Debugging statements:
     print ("Number of exceptions: %d" % (exceptions))
    for i in exception_list:
        print ("%s" % i)
    '''


    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data


def create_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        if img != ".DS_Store":
            path = os.path.join(TEST_DIR, img)
            img_num = img.split('.')[0]
    
            try:
                img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
                testing_data.append([np.array(img_data), img_num])
            except:
                print (img)


    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data

# If dataset is not created:
#train_data = create_train_data()
#test_data = create_test_data()
# If you have already created the dataset:
train_data = np.load('train_data.npy')
test_data = np.load('test_data.npy')

train = train_data[:-500]
test = train_data[-500:]

X_train = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1) #image data
y_train = [i[1] for i in train] #labels

'''
Debugging statement to check if the labels have been created successfully:

for i in train:
    print (i[1])
'''


X_test = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_test = [i[1] for i in test]


tf.reset_default_graph()

#Input is a 50x50 image with 1 color channel (grey-scale)
convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

#Step 1: Creating the convolutional network layers
# 1.1: Convolution layer with 32 filters, each 5x5
convnet = conv_2d(convnet, 32, 5, activation='relu')

#1.2: Max pooling layer
convnet = max_pool_2d(convnet, 5)

#1.3: Convolution layer with 64 filters, each 5x5
convnet = conv_2d(convnet, 64, 5, activation='relu')

#1.4: Max pooling layer
convnet = max_pool_2d(convnet, 5) #Changed from 5 to 2


'''
# Additional layers to make the model more accurate.

#1.5: Convolution layer with 128 filters, each 5x5
convnet = conv_2d(convnet, 128, 5, activation='relu')

#1.6: Max pooling layer
convnet = max_pool_2d(convnet, 5)

#1.7: Convolution layer with 128 filters, each 5x5
convnet = conv_2d(convnet, 64, 5, activation='relu')

#1.8: Max pooling layer
convnet = max_pool_2d(convnet, 5)

#1.9: Convolution layer with 128 filters, each 5x5
convnet = conv_2d(convnet, 32, 5, activation='relu')

#1.10: Max pooling layer
convnet = max_pool_2d(convnet, 5)
'''

#Step 2: Fully-connected 1024 node layer
convnet = fully_connected(convnet, 1024, activation='relu') #changed number of neurons from 512 to 1024

#Step 3: Dropout later to combat overfitting
convnet = dropout(convnet, 0.8) #Changed from 0.8 to 0.5

#Step 4: Fully connected layer with 2 outputs
convnet = fully_connected(convnet, 2, activation='softmax') 
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log', tensorboard_verbose=0)

model.fit({'input': X_train}, {'targets': y_train}, n_epoch=10, validation_set=({'input': X_test}, {'targets': y_test}), snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

'''
Testing
'''
fig = plt.figure(figsize=(16, 12))

for num, data in enumerate(test_data[:16]):

    img_num = data[1]
    img_data = data[0]

    y = fig.add_subplot(4, 4, num + 1)
    orig = img_data
    data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
    model_out = model.predict([data])[0]

    if np.argmax(model_out) == 1:
        str_label = 'Dog'
    else:
        str_label = 'Cat'

    y.imshow(orig, cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()

sys.exit()
