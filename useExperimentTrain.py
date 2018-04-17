from keras.models import Sequential
from keras.layers.core import Activation,Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import cv2, numpy as np
from keras.preprocessing.image import img_to_array, load_img

imHeight,imWidth = 224,224
input_shape = (imWidth, imHeight, 3)

def trainedPlateModel(weights_path=None):
    model = Sequential()

    model.add(ZeroPadding2D((1,1),input_shape=input_shape))
    model.add(Convolution2D(32, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(32, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))#OutputLayer

    if weights_path:
        model.load_weights(weights_path)

    return model
if __name__ == "__main__":
    im1 = cv2.resize(cv2.imread('validatePlate15.jpg'), (imWidth,imHeight)).astype(np.float)
   # im = np.array('TestPlate.jpg')
   # img = load_img('TestPlate.jpg')  # this is a PIL image
   # im = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
   # im[:,:,0] -= 103.939
   # im[:,:,1] -= 116.779
   # im[:,:,2] -= 123.68
    #im1 = im1.transpose((1,0,2))
    im1 = np.expand_dims(im1, axis=0)
    im2 = cv2.resize(cv2.imread('TestPlate.jpg'),(imWidth,imHeight)).astype(np.float)
    #im2 = im2.transpose((1,0,2))
    im2 = np.expand_dims(im2, axis=0)
    """img = load_img('validatePlate15.jpg',False,target_size=(imWidth,imHeight))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    img = load_img('plate0.jpg',False,target_size=(imWidth,imHeight))
    y = img_to_array(img)
    y = np.expand_dims(y, axis=0)"""
    model = trainedPlateModel('experiment_second_try.h5')
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
    #preds = model.predict_classes(x)
    prob1 = model.predict(im1, verbose=0).astype(np.float)
    prob2 = model.predict(im2, verbose=0).astype(np.float)
    #print(preds)
    print(prob1)
    print(prob2)
    if prob1>prob2 :
        print("Prob1")
    elif prob2>prob1 :
        print("Prob2")
    else :
        print("Failed")
    """

    # Test pretrained model
    model = trainedPlateModel('experiment_first_try.h5')
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    out = model.predict(im)
    print(np.argmax(out))
    """