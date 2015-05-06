from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD

STACK_SIZE = 3

def constructNet(self,
                 image_size,
                 nb_classes,
                 nb_layers=2,
                 layer_increase=2):
    model = Sequential()
    nb_filters = int(image_size / 32)

    for i in xrange(nb_layers -1):
        # filter size of 12x12 is used since we have relatively large
        # images
        model.add(Convolution2D(nb_filters, STACK_SIZE, 12, 12))
        model.add(Activation('relu'))
        model.add(Convolution2D(nb_filters, nb_filter, 12, 12))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(poolsize=(2,2)))
        model.add(Dropout(0.25))

        nb_filters *= layer_increase
        STACK_SIZE *= layer_increase

    model.add(Flatten())
    #I dont think 64x8x8 is the right value here.....
    model.add(Dense(64*8*8, 512))
    model.add(Dropout(0.5))
    
    model.add(Dense(512, nb_classes))
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.1, decay = 1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)

    return model
