from nn_construction import constructNet
from scipy import misc


TRAIN_SIZE = 35,126
TEST_SIZE = 53,576
IMAGE_SIZE = 1024
NB_CLASSES = 5

if __name__ == "__main__":
    import os

    train_path = "./data/train/"
    test_path = "./data/test/"

    # this needs to be flattened one more dimension...
    # should be (TRAIN_SIZE, IMAGE_SIZE*IMAGE_SIZE)
    X_train = np.zeros((TRAIN_SIZE, IMAGE_SIZE, IMAGE_SIZE), dtype='uint8')
    y_train = np.zeros((TRAIN_SIZE, NB_CLASSES), dtype='uint8')
    
    for i,file_name in enumerate(os.listdir(train_path)):
        if "jpeg" in file_name:
            #preprocess the image
            path = "{}/{}".format(train_path, file_name)

            #extract image number for
            img = misc.imread(path, flatten=True)
            #this might not work....
            X_train[i] = img

            w,h = img.shape

    '''
    if we are training with 1024x1024 images the
    X_train matrix needs to be (nb_images, 1024x1024)
    
    y_train are the labels which are (nb_images, nb_labels) = (nb_images, 5)
    
    X_test matrix is (nb_test_images, 1024x1024)
    '''
    
    model = constructNet(image_size, nb_labels)
    
    model.fit(X_train, y_train, batch_size=BATCH_SIZE, nb_epoch= EPOCHS)
    #won't be able to do this, but perhaps we can generate the labels for the
    #X_test?
    #score = model.evaluate(X_test, Y_test, batch_size=BATCH_SIZE)
    #print("Test score: ", score)
            
