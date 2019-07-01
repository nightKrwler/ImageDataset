
# two classes (class1, class2)
# only replace with a directory of yours
# finally .npy files saved in your directory with names train.npy, #test.npy, train_labels.npy, test_labels.npy
import cv2
import glob
import numpy as np
#Train data
train = []
train_labels = []

NO_OF_CLASSES = 30
IMG_SIZE = 128


filepaths = ["../../datasets/texture30/*.jpg"]
files = glob.glob("../../datasets/texture30/*.jpg")

for i in range(len(files)):
    j = int(files[i].split('/')[4][1:4])
    print(str(files[i].split('/')[4])," , ",str(j))
    train_label = np.zeros(NO_OF_CLASSES,'int8')
    train_label[j-1] = 1
        #prvchgint(myFile)
    image = cv2.imread(files[i], 0)
    train.append(cv2.resize(image, (IMG_SIZE, IMG_SIZE)))
    train_labels.append(train_label)
    #print(train_label)


train_labels = np.array(train_labels,dtype='int8') #as mnist
train = np.array(train) 
print(train.shape)
# convert (number of images x height x width x number of channels) to (number of images x (height * width *3)) 
# for example (120 * 40 * 40 * 3)-> (120 * 4800)

train = np.reshape(train,[train.shape[0],train.shape[1]*train.shape[2]])
print(train.shape)


# # save numpy array as .npy formats
np.save('tex45Img',train)
np.save('tex45Imglabels',train_labels)

# #Test data
# test = []
# test_labels = []
# files = glob.glob ("/data/test/class1/*.JPG")
# for myFile in files:
#     image = cv2.imread (myFile)
#     test.append (image)
#     test_labels.append([1., 0.]) # class1
# files = glob.glob ("/data/test/class2/*.JPG")
# for myFile in files:
#     image = cv2.imread (myFile)
#     test.append (image)
#     test_labels.append([0., 1.]) # class2

# test = np.array(test,dtype='float32') #as mnist example
# test_labels = np.array(test_labels,dtype='float64') #as mnist
# test = np.reshape(test,[test.shape[0],test.shape[1]*test.shape[2]*test.shape[3]])

# # save numpy array as .npy formats
# np.save('test',test) # saves test.npy
# np.save('test_labels',test_labels)