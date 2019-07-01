

 # two classes (class1, class2)
# only replace with a directory of yours
# finally .npy files saved in your directory with names train.npy, #test.npy, train_labels.npy, test_labels.npy
import cv2
import glob
import numpy as np
#Train data
train = []
train_labels = []
print("hellp")

filepaths = ["../../datasets/digitsDataset/digits/Dataset/0/*.JPG","../../datasets/digitsDataset/digits/Dataset/1/*.JPG",
"../../datasets/digitsDataset/digits/Dataset/2/*.JPG", "../../datasets/digitsDataset/digits/Dataset/3/*.JPG",
"../../datasets/digitsDataset/digits/Dataset/4/*.JPG", "../../datasets/digitsDataset/digits/Dataset/5/*.JPG",
"../../datasets/digitsDataset/digits/Dataset/6/*.JPG", "../../datasets/digitsDataset/digits/Dataset/7/*.JPG",
"../../datasets/digitsDataset/digits/Dataset/8/*.JPG", "../../datasets/digitsDataset/digits/Dataset/9/*.JPG"]

for i in range(len(filepaths)):
    #print(filepaths[i])
    files = glob.glob(filepaths[i])
    
    train_label = np.zeros(10)
    train_label[i] = 1
    for myFile in files:
        #print(myFile)
        image = cv2.imread(myFile)
        IMG_SIZE = 100
        train.append(cv2.resize(image, (IMG_SIZE, IMG_SIZE)))
        train_labels.append(train_label)
    #print(train_label)


train_labels = np.array(train_labels,dtype='float64') #as mnist
train = np.array(train) #as mnist
print(train.shape)
# convert (number of images x height x width x number of channels) to (number of images x (height * width *3)) 
# for example (120 * 40 * 40 * 3)-> (120 * 4800)

train = np.reshape(train,[train.shape[0],train.shape[1]*train.shape[2]*train.shape[3]])
print(train.shape)

# # save numpy array as .npy formats
np.save('train',train)
np.save('train_labels',train_labels)

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