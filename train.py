import glob
import os
from random import shuffle
import cv2
import numpy as np
from cnn_model import get_model
size_img = 128
nb=8
nb_of_images=1500
def train_data(path):
    training_data = []
    folders=os.listdir(path)[0:nb]
    for i in range(len(folders)):
        label = [0 for i in range(nb)]
        label[i] = 1
        print(folders[i])
        k=0
        for j in glob.glob(path+"\\"+folders[i]+"\\*.jpg"):            
            if(k==nb_of_images):
                break
            k=k+1
            img = cv2.imread(j)
            img = cv2.resize(img, (size_img,size_img))
            training_data.append([np.array(img),np.array(label)])
    np.save('./model/training_{}.npz'.format(nb),training_data)
    shuffle(training_data)
    return training_data,folders
path = r'./Data'
training_data,labels=train_data(path)
size=int(len(training_data)*0.3)
train = training_data[:-size]
test=training_data[-size:]

X = np.array([i[0] for i in train]).reshape(-1,size_img,size_img,3)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1,size_img,size_img,3)
test_y = [i[1] for i in test]

model=get_model(size_img,nb,1e-3)

model.fit({'input': X}, {'targets': Y}, n_epoch=10, validation_set=({'input': test_x}, {'targets': test_y}), 
    snapshot_step=500, show_metric=True, run_id='model_')

model_save_at=os.path.join("model",'model_')
model.save(model_save_at)
print("Model Save At",model_save_at)