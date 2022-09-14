import os
import matplotlib.pyplot as plt
from calories import calories
from cnn_model import get_model
import cv2
import time
from api import *
import tensorflow as tf
from PIL import ImageOps
from tensorflow.keras.models import model_from_json
size = 128
nb = 8
model_save_at = os.path.join("model", 'model_')

a=int(input("Voulez vous utiliser votre Webcome ( choix:1) ou télécharger une image(choix:2) "))
if(a==1):
    print("tapper _S_ pour sortir")
    food_list = list(np.load('labels.npy'))
    with open('model/model.json') as file:
        model_json = file.read()
    model = model_from_json(model_json)
    model.load_weights('model/weights_model.h5')

    prev_frame_time = 0
    new_frame_time = 0
    vid = cv2.VideoCapture(0)
    while (True):
        ret, frame = vid.read()
        camera_img_size = (frame.shape[1], frame.shape[0])
        resized_frame = cv2.resize(frame, (size, size))
        x = np.expand_dims(resized_frame, axis=0)
        y = model.predict(x)
        mask = np.argmax(y[0], axis=-1)
        mask = np.expand_dims(mask, axis=-1)
        img = ImageOps.autocontrast(tf.keras.preprocessing.image.array_to_img(mask))
        img = tf.keras.preprocessing.image.img_to_array(img.resize(camera_img_size))
        maskImg = np.zeros(frame.shape, frame.dtype)
        maskImg[:, :, 0] = img[:, :, 0]
        maskImg[:, :, 1] = img[:, :, 0]
        maskImg[:, :, 2] = img[:, :, 0]
        ids = np.unique(mask)
        [print(food_list[int(id)], get_info_from_db(food_list[int(id)])) if id != 0 and id < 6 else print("") for id in
         ids]
        overlap = cv2.addWeighted(frame, 1, maskImg, 0.6, 0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps = int(fps)
        fps = str(fps)
        cv2.putText(overlap, fps, (7, 50), font, 1, (100, 255, 0), 2)
        cv2.imshow('frame', overlap)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break
    vid.release()
    cv2.destroyAllWindows()
elif(a==2):
    model = get_model(size, nb, 1e-3)
    model.load(model_save_at)
    labels = list(np.load('labels.npy'))
    test_data = 'test_image.JPG'
    img = cv2.imread(test_data)
    img1 = cv2.resize(img, (size, size))
    model_out = model.predict([img1])
    result = np.argmax(model_out)
    name = labels[result]
    cal = round(calories(result + 1, img), 2)
    plt.imshow(img)
    b = int(input("Voulez vous avoir une estimation des calories des elt alimentaires ( choix:1) ou avoir une analyse nutritionnelle complète de votre plat (choix:2) "))
    if(b==1):
        print(name, cal, "cal")
        plt.title('{}(calorie:{})'.format("tomate",25))

    else:
        print(name, get_info_from_db(name))
        plt.title('{}({}kcal)'.format(name, get_info_from_db(name)))

    plt.axis('off')
    plt.show()
