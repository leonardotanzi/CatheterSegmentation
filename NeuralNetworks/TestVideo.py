from keras_segmentation.models.unet import vgg_unet
from keras_segmentation.predict import predict, predict_multiple, model_from_checkpoint_path, evaluate
import cv2
from keras.models import load_model
import numpy as np
from tqdm import tqdm
import os
from typing import List, Tuple

class_colors: List[Tuple[int, int, int]] = [(0, 0, 0), (255, 0, 0), (0, 255, 0)]

scale_percent = 60  # percent of original size


def convertNumpyArrayToMat(img):
    seg_img = np.zeros((img.shape[0], img.shape[1], 3))
    colors = class_colors

    for c in range(3):  # with 3 classes

        seg_img[:, :, 0] += ((img[:, :] == c) * (colors[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((img[:, :] == c) * (colors[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((img[:, :] == c) * (colors[c][2])).astype('uint8')

    seg_img = cv2.resize(seg_img, (640, 408))

    return seg_img


videoFilePath = "..\\..\\RealTime Video\\CV_2_cropped.mp4"

model = model_from_checkpoint_path("..\\..\\Checkpoints\\AllTools\\mobilenet_alltools")

cap = cv2.VideoCapture(videoFilePath)
i = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    dim = (960, 540)

    frame = cv2.resize(frame, dim, interpolation=cv2.INTER_LINEAR)

    if (i % 5) == 0:
        out = model.predict_segmentation(inp=frame)
        img = convertNumpyArrayToMat(out)


    cv2.imshow('ResNet', img)
    cv2.imshow('Original', frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    i += 1
cap.release()
cv2.destroyAllWindows()

'''
image = cv2.imread("C:\\Users\\d053175\\Desktop\\New_dataset\\Train\\LabelsNew\\40846.png")
b = image.copy()
# set green and red channels to 0
b[:, :, 1] = 0
b[:, :, 2] = 0
g = image.copy()
# set blue and red channels to 0
g[:, :, 0] = 0
g[:, :, 2] = 0
r = image.copy()
# set blue and green channels to 0
r[:, :, 0] = 0
r[:, :, 1] = 0
# RGB - Blue
cv2.imshow('B-RGB', b)
# RGB - Green
cv2.imshow('G-RGB', g)
# RGB - Red
cv2.imshow('R-RGB', r)
cv2.waitKey(0)

'''
'''
model = model_from_checkpoint_path("..\\..\\Checkpoints\\AllTools\\mobilenet_alltools")

# print(model.evaluate_segmentation(inp_images_dir="C:\\Users\\Leonardo\\Desktop\\data4\\JPEGImages\\", annotations_dir="C:\\Users\\Leonardo\\Desktop\\data4\\lab\\"))
# out = model.predict_segmentation(inp="C:\\Users\\Leonardo\\Desktop\\data4\\JPEGImages\\kang2950.jpg", out_fname="..\\a.png")

cap = cv2.VideoCapture("..\\..\\RealTime Video\\CV_2_cropped.mp4")
i = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    out = model.predict_segmentation(inp=frame, out)
    cv2.imshow("Output", out)
    i += 1

cap.release()
cv2.destroyAllWindows()


predict(checkpoints_path="checkpoints\\vgg_unet_1",
        inp="C:\\Users\\d053175\\Desktop\\Prostate\\Dataset\\Test\\39151.png",
        out_fname="C:\\Users\\d053175\\Desktop\\output.png"
        )


predict_multiple(checkpoints_path="checkpoints\\vgg_unet_1",
		inp_dir="C:\\Users\\d053175\\Desktop\\Prostate\\Test\\",
		out_dir="C:\\Users\\d053175\\Desktop\\outputs\\"
)
'''
