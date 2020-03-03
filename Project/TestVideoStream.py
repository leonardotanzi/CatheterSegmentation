# questo file per usare la rete

from typing import List, Tuple
import cv2
import numpy as np
from keras_segmentation.predict import predict, model_from_checkpoint_path

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


videoFilePath = "..\\RealTime Video\\CV_2_cropped.mp4"

model_vgg = model_from_checkpoint_path("..\\Checkpoints\\Tool\\vgg_unet_tool")
model = model_from_checkpoint_path("..\\Checkpoints\\Tool\\resnet_unet_tool")

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

        out_vgg = model_vgg.predict_segmentation(inp=frame)
        img_vgg = convertNumpyArrayToMat(out_vgg)

    cv2.imshow('ResNet', img)
    cv2.imshow('VGG', img_vgg)
    cv2.imshow('Original', frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    i += 1
cap.release()
cv2.destroyAllWindows()