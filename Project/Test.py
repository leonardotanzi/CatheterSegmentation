from keras_segmentation.models.unet import vgg_unet
from keras_segmentation.predict import predict, predict_multiple, model_from_checkpoint_path
import cv2
from keras.models import load_model
import numpy as np
from tqdm import tqdm
import os


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

model = model_from_checkpoint_path("..\\Checkpoints\\vgg_unet_1")

cap = cv2.VideoCapture("CV_2_cropped.mp4")
i = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    out = model.predict_segmentation(inp=frame, out_fname="..\\Output\\{}.png".format(i))
    i += 1

cap.release()
cv2.destroyAllWindows()

'''
predict(checkpoints_path="checkpoints\\vgg_unet_1",
        inp="C:\\Users\\d053175\\Desktop\\Prostate\\Dataset\\Test\\39151.png",
        out_fname="C:\\Users\\d053175\\Desktop\\output.png"
        )
'''
'''
predict_multiple(checkpoints_path="checkpoints\\vgg_unet_1",
		inp_dir="C:\\Users\\d053175\\Desktop\\Prostate\\Test\\",
		out_dir="C:\\Users\\d053175\\Desktop\\outputs\\"
)
'''
