# questo file per usare la rete

from typing import List, Tuple
import matplotlib.pyplot as plt
import cv2
import numpy as np
from keras_segmentation.predict import predict, model_from_checkpoint_path
from PIL import Image
import scipy
from scipy.ndimage import gaussian_filter

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


def merge_fig(img1, img2, name):

    img1 = img1.astype(int)
    img2 = cv2.resize(img2, (640, 408))
    imgs = [img1, img2]
    # imgs = [Image.open(i) for i in list_im]
    # pick the image which is the smallest, and resize the others to match it (can be arbitrary image shape here)
    # min_shape = sorted([(np.sum(i.size), i.size) for i in imgs])[0][1]
    imgs_comb = np.hstack(imgs)
    imgs_comb = imgs_comb.astype(np.uint8)
    # plt.imsave(("..\\Output\\{}.png".format(name)), imgs_comb)
    return imgs_comb
    # save that beautiful picture
    # imgs_comb = Image.fromarray(imgs_comb)
    # imgs_comb.save("..\\Output\\{}.png".format(name))


videoFilePath = "..\\RealTime Video\\CV_2_cropped.mp4"

# model_vgg = model_from_checkpoint_path("..\\Checkpoints\\NewTool\\new_vgg_unet_tool")
# model_resnet = model_from_checkpoint_path("..\\Checkpoints\\NewTool\\new_resnet_unet_tool")
model_mobilenet = model_from_checkpoint_path("..\\Checkpoints\\NewTool\\new_mobilenet_unet_tool")
# model_unet = model_from_checkpoint_path("..\\Checkpoints\\NewTool\\new_unet_tool")

cap = cv2.VideoCapture(videoFilePath)
i = 0
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    dim = (960, 540)

    frame = cv2.resize(frame, dim, interpolation=cv2.INTER_LINEAR)

    if (i % 5) == 0:
        # out_resnet = model_resnet.predict_segmentation(inp=frame)
        # img_resnet = convertNumpyArrayToMat(out_resnet)

        # out_vgg = model_vgg.predict_segmentation(inp=frame)
        # img_vgg = convertNumpyArrayToMat(out_vgg)

        out_mobilenet = model_mobilenet.predict_segmentation(inp=frame)
        img_mobilenet = convertNumpyArrayToMat(out_mobilenet)

        img_mobilenet = img_mobilenet.astype(np.uint8)
        imgGaussian_mobilenet = cv2.GaussianBlur(img_mobilenet, (7,7), 0)

        kernel = np.ones((5, 5), np.uint8)
        imgGaussian_mobilenet = cv2.erode(imgGaussian_mobilenet, kernel, iterations=2)
        gray = cv2.cvtColor(imgGaussian_mobilenet, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 5, 15, apertureSize=3)
        minLineLength = 100
        maxLineGap = 50
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength, maxLineGap)

        if lines is not None:
            for x1, y1, x2, y2 in lines[0]:
                cv2.line(imgGaussian_mobilenet, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # merge_fig(img_mobilenet, frame, i)

        # out_unet = model_unet.predict_segmentation(inp=frame)
        # img_unet = convertNumpyArrayToMat(out_unet)

    # cv2.imshow('ResNet', img_resnet)
    # cv2.imshow('VGG', img_vgg)
    cv2.imshow('MobileNet', img_mobilenet)
    cv2.imshow('Gaussian', imgGaussian_mobilenet)
    # cv2.imshow('Unet', img_unet)

    # cv2.imshow('Original', frame)


    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    i += 1
cap.release()
cv2.destroyAllWindows()