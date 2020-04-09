from typing import List, Tuple
import matplotlib.pyplot as plt
import cv2
import numpy as np
from keras_segmentation.predict import predict, model_from_checkpoint_path
from PIL import Image
import scipy
from scipy.ndimage import gaussian_filter
import math

class_colors: List[Tuple[int, int, int]] = [(0, 0, 0), (0, 0, 0), (0, 255, 0)]  #second is (255, 0, 0) if I want tools

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


def calculate_distance(a1, b1, a2, b2):
    dist = math.sqrt((a2 - a1)**2 + (b2 - b1)**2)
    return dist


def build_line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C


def intersection(L1, L2):
    D = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return int(x), int(y)
    else:
        return 0, 0

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

    if (i % 10) == 0:
        # out_resnet = model_resnet.predict_segmentation(inp=frame)
        # img_resnet = convertNumpyArrayToMat(out_resnet)
        # out_vgg = model_vgg.predict_segmentation(inp=frame)
        # img_vgg = convertNumpyArrayToMat(out_vgg)
        # out_unet = model_unet.predict_segmentation(inp=frame)
        # img_unet = convertNumpyArrayToMat(out_unet)

        out_mobilenet = model_mobilenet.predict_segmentation(inp=frame)
        img_original = convertNumpyArrayToMat(out_mobilenet)

        img_original = img_original.astype(np.uint8)
        kernel = np.ones((11, 11), np.uint8)
        img_mobilenet = cv2.dilate(img_original, kernel, iterations=3)
        img_mobilenet = cv2. erode(img_mobilenet, kernel, iterations=3)
        # img_mobilenet = cv2.GaussianBlur(img_mobilenet, (7, 7), 0)
        #img_mobilenet = cv2.erode(img_mobilenet, kernel, iterations=2)
        img_mobilenet = cv2.cvtColor(img_mobilenet, cv2.COLOR_BGR2GRAY)
        img_mobilenet = cv2.Canny(img_mobilenet, 5, 15, apertureSize=3)
        kernel_dilate = np.ones((7, 7), np.uint8)
        img_mobilenet = cv2.dilate(img_mobilenet, kernel_dilate, iterations=2)
        minLineLength = 50
        maxLineGap = 0
        lines = cv2.HoughLinesP(img_mobilenet, 1, np.pi / 180, 100, minLineLength, maxLineGap)

        if lines is not None:
            n_line = 0
            for line in lines:
                for x1, y1, x2, y2 in line:
                    if n_line == 0:
                        n_line += 1
                        l1 = build_line([x1, y1], [x2, y2])
                        cv2.line(img_mobilenet, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        x1_line1 = x1
                        x2_line1 = x2
                        y1_line1 = y1
                        y2_line1 = y2
                    if n_line > 0:
                        l_tmp = build_line([x1, y1], [x2, y2])
                        dist1 = abs(x2 - x2_line1)
                        if dist1 > 20:
                            l2 = l_tmp
                            n_line += 1
                            cv2.line(img_mobilenet, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            break
                if n_line == 2:
                    inter_x, inter_y = intersection(l1, l2)
                    if inter_y < 0:
                        inter_y = 0
                    if inter_y > 408:
                        inter_y = 408
                    if inter_x < 0:
                        inter_x = 0
                    if inter_x > 640:
                        inter_x = 640
                    print(inter_x, inter_y)
                    cv2.circle(img_mobilenet, (inter_x, inter_y), 10, (255, 0, 0), thickness=10)
                    break
            # merge_fig(img_mobilenet, frame, i)


    # cv2.imshow('ResNet', img_resnet)
    # cv2.imshow('VGG', img_vgg)
    cv2.imshow('MobileNet', frame)
    cv2.imshow('Gaussian', img_mobilenet)
    # cv2.imshow('Unet', img_unet)
    # cv2.imshow('Original', frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    i += 1
cap.release()
cv2.destroyAllWindows()