# questo file per usare la rete

from typing import List, Tuple
import cv2
import numpy as np
from keras_segmentation.predict import predict, model_from_checkpoint_path
from scipy.ndimage import gaussian_filter

class_colors: List[Tuple[int, int, int]] = [(0, 0, 0), (255, 0, 0), (0, 255, 0)]

scale_percent = 60  # percent of original size


def convertNumpyArrayToMat(img):
    seg_img = np.zeros((img.shape[0], img.shape[1], 3))
    colors = class_colors

    for c in range(3):  # with 3 classes

        seg_img[:, :, 0] += ((img[:, :] == c) * (colors[c][0])).astype('float')
        seg_img[:, :, 1] += ((img[:, :] == c) * (colors[c][1])).astype('float')
        seg_img[:, :, 2] += ((img[:, :] == c) * (colors[c][2])).astype('float')

    seg_img = cv2.resize(seg_img, (960, 540))

    return seg_img


videoFilePath = "..\\RealTime Video\\CV_2_cropped.mp4"

model = model_from_checkpoint_path("..\\Checkpoints\\NewTool\\new_mobilenet_unet_tool")

cap = cv2.VideoCapture(videoFilePath)
i = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    dim = (960, 540)

    frame = cv2.resize(frame, dim, interpolation=cv2.INTER_LINEAR)
    angle = 0.0

    if (i % 5) == 0:
        out = model.predict_segmentation(inp=frame)
        img = convertNumpyArrayToMat(out)
        (channel_b, channel_g, channel_r) = cv2.split(img)  # ciascuna con un solo canale

        contours_b, hierarchy_b = cv2.findContours(channel_b.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_g, hierarchy_g = cv2.findContours(channel_g.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        output = frame.copy()

        if len(contours_b) != 0:
            # cv2.drawContours(output, contours_b, -1, (125,125,125), 1)

            # find the biggest countour (c) by the area
            c = max(contours_b, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)

            # draw the biggest contour (c) in blue
            # cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)

        if len(contours_g) != 0:

            # contours_gs = sorted(contours_g, key=cv2.contourArea, reverse=True)
            # selection of the biggest area
            max_c = max(contours_g, key=cv2.contourArea)
            # draw all the contours
            # cv2.drawContours(output, contours_g, -1, (0, 255, 0), 1)
            hull = cv2.convexHull(max_c)
            cv2.drawContours(output, [hull], 0, (0, 0, 150), 2)

            # epsilon = 0.03 * cv2.arcLength(max_c, True)
            # approx = cv2.approxPolyDP(max_c, epsilon, True)
            # cv2.drawContours(output, [approx], 0, (0, 0, 50), 2)

            if len(max_c) > 5:
                rot_rect = cv2.fitEllipse(max_c)
                angle = rot_rect[2]
                box = cv2.boxPoints(rot_rect)
                box = np.int0(box)
                #cv2.drawContours(output, [box], 0, (0, 0, 255), 2)
                if angle > 90:
                    angle = angle - 180
                print("{:.2f}".format(angle))

            # x1, y1, w1, h1 = cv2.boundingRect(contours_gs[0][1])
            # x2, y2, w2, h2 = cv2.boundingRect(contours_gs[0][2])
            # x3, y3, w3, h3 = cv2.boundingRect(contours_gs[0][3])
            # x4, y4, w4, h4 = cv2.boundingRect(contours_gs[0][4])

            # cv2.rectangle(output, (x3, y3), (x3 + w3, y3 + h3), (0, 155, 0), 2)
            # cv2.rectangle(output, (x4, y4), (x4 + w4, y4 + h4), (0, 255, 0), 2)
            # cv2.rectangle(output, (x2, y2), (x2 + w2, y2 + h2), (0, 105, 0), 2)
            # cv2.rectangle(output, (x1, y1), (x1 + w1, y1 + h1), (0, 55, 0), 2)

            # find the biggest countour (c) by the area
           # c = max(contours_g, key=cv2.contourArea)
           # x, y, w, h = cv2.boundingRect(c)
#
           # # draw the biggest contour (c) in green
           # cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # cv2.imshow('Original', frame)
    # cv2.imshow("CNN", img)
    cv2.imshow("output", output)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    i += 1
cap.release()
cv2.destroyAllWindows()
