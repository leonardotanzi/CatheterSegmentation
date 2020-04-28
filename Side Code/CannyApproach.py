from typing import List, Tuple
import matplotlib.pyplot as plt
import cv2
import numpy as np
from keras_segmentation.predict import predict, model_from_checkpoint_path
from PIL import Image
import scipy
from scipy.ndimage import gaussian_filter
import math
import argparse

class_colors: List[Tuple[int, int, int]] = [(0, 0, 0), (0, 0, 0), (0, 255, 0)]  #second is (255, 0, 0) if I want tools

scale_percent = 60  # percent of original size


def convert_np_to_mat(img):
    seg_img = np.zeros((img.shape[0], img.shape[1], 3))
    colors = class_colors

    for c in range(3):  # with 3 classes

        seg_img[:, :, 0] += ((img[:, :] == c) * (colors[c][0])).astype("uint8")
        seg_img[:, :, 1] += ((img[:, :] == c) * (colors[c][1])).astype("uint8")
        seg_img[:, :, 2] += ((img[:, :] == c) * (colors[c][2])).astype("uint8")

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
    a = (p1[1] - p2[1])
    b = (p2[0] - p1[0])
    c = (p1[0]*p2[1] - p2[0]*p1[1])
    return a, b, -c


def intersection(line1, line2):
    delta = line1[0] * line2[1] - line1[1] * line2[0]
    delta_x = line1[2] * line2[1] - line1[1] * line2[2]
    delta_y = line1[0] * line2[2] - line1[2] * line2[0]
    if delta != 0:
        x = delta_x / delta
        y = delta_y / delta
        return int(x), int(y)
    else:
        return 0, 0


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True, help="Select the model to use "
                                                          "(0 for MobileNet, 1 for VGG, 2 for ResNet, 3 for Unet")
    args = vars(ap.parse_args())
    run_model = int(args["model"])
    models = ["MobileNet", "VGG", "ResNet", "U-Net"]
    print("Running the {} model.".format(models[run_model]))

    video_path = "..\\RealTime Video\\CV_2_cropped.mp4"

    if run_model == 0:
        model = model_from_checkpoint_path("..\\Checkpoints\\NewTool\\new_mobilenet_unet_tool")
    elif run_model == 1:
        model = model_from_checkpoint_path("..\\Checkpoints\\NewTool\\new_vgg_unet_tool")
    elif run_model == 2:
        model = model_from_checkpoint_path("..\\Checkpoints\\NewTool\\new_resnet_unet_tool")
    elif run_model == 3:
        model = model_from_checkpoint_path("..\\Checkpoints\\NewTool\\new_unet_tool")

    cap = cv2.VideoCapture(video_path)
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        dim = (960, 540)

        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_LINEAR)

        if (i % 10) == 0:

            out = model.predict_segmentation(inp=frame)

            img_original = convert_np_to_mat(out)
            img_original = img_original.astype(np.uint8)
            # Phase 1: first erode to delete outliers area, then dilate and erode to fill holes
            kernel = np.ones((11, 11), np.uint8)
            kernel2 = np.ones((9, 9), np.uint8)
            n_it = 3
            img_first_erosion = cv2.erode(img_original, kernel, iterations=1)
            img_dilated = cv2.dilate(img_first_erosion, kernel, iterations=n_it)
            img_eroded = cv2. erode(img_dilated, kernel, iterations=n_it)

            # img_mobilenet = cv2.GaussianBlur(img_mobilenet, (7, 7), 0)
            #img_mobilenet = cv2.erode(img_mobilenet, kernel, iterations=2)

            # Phase 2: convert to gray and apply canny for edge detection, then dilate to enlarge edges
            img_gray = cv2.cvtColor(img_eroded, cv2.COLOR_BGR2GRAY)
            img_edged = cv2.Canny(img_gray, 5, 15, apertureSize=3)
            kernel_dilate = np.ones((7, 7), np.uint8)
            img_final = cv2.dilate(img_edged, kernel_dilate, iterations=2)

            # Phase 3: find lines
            minLineLength = 50
            maxLineGap = 0
            lines = cv2.HoughLinesP(img_final, 1, np.pi / 180, 100, minLineLength, maxLineGap)

            frame = cv2.resize(frame, (640, 408), interpolation=cv2.INTER_LINEAR)

            # if the function find some lines
            if lines is not None:
                n_line = 0
                for line in lines:
                    for x1, y1, x2, y2 in line:
                        # obtain the first line and keep the coordinates
                        if n_line == 0:
                            n_line += 1
                            l1 = build_line([x1, y1], [x2, y2])
                            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            x1_line1 = x1
                            x2_line1 = x2
                            y1_line1 = y1
                            y2_line1 = y2
                        # if there are more than one line
                        if n_line > 0:
                            l_tmp = build_line([x1, y1], [x2, y2])
                            # if the two lines are closer than 20, discard the line and continue to the next one
                            dist1 = abs(x2 - x2_line1)
                            if dist1 > 20:
                                l2 = l_tmp
                                n_line += 1
                                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                break
                    # after finding two lines, compute intersection and save the values on a txt files
                    if n_line == 2:
                        inter_x, inter_y = intersection(l1, l2)
                        filename = "intersections.txt"
                        myfile = open(filename, "a+")
                        myfile.write("Frame {}: {} / {}\n".format(i, inter_x, inter_y))

                        # this is done in order to make the intersection circle stay inside the window
                        if inter_y < 0:
                            inter_y = 0
                        if inter_y > 408:
                            inter_y = 408
                        if inter_x < 0:
                            inter_x = 0
                        if inter_x > 640:
                            inter_x = 640
                        print(inter_x, inter_y)
                        cv2.circle(frame, (inter_x, inter_y), 10, (255, 0, 0), thickness=10)

                        cv2.imwrite("../OutputTMP/Frame{}.png".format(i), frame)
                        break
                # merge_fig(img_mobilenet, frame, i)

            frame = cv2.resize(frame, (640, 408), interpolation=cv2.INTER_LINEAR)
            cv2.moveWindow("Original", 0, 50)
            cv2.imshow("Original", frame)

        cv2.moveWindow("Segmented", 640, 50)
        cv2.imshow("Segmented", img_original)
        cv2.moveWindow("Phase 1 - First Erosion", 1280, 50)
        cv2.imshow("Phase 1 - First Erosion", img_first_erosion)
        cv2.moveWindow("Phase 1 - Dilation", 0, 500)
        cv2.imshow("Phase 1 - Dilation", img_dilated)
        cv2.moveWindow("Phase 1 - Erosion", 640, 500)
        cv2.imshow("Phase 1 - Erosion", img_eroded)
        #cv2.moveWindow("Phase 2 - Gray + Edge", 1280, 420)
        #cv2.imshow("Phase 2 - Gray + Edge", img_edged)
        cv2.moveWindow("Phase 3 - Final", 1280, 500)
        cv2.imshow("Phase 3 - Final", img_final)

        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

        i += 1

    cap.release()
    cv2.destroyAllWindows()