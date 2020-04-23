from __future__ import print_function
from typing import List, Tuple
import cv2
import numpy as np
from keras_segmentation.predict import predict, model_from_checkpoint_path
from scipy.ndimage import gaussian_filter
import argparse
import imutils
from math import sin, cos, sqrt
from vtk import *
import threading
import time

class_colors: List[Tuple[int, int, int]] = [(0, 0, 0), (255, 0, 0), (0, 255, 0)]

scale_percent = 60  # percent of original size

z_angle = None
x_angle = None
area = None
name_path = None
cX = None
cY = None
height = None
width = None

def convert_np_to_mat(img):
    seg_img = np.zeros((img.shape[0], img.shape[1], 3))
    colors = class_colors

    for c in range(3):  # with 3 classes

        seg_img[:, :, 0] += ((img[:, :] == c) * (colors[c][0])).astype('float')
        seg_img[:, :, 1] += ((img[:, :] == c) * (colors[c][1])).astype('float')
        seg_img[:, :, 2] += ((img[:, :] == c) * (colors[c][2])).astype('float')

    seg_img = cv2.resize(seg_img, (960, 540))

    return seg_img
def build_line(p1, p2):
    a = (p1[1] - p2[1])
    b = (p2[0] - p1[0])
    c = (p1[0] * p2[1] - p2[0] * p1[1])
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
def close_window(iren):
    render_window = iren.GetRenderWindow()
    render_window.Finalize()
    iren.TerminateApp()
def apex(x1, y1, x2, y2):
    return np.int((x1 + x2) / 2), np.int((y1 + y2) / 2)
# funzione che trasforma un punto su un piano in un punto 3d nello spazio della telecamera
def get_3d_matrix(point):
    point_3d = [point[0], 500 - point[1], 0]  # 0 è la distanza dallo schermo per ora, 500 metà della finestra

    origin, direction = converTo3DRay(point_3d)
    # converto to 3d ray,

    '''
    Affine3d ret;
    Vec3d v;

    if (isAutoScaling) computeSize(); // è attivo lo scalamento automatico, lo computa...
    v[0] = origin.x + direction.val[0] * sz;
    v[1] = origin.y + direction.val[1] * sz;
    v[2] = origin.z + direction.val[2] * sz;

    // cout << "v: " << v << endl;
    ret.translation(v);

    return ret;

    '''
# a questa io passo il punto di aggancio e alpa, e lui restituisce la matrice da applicare a tutti gli oggetti (cioè al catetere nel nostro caso) per spostarlo li
def position_and_rotate(apex, alpha):
    wm = []
    wm3 = []

    # CHE MATRICI SONO? 3X3?
    wm[0] = wm[5] = cos(alpha)
    wm[1] = -sin(alpha)
    wm[4] = sin(alpha)

    # COME RICAVO BETA?
    wm3[5] = wm3[10] = cos(beta_ / 180.0 * np.pi)
    wm3[6] = -sin(beta_ / 180.0 * np.pi)
    wm3[9] = sin(beta_ / 180.0 * np.pi)

    wm = get_3d_matrix(apex) * wm * wm3

    # poi devo applicare questo all'oggetto catetere
def project_point_plane(pointxy, point_z=1.0, origin=[0, 0, 0], normal=[0, 0, 1]):
    projected_point = np.zeros(3)
    p = [pointxy[0], 540 - pointxy[1], point_z]
    vtkPlane.ProjectPoint(p, origin, normal, projected_point)
    return projected_point
def getScreenshot(self, fname, mag=10):
    r"""
    Generate a screenshot of the window and save to a png file

    Parameters
    ----------
    fname: str
        The file handle to save the image to
    mag: int, default 10
        The magnificaiton of the image, this will scale the resolution of
        the saved image by this face

    """
    self.SetAlphaBitPlanes(1)
    w2if = vtk.vtkWindowToImageFilter()
    w2if.SetInput(self)
    w2if.SetScale(mag)
    w2if.SetInputBufferTypeToRGBA()
    w2if.Update()

    writer = vtk.vtkPNGWriter()
    writer.SetFileName(fname)
    writer.SetInputConnection(w2if.GetOutputPort())
    writer.Write()

class vtkTimerCallback():
    def __init__(self):
        self.timer_count = 0

    def execute(self, obj, event):
        # print(self.timer_count)
        print("i = " + str(i))
        bounds = self.actor.GetBounds()
        # Compute original width and height
        original_h = abs(bounds[3] - bounds[2])
        original_w = abs(bounds[1] - bounds[0])
        # project 2d point to 3d and set position
        a = project_point_plane((cX, cY))
        self.actor.SetPosition(a[0], a[1], a[2])
        # found scaling for x (same as z) and y
        sy = height / original_h
        sx = width / original_w
        #self.actor.SetScale(sx, sy, sx)
        #self.actor.RotateZ(-z_angle)
        print(cX, cY)
        iren = obj
        iren.GetRenderWindow().Render()
        self.timer_count += 1
        time.sleep(1)
        getScreenshot(iren.GetRenderWindow(), "output{}.png".format(i), 1)


def overlay_catheter():
    reader_img = vtkPNGReader()
    reader_img.SetFileName(name_path)
    reader_img.Update()
    image_data = reader_img.GetOutput()
    size = image_data.GetExtent()
    image_geometry_filter = vtkImageDataGeometryFilter()
    image_geometry_filter.SetInputConnection(reader_img.GetOutputPort())
    image_geometry_filter.Update()

    mapper_img = vtkPolyDataMapper()
    mapper_img.SetInputConnection(image_geometry_filter.GetOutputPort())

    actor_img = vtkActor()
    actor_img.SetMapper(mapper_img)

    path_obj = "..\\Germano\\Oriented\\CATETHER.obj"
    reader_obj = vtkOBJReader()
    reader_obj.SetFileName(path_obj)

    mapper_obj = vtkPolyDataMapper()
    mapper_obj.SetInputConnection(reader_obj.GetOutputPort())

    actor_obj = vtkActor()
    actor_obj.SetMapper(mapper_obj)  # oggetto invisibile finchè non lo metto nella piepeline, ci entra con il mapper
    bounds = actor_obj.GetBounds()
    # Compute original width and height
    original_h = abs(bounds[3] - bounds[2])
    original_w = abs(bounds[1] - bounds[0])
    # project 2d point to 3d and set position
    a = project_point_plane((cX, cY))
    actor_obj.SetPosition(a[0], a[1], a[2])
    # found scaling for x (same as z) and y
    sy = height / original_h
    sx = width / original_w
    actor_obj.SetScale(sx, sy, sx)
    actor_obj.RotateZ(-z_angle)  # destra sinitra
    if x_angle == -1:
        actor_obj.RotateX(-10)
    elif x_angle == 1:
        actor_obj.RotateX(10)

    # Setup rendering
    renderer = vtkRenderer()
    renderer.AddActor(actor_obj)
    renderer.AddActor(actor_img)
    renderer.SetBackground(0, 0, 0)
    # renderer.ResetCamera()

    render_window = vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(1000, 1000)

    render_window_interactor = vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)
    render_window_interactor.Initialize()

    cb = vtkTimerCallback()
    cb.actor = actor_obj
    render_window_interactor.AddObserver('TimerEvent', cb.execute)
    timerId = render_window_interactor.CreateRepeatingTimer(100)

    render_window_interactor.Start()

    getScreenshot(render_window, "output{}.png.format(i)", 1)


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True, help="Select the model to use "
                                                         "(0 for MobileNet, 1 for VGG, 2 for ResNet, 3 for Unet")
    args = vars(ap.parse_args())
    run_model = int(args["model"])
    models = ["MobileNet", "VGG", "ResNet", "U-Net"]
    print("Running the {} model.".format(models[run_model]))

    videoFilePath = "..\\RealTime Video\\CV_2_Cropped.mp4"

    if run_model == 0:
        model = model_from_checkpoint_path("..\\Checkpoints\\NewTool\\new_mobilenet_unet_tool")
    elif run_model == 1:
        model = model_from_checkpoint_path("..\\Checkpoints\\NewTool\\new_vgg_unet_tool")
    elif run_model == 2:
        model = model_from_checkpoint_path("..\\Checkpoints\\NewTool\\new_resnet_unet_tool")
    elif run_model == 3:
        model = model_from_checkpoint_path("..\\Checkpoints\\NewTool\\new_unet_tool")

    cap = cv2.VideoCapture(videoFilePath)

    i = 0

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        if (i % 15) == 0:

            dim = (960, 540)

            frame = cv2.resize(frame, dim, interpolation=cv2.INTER_LINEAR)
            name_path = "../OutputTMP/{}.png".format(i)
            cv2.imwrite(name_path, frame)
            angle = 0.0
            x_angle = 0

            out = model.predict_segmentation(inp=frame)
            img = convert_np_to_mat(out)

            (channel_b, channel_g, channel_r) = cv2.split(img)  # ciascuna con un solo canale

            contours_b, hierarchy_b = cv2.findContours(channel_b.astype('uint8'), cv2.RETR_TREE,
                                                       cv2.CHAIN_APPROX_SIMPLE)
            contours_g, hierarchy_g = cv2.findContours(channel_g.astype('uint8'), cv2.RETR_TREE,
                                                       cv2.CHAIN_APPROX_SIMPLE)

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
                area = cv2.contourArea(hull)

                # draw the convex hull contour in a blank image
                img = np.zeros([540, 960, 3], dtype=np.uint8)
                img.fill(0)
                cv2.drawContours(img, [hull], 0, (0, 0, 150), 2)

                ext_left = tuple(max_c[max_c[:, :, 0].argmin()][0])
                ext_right = tuple(max_c[max_c[:, :, 0].argmax()][0])
                ext_top = tuple(max_c[max_c[:, :, 1].argmin()][0])
                ext_bot = tuple(max_c[max_c[:, :, 1].argmax()][0])
                M = cv2.moments(max_c)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.circle(output, (cX, cY), 7, (0, 0, 0), -1)
                # cv2.circle(output, ext_bot, 7, (0, 0, 0), -1)
                # cv2.circle(output, ext_top, 7, (0, 0, 0), -1)
                # cv2.circle(output, ext_left, 7, (0, 0, 0), -1)
                # cv2.circle(output, ext_right, 7, (0, 0, 0), -1)

                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                kernel_dilate = np.ones((7, 7), np.uint8)
                img = cv2.dilate(img, kernel_dilate, iterations=1)

                cv2.imshow("Original Image", img)

                # output = cv2.resize(frame, (640, 408), interpolation=cv2.INTER_LINEAR)
                # img = cv2.resize(img, (640, 408), interpolation=cv2.INTER_LINEAR)
                # frame = cv2.resize(frame, (640, 408), interpolation=cv2.INTER_LINEAR)

                minLineLength = 50
                maxLineGap = 0
                lines = cv2.HoughLinesP(img, 1, np.pi / 180, 100, minLineLength, maxLineGap)

                if lines is not None:
                    n_line = 0
                    for line in lines:
                        for x1, y1, x2, y2 in line:
                            if n_line == 0:
                                n_line += 1
                                l1 = build_line([x1, y1], [x2, y2])
                                cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
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
                                    cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                    break
                        if n_line == 2:
                            inter_x, inter_y = intersection(l1, l2)

                            if inter_y < 0:
                                inter_y = 0
                            if inter_y > 540:
                                inter_y = 540
                            if inter_x < 0:
                                inter_x = 0
                            if inter_x > 960:
                                inter_x = 960
                            # print(inter_x, inter_y)
                            if cX - 100 < inter_x < cX + 100:
                                cv2.circle(output, (inter_x, inter_y), 10, (255, 0, 0), thickness=10)
                                if inter_y > 270:
                                    x_angle = 1  # interseca sopra
                                elif inter_y <= 270:
                                    x_angle = -1  # interseca sotto
                            break
                # merge_fig(img_mobilenet, frame, i)

                # epsilon = 0.03 * cv2.arcLength(max_c, True)
                # approx = cv2.approxPolyDP(max_c, epsilon, True)
                # cv2.drawContours(output, [approx], 0, (0, 0, 50), 2)

                if len(max_c) > 5:
                    rot_rect = cv2.fitEllipse(max_c)
                    z_angle = rot_rect[2]
                    box = cv2.boxPoints(rot_rect)
                    box = np.int0(box)
                    a = box[0]
                    # Trovo il punto di apice
                    apex_point = apex(box[0][0], box[0][1], box[3][0], box[3][1])
                    cv2.circle(output, apex_point, 2, (255, 0, 0), thickness=2)
                    # Trovo l'altezza e la larghezza del catetere
                    height = sqrt((box[0][0] - box[1][0]) ** 2 + (box[0][1] - box[1][1]) ** 2)
                    width = sqrt((box[0][0] - box[3][0]) ** 2 + (box[0][1] - box[3][1]) ** 2)

                    if z_angle > 90:
                        z_angle = z_angle - 180
                    print("Z_Angle is {:.2f}".format(z_angle))

                print("X_angle is {:.2f}".format(x_angle))

                cv2.imshow("Output", output)

                if i == 15:
                    t = threading.Thread(target=overlay_catheter)
                    t.start()

                # time.sleep(5)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        i += 1

    cap.release()
    cv2.destroyAllWindows()