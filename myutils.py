import numpy as np
from typing import List, Tuple
import cv2

class_colors: List[Tuple[int, int, int]] = [(0, 0, 0), (255, 0, 0), (0, 255, 0)]
scale_percent = 60  # percent of original size

# classe buffer circolare
class CircularBuffer:

    def __init__(self, len):
        self.len = len
        self.data = []
        self.count = 0
    # aggiunge elemento al buffer
    def add_to_buf(self, val):
        self.count += 1

        if self.count <= self.len:
            self.data.append(val)
        else:
            for i in range(self.len - 1):
                tmp = self.data[i + 1]
                self.data[i] = tmp
            self.data[self.len - 1] = val

        return self.data
    # ritorna lista coi valori
    def get(self):
        return self.data
    # valore medio degli angoli
    def avg(self):
        m = np.mean(self.data)
        if self.count == 0:
             return 0
        else:
            return m


# restituisce angolo in gradi, se si vogliono radianti basta togliere np.degrees
def compute_angle(ax, ay, bx, by, cx, cy, dx, dy):
    alpha0 = np.degrees(np.arctan2(by - ay, bx - ax))
    alpha1 = np.degrees(np.arctan2(dy - cy, dx - cx))
    return abs(alpha1 - alpha0)


def compute_angle_given_coeff(m1, m2):
    tang = abs ((m2 - m1) / (1 + m1 * m2 + 0.0000001))
    return np.arctan(tang)


# dati due punti e un valore di y, trova i valori di x corrispondente dall'equazione della retta
def obtain_universalx(x1, y1, x2, y2, y_gen):
    m = (y2 - y1) / (x2 - x1 + + 0.00000001)
    b = y1 - m*x1
    return (y_gen - b) / m


def find_line_from_coef(m, xp, yp):
    y1 = 1
    x1 = int(((y1 - yp) / (m*(-1))) + xp)
    y2 = 539
    x2 = int(((y2 - yp) / (m*(-1))) + xp)
    return (x1, y1), (x2, y2)



def convert_np_to_mat(img):
    seg_img = np.zeros((img.shape[0], img.shape[1], 3))
    colors = class_colors

    for c in range(3):  # with 3 classes

        seg_img[:, :, 0] += ((img[:, :] == c) * (colors[c][0])).astype('float')
        seg_img[:, :, 1] += ((img[:, :] == c) * (colors[c][1])).astype('float')
        seg_img[:, :, 2] += ((img[:, :] == c) * (colors[c][2])).astype('float')

    seg_img = cv2.resize(seg_img, (960, 540))

    return seg_img


# restituisce l'intersezione fra due linee
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


# costruisce una linea
def build_line(p1, p2):
    a = (p1[1] - p2[1])
    b = (p2[0] - p1[0])
    c = (p1[0] * p2[1] - p2[0] * p1[1])
    return a, b, -c


def take_second(elem):
    return elem[1]


# restituisce l'apice
def apex(v):
    v = sorted(v, key=take_second)
    p = np.int((v[0][0] + v[1][0]) / 2), np.int((v[0][1] + v[1][1]) / 2)
    return p

