import numpy as np
from vtk import *

# 540 è l'altezza dello schermo
def project_point_plane(pointxy, point_z=1.0, origin=[0, 0, 0], normal=[0, 0, 1]):
    projected_point = np.zeros(3)
    p = [pointxy[0], 540 - pointxy[1], point_z]
    vtkPlane.ProjectPoint(p, origin, normal, projected_point)
    return projected_point


def get_screenshot(ren_win, fname, mag=10):
    """
    fname: str
        The file handle to save the image to
    mag: int, default 10
        The magnificaiton of the image, this will scale the resolution of
        the saved image by this face

    """
    ren_win.SetAlphaBitPlanes(1)
    w2if = vtk.vtkWindowToImageFilter()
    w2if.SetInput(ren_win)
    w2if.SetScale(mag)
    w2if.SetInputBufferTypeToRGBA()
    w2if.Update()

    writer = vtk.vtkPNGWriter()
    writer.SetFileName(fname)
    writer.SetInputConnection(w2if.GetOutputPort())
    writer.Write()


# queste due funzioni sono il tentativo (terribile) di riscrivere le tue da c++, non le utilizzo per ora
''''
def get_3d_matrix(point):
    point_3d = [point[0], 500 - point[1], 0]  # 0 è la distanza dallo schermo per ora, 500 metà della finestra

    origin, direction = converTo3DRay(point_3d)
    # converto to 3d ray,

    Affine3d ret;
    Vec3d v;

    if (isAutoScaling) computeSize(); // è attivo lo scalamento automatico, lo computa...
    v[0] = origin.x + direction.val[0] * sz;
    v[1] = origin.y + direction.val[1] * sz;
    v[2] = origin.z + direction.val[2] * sz;

    // cout << "v: " << v << endl;
    ret.translation(v);

    return ret;

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
'''