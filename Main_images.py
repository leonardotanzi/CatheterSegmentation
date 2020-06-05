from __future__ import print_function

import sys

from keras_segmentation.predict import predict, model_from_checkpoint_path
from math import sin, cos, sqrt
import shutil
import os
from myutils import *
from myutils3D import *
from os import walk
from vtk.vtkIOKitPython import vtkOBJReader
from vtk.vtkRenderingKitPython import vtkPolyDataMapper, vtkActor
import numpy as np
from tqdm import tqdm
import argparse

class_colors: List[Tuple[int, int, int]] = [(0, 0, 0), (255, 0, 0), (0, 255, 0)]

# qualche global
obj_path = "Objects\\"
dim = (960, 540)  # dimensioni finestra

freezed = False
autoRotateX = False
frameNumber = 0
slide = 0
viewangle = 45  # di default vtk mette 30

next_step = False
i = 0
name_image = ""
folder = ""
finish = False

modelfiles = []
modelactors = list()
modelopacity = list()
modelmappers = list()
modelnames = list()  # qui ci sono i soli nomi delle mesh. L'indice del nome corrisponde all'indice dell'actor.

world_matrix = vtk.vtkMatrix4x4()  # la matrice di posizionamento delle mesh, globale

sceneRen = vtk.vtkRenderer()  # questo per il catetere e la prostata 3D
sceneRen.SetLayer(1)
backgr = vtk.vtkRenderer()  # questo per l'immagine di sfondo
backgr.SetLayer(0)
renWin = vtk.vtkRenderWindow()

# scala automatica del catetere
STARTING_SIZE = 80.0
distance = STARTING_SIZE  # distanza della prostata.
STARTING_BETA = 0.0
beta = STARTING_BETA  # rotazione della prostata antero-posteriore
registeredLenght = 80.0  # lunghezza in pixel del catetere... viene modificata a mano coi tasti + e -
newLenght = 1.0  # lunghezza della retta che interseca il punto d'apice (segmento superiore del bbox)

x_angle_buffer = [0, 0,
                  0]  # in posizione [0] c'è l'angolo del frame attuale, in [1] quello del frame -1, in [2] quello del frame -2 in radianti

# per le operazioni di filtraggio
lastframeapex = [0, 0]
lastframedistance = 0
lastframeaZangle = 0
lastframeaXangle = 0

# GUI globals
fov_text_actor = vtk.vtkTextActor()
size_text_actor = vtk.vtkTextActor()
xrot_text_actor = vtk.vtkTextActor()
slide_text_actor = vtk.vtkTextActor()
freezed_text_actor = vtk.vtkTextActor()
isBoxWidgetOn = False
boxWidget = vtk.vtkBoxWidget()

# aggiungere un tasto che quando si preme T salva in excel e interrompe il ciclo

class MyInteractorStyle(vtk.vtkInteractorStyle):

    def __init__(self, parent=None):

        self.parent = iren
        self.AddObserver("KeyPressEvent", self.keyPressEvent)
        self.AddObserver("CharEvent", self.charEvent)

    def keyPressEvent(self, obj, event):
        global STARTING_SIZE, STARTING_BETA, sceneRen
        global freezed, slide, viewangle, autoRotateX
        global fov_text_actor, size_text_actor, xrot_text_actor, slide_text_actor, freezed_text_actor, frameNumber
        global isBoxWidgetOn
        global next_step, i, name_image, folder, finish
        # usare global per importare variabili,

        key = self.parent.GetKeySym()
        if key == 'Shift_L':
            SwitchAllMeshOpacity()

        elif key == "t":
            print("beta {}, starting beta {}".format(beta, STARTING_BETA))
            edit_excel(folder, beta, name_image, i)
            next_step = True
        elif key == "q":
            finish = True

        elif key == '0':
            SetAllMeshOpacity(1)
        elif key == '1':
            SwitchMeshOpacity("iles")
        elif key == '2':
            SwitchMeshOpacity("eles")
        elif key == '3':
            SwitchMeshOpacity("inte")
            SwitchMeshOpacity("porz")
        elif key == '4':
            SwitchMeshOpacity("lfas")
        elif key == '5':
            SwitchMeshOpacity("pros")
        elif key == '6':
            SwitchMeshOpacity("rfas")
        elif key == '7':
            SwitchMeshOpacity("uret")
        elif key == '8':
            SwitchMeshOpacity("sfin")
        elif key == 'minus':
            STARTING_SIZE -= 20
            size_text_actor.SetInput("C.Size: " + str(STARTING_SIZE))
        elif key == 'plus':
            STARTING_SIZE += 20
            size_text_actor.SetInput("C.Size: " + str(STARTING_SIZE))
        elif key == 'n':
            STARTING_BETA -= 0.02
            xrot_text_actor.SetInput("X Rot.: " + str(round(beta, 3)))
        elif key == 'm':
            STARTING_BETA += 0.02
            xrot_text_actor.SetInput("X Rot.: " + str(round(beta, 3)))
        elif key == 'x':
            freezed = not freezed

            if freezed:
                freezed_text_actor.SetInput("Tracking off!")
                freezed_text_actor.GetTextProperty().SetColor((1, 0, 0))
            else:
                freezed_text_actor.SetInput("Tracking on")
                freezed_text_actor.GetTextProperty().SetColor((0, 1, 0))

        elif key == 'z':
            if (autoRotateX == False):
                autoRotateX = True
                frameNumber = 0
        elif key == 'p':
            slide += 1
            slide_text_actor.SetInput("Slide: " + str(slide))
        elif key == 'l':
            slide -= 1
            slide_text_actor.SetInput("Slide: " + str(slide))
        elif key == 'o':
            viewangle += 1
            sceneRen.GetActiveCamera().SetViewAngle(viewangle)
            fov_text_actor.SetInput("FOV: " + str(viewangle))
        elif key == 'k':
            viewangle -= 1
            sceneRen.GetActiveCamera().SetViewAngle(viewangle)
            fov_text_actor.SetInput("FOV: " + str(viewangle))
        elif key == 'b':
            isBoxWidgetOn = not isBoxWidgetOn
            boxWidget.SetEnabled(isBoxWidgetOn)
            # print(boxWidget)
        elif key == 'v':
            resetMeshDeformation()
        elif key == 'Escape':
            iren.GetRenderWindow().Finalize()
            iren.TerminateApp()
            cap.release()
        print(key)
        return

    def charEvent(self, obj, event):  # necessario per bloccare gli eventi automatici della finestra
        return


def MakeAxesActor():
    axes = vtk.vtkAxesActor()
    axes.SetShaftTypeToCylinder()
    axes.SetXAxisLabelText('X')
    axes.SetYAxisLabelText('Y')
    axes.SetZAxisLabelText('Z')
    axes.SetTotalLength(1.0, 1.0, 1.0)
    axes.SetCylinderRadius(0.5 * axes.GetCylinderRadius())
    axes.SetConeRadius(1.025 * axes.GetConeRadius())
    axes.SetSphereRadius(1.5 * axes.GetSphereRadius())
    return axes


def SwitchSingleMeshOpacity(meshid):
    if modelopacity[meshid] == 1:
        modelactors[meshid].GetProperty().SetOpacity(0.5)
        modelopacity[meshid] = 0.5
    elif modelopacity[meshid] == 0.5:
        modelactors[meshid].GetProperty().SetOpacity(0)
        modelopacity[meshid] = 0
    elif modelopacity[meshid] == 0:
        modelactors[meshid].GetProperty().SetOpacity(1)
        modelopacity[meshid] = 1


def SwitchMeshOpacity(name):
    # recupera l'indice dell'actor della mesh con questo nome

    for i, n in enumerate(modelnames):
        if (n[0:4] == name):
            SwitchSingleMeshOpacity(
                i)  # fatto in questo modo dovrebbe funzionare per tutte le mesh che iniziano con lo stesso nome


def SetAllMeshOpacity(value):
    for a in modelactors:
        a.GetProperty().SetOpacity(value)


def SwitchAllMeshOpacity():
    for i in range(len(modelactors)):
        if (modelopacity[i] == 1):
            SetAllMeshOpacity(0.5)
            modelopacity[i] = 0.5
        elif (modelopacity[i] == 0.5):
            SetAllMeshOpacity(0)
            modelopacity[i] = 0
        elif (modelopacity[i] == 0):
            SetAllMeshOpacity(1)
            modelopacity[i] = 1


def Get3dWorldPointFrom2dPoint(point2d):
    # retMat = np.eye(4) #matrice 4x4 con gli 1 in diagonale

    #per fare in modo che la finestra sia ridimensionabile considero la dimensione finestra
    w, h = renWin.GetSize()
    x = (point2d[0] * w) / dim[0]
    y = ((dim[1] - point2d[1]) * h) / dim[1]# correggo l'orientamento del punto sull'immagine per l'asse Y

    coord = vtk.vtkCoordinate()
    coord.SetCoordinateSystemToViewport()
    #coord.SetValue(point2d[0], dim[1] - point2d[1])  # correggo l'orientamento del punto sull'immagine per l'asse Y
    coord.SetValue(x, y)
    cx, cy, cz = coord.GetComputedWorldValue(sceneRen)

    origin = sceneRen.GetActiveCamera().GetPosition()
    # direction = findVec(origin, [cx, cy, cz])
    direction = np.array(findVec(origin, [cx, cy, cz]))

    normalized_direction = direction / np.sqrt(np.sum(direction ** 2))

    vx = origin[0] + normalized_direction[0] * distance
    vy = origin[1] + normalized_direction[1] * distance
    vz = origin[2] + normalized_direction[2] * distance

    # print("origine camera   : " + str(origin[0]) + ", " + str(origin[1]) + ", " + str(origin[2]))
    # print("vettore direzione: " + str(normalized_direction[0]) + ", " + str(normalized_direction[1]) + ", " + str(normalized_direction[2]))
    # print("valori finali    : " + str(vx) + ", " + str(vy) + ", " + str(vz))

    return [vx, vy, vz]


# effettua un ciclo su ogni actor (mesh) e imposta la sua world matrix
def SetWorldMatrix(apx):
    for a in modelactors:
        # a.SetPosition(apx)
        a.SetUserMatrix(apx)


# dato un punto e un angolo X + un angolo Y crea la matrice di roto/traslazione
def CreateWorldMatrix(apex3d, angleZ):
    t = vtk.vtkMatrix4x4()
    rz = vtk.vtkMatrix4x4()
    # ry = vtk.vtkMatrix4x4()
    rx = vtk.vtkMatrix4x4()
    w = vtk.vtkMatrix4x4()

    # traslazione
    t.SetElement(0, 3, apex3d[0])
    t.SetElement(1, 3, apex3d[1])
    t.SetElement(2, 3, apex3d[2])

    # setting Z rotation
    rz.SetElement(0, 0, cos(angleZ))
    rz.SetElement(1, 1, cos(angleZ))
    rz.SetElement(0, 1, -sin(angleZ))
    rz.SetElement(1, 0, sin(angleZ))

    # rotazione y (+180 se mesh caricata storta)
    # ry.SetElement(0, 0, cos(np.pi))
    # ry.SetElement(0, 2, sin(np.pi))
    # ry.SetElement(2, 0, -sin(np.pi))
    # ry.SetElement(2, 2, cos(np.pi))

    # rotazione x
    rx.SetElement(1, 1, cos(beta))
    rx.SetElement(1, 2, -sin(beta))
    rx.SetElement(2, 1, sin(beta))
    rx.SetElement(2, 2, cos(beta))

    # setting X rotation w = t * rz * ry * rx
    vtk.vtkMatrix4x4().Multiply4x4(t, rz, w)
    vtk.vtkMatrix4x4().Multiply4x4(w, rx, w)
    # vtk.vtkMatrix4x4().Multiply4x4(w, ry, w)
    return w


def SetMeshColor(name):
    nc = vtk.vtkNamedColors()

    if name[0:4].lower() == "eles":
        color = nc.GetColor3d("Lime")  # verde
    if name[0:4].lower() == "iles":
        color = nc.GetColor3d("DarkGreen")  # verde scuro
    if name[0:4].lower() == "lfas":
        color = nc.GetColor3d("Blue")  # blu
    if name[0:4].lower() == "rfas":
        color = nc.GetColor3d("Blue")  # blu
    if name[0:4].lower() == "inte":
        color = nc.GetColor3d("Black")  # nero
    if name[0:4].lower() == "porz":
        color = nc.GetColor3d("Orange")  # arancione
    if name[0:4].lower() == "pros":
        color = nc.GetColor3d("Gray")  # grigio
    if name[0:4].lower() == "sfin":
        color = nc.GetColor3d("DarkOrange")  # arancione scuro
    if name[0:4].lower() == "uret":
        color = nc.GetColor3d("Yellow")  # giallo
    return color


# funzione che carica i file diversi file della prostata e li aggiunge al latey di renderizzazione
def load_models_and_create_actors():
    for (dirpath, dirnames, filenames) in walk(obj_path):
        modelfiles = [fi for fi in filenames if fi.endswith(".obj")]
        break

    idx = 0
    for f in modelfiles:  # ogni mesh ha bisogno del suo actor e mapper
        print("Importing " + f)
        modelnames.append(os.path.splitext(f.lower())[0])  # tolgo l'estensione dal nome

        importer = vtkOBJReader()
        importer.SetFileName(obj_path + f)
        importer.Update()

        modelmappers.append(vtkPolyDataMapper())
        modelmappers[idx].SetInputConnection(importer.GetOutputPort())

        modelactors.append(vtkActor())
        modelactors[idx].SetMapper(modelmappers[idx])

        modelactors[idx].GetProperty().SetColor(SetMeshColor(f))
        modelopacity.append(1)

        sceneRen.AddActor(modelactors[idx])
        idx += 1


def SetTextWidgets():
    fov_text_actor.SetInput("FOV: " + str(viewangle))
    fov_text_actor.GetTextProperty().SetColor((1, 1, 1))

    size_text_actor.SetInput("C.Size: " + str(STARTING_SIZE))
    size_text_actor.GetTextProperty().SetColor((1, 1, 1))

    xrot_text_actor.SetInput("X Rot.: " + str(beta))
    xrot_text_actor.GetTextProperty().SetColor((1, 1, 1))

    slide_text_actor.SetInput("Slide: " + str(slide))
    slide_text_actor.GetTextProperty().SetColor((1, 1, 1))

    freezed_text_actor.SetInput("Tracking on")
    freezed_text_actor.GetTextProperty().SetColor((0, 1, 0))

    sceneRen.AddActor(fov_text_actor)
    sceneRen.AddActor(size_text_actor)
    sceneRen.AddActor(xrot_text_actor)
    sceneRen.AddActor(slide_text_actor)
    sceneRen.AddActor(freezed_text_actor)

    # Create the text representation. Used for positioning the text_actor
    text_representation1 = vtk.vtkTextRepresentation()
    text_representation1.GetPositionCoordinate().SetValue(0.01, 0.85)

    text_representation2 = vtk.vtkTextRepresentation()
    text_representation2.GetPositionCoordinate().SetValue(0.01, 0.75)

    text_representation3 = vtk.vtkTextRepresentation()
    text_representation3.GetPositionCoordinate().SetValue(0.01, 0.65)

    text_representation4 = vtk.vtkTextRepresentation()
    text_representation4.GetPositionCoordinate().SetValue(0.01, 0.55)

    text_representation5 = vtk.vtkTextRepresentation()
    text_representation5.GetPositionCoordinate().SetValue(0.01, 0.25)

    text_widget1 = vtk.vtkTextWidget()
    text_widget1.SetRepresentation(text_representation1)
    text_widget1.SetInteractor(iren)
    text_widget1.SetTextActor(fov_text_actor)
    text_widget1.GetTextActor().SetTextScaleModeToNone()
    text_widget1.GetTextActor().GetTextProperty().SetJustificationToLeft()
    text_widget1.GetTextActor().GetTextProperty().SetFontSize(20)
    text_widget1.SelectableOff()
    text_widget1.On()

    text_widget2 = vtk.vtkTextWidget()
    text_widget2.SetRepresentation(text_representation2)
    text_widget2.SetInteractor(iren)
    text_widget2.SetTextActor(size_text_actor)
    text_widget2.GetTextActor().SetTextScaleModeToNone()
    text_widget2.GetTextActor().GetTextProperty().SetJustificationToLeft()
    text_widget2.GetTextActor().GetTextProperty().SetFontSize(20)
    text_widget2.SelectableOff()
    text_widget2.On()

    text_widget3 = vtk.vtkTextWidget()
    text_widget3.SetRepresentation(text_representation3)
    text_widget3.SetInteractor(iren)
    text_widget3.SetTextActor(xrot_text_actor)
    text_widget3.GetTextActor().SetTextScaleModeToNone()
    text_widget3.GetTextActor().GetTextProperty().SetJustificationToLeft()
    text_widget3.GetTextActor().GetTextProperty().SetFontSize(20)
    text_widget3.SelectableOff()
    text_widget3.On()

    text_widget4 = vtk.vtkTextWidget()
    text_widget4.SetRepresentation(text_representation4)
    text_widget4.SetInteractor(iren)
    text_widget4.SetTextActor(slide_text_actor)
    text_widget4.GetTextActor().SetTextScaleModeToNone()
    text_widget4.GetTextActor().GetTextProperty().SetJustificationToLeft()
    text_widget4.GetTextActor().GetTextProperty().SetFontSize(20)
    text_widget4.SelectableOff()
    text_widget4.On()

    text_widget5 = vtk.vtkTextWidget()
    text_widget5.SetRepresentation(text_representation5)
    text_widget5.SetInteractor(iren)
    text_widget5.SetTextActor(freezed_text_actor)
    text_widget5.GetTextActor().SetTextScaleModeToNone()
    text_widget5.GetTextActor().GetTextProperty().SetJustificationToLeft()
    text_widget5.GetTextActor().GetTextProperty().SetFontSize(18)
    text_widget5.SelectableOff()
    text_widget5.On()


def resetMeshDeformation():
    for i, ma in enumerate(modelactors):
        ma.SetMapper(modelmappers[i])


def SetBoxWidget():
    boxWidget.SetInteractor(iren)
    for i, n in enumerate(modelnames):
        if (n[0:4] == "pros"):
            boxWidget.SetProp3D(modelactors[i])
    boxWidget.SetPlaceFactor(1.5)  # Make the box larger than the actor
    boxWidget.PlaceWidget()
    # boxWidget.AddObserver("InteractionEvent", boxCallback)
    boxWidget.AddObserver("EndInteractionEvent", boxEndInteractionCallback)
    boxWidget.AddObserver("InteractionEvent", boxInteractionCallback)


def boxEndInteractionCallback(obj, event):
    for i, n in enumerate(modelnames):
        if (n[0:4] == "pros"):
            boxWidget.SetProp3D(modelactors[i])
            #boxWidget.SetPlaceFactor(1.5)
            #boxWidget.PlaceWidget()


def boxInteractionCallback(obj, event):
    pd_box = vtk.vtkPolyData()  # i dati DEI VERTICI DELLA box widget
    obj.GetPolyData(pd_box)

    # triangolo la mesh del boxWidged che ha le facce quad alrimenti SetControlMeshData si incazza
    tringlefilter = vtk.vtkTriangleFilter()
    tringlefilter.SetInputData(pd_box)
    tringlefilter.Update()

    for i, n in enumerate(modelnames):

        deform = vtk.vtkDeformPointSet()  # questo effettivamente deforma secondo il box del widget
        deform.SetInputData(modelactors[i].GetMapper().GetInput())
        deform.SetControlMeshData(tringlefilter.GetOutput())  # il boxWidget Triangolato è la mesh che guida la deformazione
        deform.Update()
        polyMapper = vtk.vtkPolyDataMapper()
        polyMapper.SetInputConnection(deform.GetOutputPort())
        modelactors[i].SetMapper(polyMapper)
        modelactors[i].SetUserMatrix(world_matrix)


def MatToVtKImageRender(imageToVtk):
    # qui trasformiamo img in una immagine buona per VTK
    dataImporter.SetDataSpacing(1, 1, 1)
    dataImporter.SetWholeExtent(0, dim[0] - 1, 0, dim[1] - 1, 0, 2)
    dataImporter.SetDataExtentToWholeExtent()
    dataImporter.SetDataScalarTypeToUnsignedChar()
    dataImporter.SetNumberOfScalarComponents(3)
    dataImporter.SetImportVoidPointer(imageToVtk)
    dataImporter.Update()
    # e la applichiamo all'attore...
    imageActor.SetInputData(dataImporter.GetOutput())
    # Rieseguiamo 1 render (MI CI SONO SPACCATO LA TESTA: viene richiamato una prima volta affinchè crei l'oggetto camera)
    renWin.Render()
    # Serie di operazioni ridicolmente complesse per adattare l'immagine alla finestra (sennò viene piccola)
    origin = dataImporter.GetOutput().GetOrigin()
    spacing = dataImporter.GetOutput().GetSpacing()
    extent = dataImporter.GetOutput().GetExtent()

    camera = backgr.GetActiveCamera()
    camera.ParallelProjectionOn()

    xc = origin[0] + 0.5 * (extent[0] + extent[1]) * spacing[0]
    yc = origin[1] + 0.5 * (extent[2] + extent[3]) * spacing[1]
    # xd = (extent[1] - extent[0] + 1) * spacing[0]
    yd = (extent[3] - extent[2] + 1) * spacing[1]
    d = camera.GetDistance()
    camera.SetParallelScale(0.5 * yd)
    camera.SetFocalPoint(xc, yc, 0.0)
    camera.SetPosition(xc, yc, d)

    # Render again to set the correct view
    backgr.Render()


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required=True, help="Select the folder to access.")

    args = vars(ap.parse_args())
    folder = args["path"]

    #creo gli attori leggendo i file obj
    load_models_and_create_actors()

    # parte di creazione finestra vtk
    # due layers, scena e background, sono due renderer necessari per buttarci dentro l'immagine
    renWin.SetNumberOfLayers(2)
    renWin.AddRenderer(backgr)
    renWin.AddRenderer(sceneRen)
    # si crea la finestra, cioè il contenitore, invece il renderer è il buffer
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    renWin.SetSize(dim)
    renWin.SetWindowName('VideoTest')

    # aggiunta actor, oggetti che verrano rendirizzati sul renderer
    imageActor = vtk.vtkImageActor()
    dataImporter = vtk.vtkImageImport()  # per acquisire l'immagine
    dataImporter.SetWholeExtent(0, dim[0] - 1, 0, dim[1] - 1, 0, 0)  # definisco la grandezza del contenitore
    imageActor.SetInputData(dataImporter.GetOutput())  # image actor dev'essere collegato al data importer
    backgr.AddActor(imageActor)  # lo aggiungiamo al layer background

    # iren.AddObserver("KeyPressEvent", key_pressed_callback)
    # iren è l'oggetto gestione comandi, faccio un'overwrite rispetto al default
    iren.SetInteractorStyle(MyInteractorStyle())

    #################### il layer di backgound è flippato x via della direzione Y tra texture e 3D
    # se dovessero esserci errori, l'altro modo è:
    # vtkimg = cv2.flip(vtkimg, 0) #altrimenti l'immagine risulta capolvolta (solita question del Y-flip)
    # backgr.GetActiveCamera().SetViewUp(0, -1, 0)

    # aggiungo degli assi di riferimento
    refaxes = MakeAxesActor()
    om = vtk.vtkOrientationMarkerWidget()
    om.SetOrientationMarker(refaxes)
    om.SetInteractor(iren)
    om.EnabledOn()
    om.InteractiveOn()

    SetTextWidgets()

    # inizializzazione interactor
    iren.Initialize()
    # iren.Start() #non serve perchè abbiamo il nostro ciclo
    sceneRen.GetActiveCamera().SetViewAngle(viewangle)

    fov_text_actor.SetInput("FOV: " + str(viewangle))

    #fine parte di vtk
    # VARIABLES
    len_buf = 150
    buf = CircularBuffer(len_buf)
    weight_with_Xprevious = 0
    weight_gradual = 0
    inters_pos = "Flat"
    frame_interval = 1
    angle_between = 0.0
    dim = (960, 540)
    z_angle = 0

    w_gradual = False
    using_avg_lines = False
    save_video = "N"
    plot_3d = "N"
    frame_interval_for_3d = 1
    plot_main_lines = "N"
    plot_center = "N"
    plot_apex = "N"
    plot_all_lines = "N"
    plot_intersection = "N"
    plot_contour = "N"
    plot_ellipse = "N"
    show_original = "N"
    show_segmentation = "N"
    show_contour = "N"
    show_output = "N"

    # First we select the model to use, best results were obtained with mobilnet, so run the program with "-m 0"
    model = model_from_checkpoint_path("Checkpoints\\mobilenet_alltools")

    angleFilter = [0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0]

    SetBoxWidget()
    tform = vtk.vtkTransform()

    base_path = os.path.join("C:\\Users\\Leonardo\\Desktop\\DatasetToTag", folder)

    create_excel(folder)

    # while in modo che passi all'immagine successiva solo quando settano il modello 3d

    all_images = []
    all_label = []

    for img in tqdm(os.listdir(base_path)):
        if img.endswith(".png") or img.endswith(".jpg"):
            image = cv2.imread(os.path.join(os.path.join(base_path, img)))
            all_images.append(image)
            all_label.append(img)

    while True:

        if i == len(all_images):
            break
        if finish:
            break

        name_image = all_label[i]
        frame = cv2.resize(all_images[i], dim, interpolation=cv2.INTER_LINEAR)

        vtkimg = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #necessario perchè opencv ha le immagini in BGR, e vtk no.
        vtkimg = cv2.flip(vtkimg, 0)

        # Out è il risultato della segmentazione
        out = model.predict_segmentation(inp=frame)
        img = convert_np_to_mat(out)

        # Salvo l'output della segmentazione per poterla plottare
        img_segmented = img

        (channel_b, channel_g, channel_r) = cv2.split(img)  # ciascuna con un solo canale

        if (autoRotateX):
            STARTING_BETA = 0
            # parte alternativa per il calcolo della Xrot basata sulla quantità di pixel trovati...
            # entra nel ciclo a richiesta
            angleratio = getAngleFromGreenPixels(channel_g)
            angleFilter[frameNumber] = angleratio
            angleratiomean = np.mean(angleFilter)

            beta = STARTING_BETA - (angleratiomean - 1) * 0.525  # circa 30 gradi
            xrot_text_actor.SetInput("X Rot.: " + str(round(beta, 3)))

            STARTING_BETA = beta

            print("Calibrating X rot: " + str(round(100 * frameNumber / 24, 0)))

            if (frameNumber >= 24):
                autoRotateX = False

            frameNumber += 1
        else:
            beta = STARTING_BETA

        image_b, contours_b, hierarchy_b = cv2.findContours(channel_b.astype('uint8'), cv2.RETR_TREE,
                                                            cv2.CHAIN_APPROX_SIMPLE)
        image_g, contours_g, hierarchy_g = cv2.findContours(channel_g.astype('uint8'), cv2.RETR_TREE,
                                                            cv2.CHAIN_APPROX_SIMPLE)

        # copio il frame dato che opencv va a disegnare sull'originale
        output = frame.copy()

        # qua lavoro con i contorni del catetere
        if len(contours_g) != 0:

            # seleziono il contorno con l'area maggiore
            max_c = max(contours_g, key=cv2.contourArea)
            # Trovo la convex hull del contorno massimo
            hull = cv2.convexHull(max_c)

            # disegno la convex hull in un'immagine bianca, è un po stupido ma funziona
            img = np.zeros([540, 960, 3], dtype=np.uint8)
            img.fill(0)
            cv2.drawContours(img, [hull], 0, (0, 0, 150), 2)

            # trovo i quattro estremi e il centro
            ext_left = tuple(max_c[max_c[:, :, 0].argmin()][0])
            ext_right = tuple(max_c[max_c[:, :, 0].argmax()][0])
            ext_top = tuple(max_c[max_c[:, :, 1].argmin()][0])
            ext_bot = tuple(max_c[max_c[:, :, 1].argmax()][0])
            M = cv2.moments(max_c)
            cX = int(M["m10"] / M["m00"] + 0.0000001)
            cY = int(M["m01"] / M["m00"] + 0.0000001)

            img_contours = img

            # se il contorno almeno 4 vertici
            if len(max_c) > 5:
                # trovo il rettangolo rotato che definisce un'ellisse
                rot_rect = cv2.fitEllipse(max_c)

                # trovo il punto di apice e lo disegno
                box = cv2.boxPoints(rot_rect)
                box = np.int0(box)

                apex_point = apex(box)

                #parte per gli aggiustamenti dell'aggancio (con il workaround brutto per modificare la tupla)
                directionVector = findVec([cX, cY], apex_point, True)

                # cv2.arrowedLine(output, (cX, cY), apex_point, (0, 255, 0), 10)

                apex_point_l = list(apex_point)
                apex_point_l[0] += directionVector[0] * slide
                apex_point_l[1] += directionVector[1] * slide
                apex_point = apex_point_l

                apex_point = np.int(apex_point[0] * 0.4 + lastframeapex[0] * 0.6), np.int(apex_point[1] * 0.4 + lastframeapex[1] * 0.6) #con lo smooth nel movimento
                lastframeapex = apex_point

                cv2.circle(vtkimg, apex_point, 2, (255, 0, 0), thickness=2)

                z_angle = AngleBetween([cX, cY], apex_point) #centroide --> apice
                z_angle = z_angle * 0.4 + lastframeaZangle * 0.6
                lastframeaZangle = z_angle

                newLenght = sqrt((box[0][0] - box[3][0]) ** 2 + (box[0][1] - box[3][1]) ** 2)

                # Calcola il valore distance in automatico
                # Usando la formula: distance' = distance * oldLenght / newLenght
                distance = STARTING_SIZE * (registeredLenght / newLenght) * 0.4 + lastframedistance * 0.6
                lastframedistance = distance

                apex3d = Get3dWorldPointFrom2dPoint(apex_point)
                world_matrix = CreateWorldMatrix(apex3d, z_angle)

                if not freezed:
                    SetWorldMatrix(world_matrix)
                    # la parte sotto riguarda il transform box
                    tform.SetMatrix(world_matrix)
                    boxWidget.SetTransform(tform)


                # Trovo l'altezza e la larghezza del catetere, serve per fare lo scalamento su vtk
                #height = sqrt((box[0][0] - box[1][0]) ** 2 + (box[0][1] - box[1][1]) ** 2)
                #width = sqrt((box[0][0] - box[3][0]) ** 2 + (box[0][1] - box[3][1]) ** 2)

            # plotto il frame originale, la segmentazione, il contorno e l'output finale
            MatToVtKImageRender(vtkimg)

            # con questa funziona sovrappongo il modello 3D, però procede al frame successivo solo se premo 'q'
            #overlay_catheter(z_angle, x_angle, name_path, (cX, cY), height, width, i)

        if next_step == True:
            i += 1
        next_step = False

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        # i += 1

    cv2.destroyAllWindows()
