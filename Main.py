from __future__ import print_function

import sys

from keras_segmentation.predict import predict, model_from_checkpoint_path
import argparse
from math import sin, cos, sqrt
import shutil
import os
from myutils import *
from myutils3D import *
from os import walk
from vtk.vtkIOKitPython import vtkOBJReader
from vtk.vtkRenderingKitPython import vtkPolyDataMapper, vtkActor
import numpy as np

class_colors: List[Tuple[int, int, int]] = [(0, 0, 0), (255, 0, 0), (0, 255, 0)]

# qualche global
obj_path = "Objects\\"
dim = (960, 540)  # dimensioni finestra

freezed = False
slide = 0
viewangle = 0

modelfiles = []
modelactors = list()
modelopacity = list()
modelmappers = list()
modelnames  = list() #qui ci sono i soli nomi delle mesh. L'indice del nome corrisponde all'indice dell'actor.

sceneRen = vtk.vtkRenderer()  # questo per il catetere e la prostata 3D
sceneRen.SetLayer(1)
backgr = vtk.vtkRenderer()  # questo per l'immagine di sfondo
backgr.SetLayer(0)

#scala automatica del catetere
STARTING_SIZE = 120.0
distance = STARTING_SIZE #distanza della prostata.
STARTING_BETA = 0.0
beta = STARTING_BETA #rotazione della prostata antero-posteriore
registeredLenght = 80.0 # lunghezza in pixel del catetere... viene modificata a mano coi tasti + e -
newLenght = 1.0 # lunghezza della retta che interseca il punto d'apice (segmento superiore del bbox)

#per le operazioni di filtraggio
lastframeapex = [0,0]
lastframedistance = 0
lastframeaZangle = 0
lastframeaXangle = 0

#GUI globals
fov_text_actor = vtk.vtkTextActor()
size_text_actor = vtk.vtkTextActor()
xrot_text_actor = vtk.vtkTextActor()
slide_text_actor = vtk.vtkTextActor()


class MyInteractorStyle(vtk.vtkInteractorStyle):

    def __init__(self,parent=None):

        self.parent = iren
        self.AddObserver("KeyPressEvent", self.keyPressEvent)
        self.AddObserver("CharEvent", self.charEvent)


    def keyPressEvent(self, obj, event):
        global STARTING_SIZE, beta, sceneRen
        global freezed, slide, viewangle
        global fov_text_actor, size_text_actor, xrot_text_actor, slide_text_actor

        key = self.parent.GetKeySym()
        if key == 'Shift_L':
            SwitchAllMeshOpacity()
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
            beta -= 0.02
            xrot_text_actor.SetInput("X Rot.: " + str(round(beta, 3)))
        elif key == 'm':
            xrot_text_actor.SetInput("X Rot.: " + str(round(beta, 3)))
            beta += 0.02
        elif key == 'x':
            freezed = not freezed
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

        print(key)
        return

    def charEvent(self, obj, event): #necessario per bloccare gli eventi automatici della finestra
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
    #recupera l'indice dell'actor della mesh con questo nome

    for i,n in enumerate(modelnames):
        if(n[0:4] == name):
            SwitchSingleMeshOpacity(i) #fatto in questo modo dovrebbe funzionare per tutte le mesh che iniziano con lo stesso nome


def SetAllMeshOpacity(value):
    for a in modelactors:
        a.GetProperty().SetOpacity(value)


def SwitchAllMeshOpacity():
    for i in range(len(modelactors)):
        if(modelopacity[i]==1):
            SetAllMeshOpacity(0.5)
            modelopacity[i]=0.5
        elif (modelopacity[i] == 0.5):
            SetAllMeshOpacity(0)
            modelopacity[i]=0
        elif (modelopacity[i] == 0):
            SetAllMeshOpacity(1)
            modelopacity[i]=1


def Get3dWorldPointFrom2dPoint(point2d):
    #retMat = np.eye(4) #matrice 4x4 con gli 1 in diagonale
    coord = vtk.vtkCoordinate()
    coord.SetCoordinateSystemToViewport()
    coord.SetValue(point2d[0], dim[1] - point2d[1]) #correggo l'orientamento del punto sull'immagine per l'asse Y
    cx, cy, cz = coord.GetComputedWorldValue(sceneRen)

    origin = sceneRen.GetActiveCamera().GetPosition()
    #direction = findVec(origin, [cx, cy, cz])
    direction = np.array(findVec(origin, [cx, cy, cz]))

    normalized_direction = direction / np.sqrt(np.sum(direction ** 2))

    vx = origin[0] + normalized_direction[0] * distance
    vy = origin[1] + normalized_direction[1] * distance
    vz = origin[2] + normalized_direction[2] * distance

    #print("origine camera   : " + str(origin[0]) + ", " + str(origin[1]) + ", " + str(origin[2]))
    #print("vettore direzione: " + str(normalized_direction[0]) + ", " + str(normalized_direction[1]) + ", " + str(normalized_direction[2]))
    #print("valori finali    : " + str(vx) + ", " + str(vy) + ", " + str(vz))

    return [vx, vy, vz]


#effettua un ciclo su ogni actor (mesh) e imposta la sua world matrix
def SetWorldMatrix(apx):
    for a in modelactors:
        #a.SetPosition(apx)
        a.SetUserMatrix(apx)


#dato un punto e un angolo X + un angolo Y crea la matrice di roto/traslazione
def CreateWorldMatrix(apex3d, angleZ):
    t = vtk.vtkMatrix4x4()
    rz = vtk.vtkMatrix4x4()
    #ry = vtk.vtkMatrix4x4()
    rx = vtk.vtkMatrix4x4()
    w = vtk.vtkMatrix4x4()

    #traslazione
    t.SetElement(0, 3, apex3d[0])
    t.SetElement(1, 3, apex3d[1])
    t.SetElement(2, 3, apex3d[2])

    #setting Z rotation
    rz.SetElement(0, 0, cos(angleZ))
    rz.SetElement(1, 1, cos(angleZ))
    rz.SetElement(0, 1, -sin(angleZ))
    rz.SetElement(1, 0, sin(angleZ))

    #rotazione y (+180 se mesh caricata storta)
    #ry.SetElement(0, 0, cos(np.pi))
    #ry.SetElement(0, 2, sin(np.pi))
    #ry.SetElement(2, 0, -sin(np.pi))
    #ry.SetElement(2, 2, cos(np.pi))

    #rotazione x
    rx.SetElement(1, 1, cos(beta))
    rx.SetElement(1, 2, -sin(beta))
    rx.SetElement(2, 1, sin(beta))
    rx.SetElement(2, 2, cos(beta))


    #setting X rotation w = t * rz * ry * rx
    vtk.vtkMatrix4x4().Multiply4x4(t, rz, w)
    vtk.vtkMatrix4x4().Multiply4x4(w, rx, w)
    #vtk.vtkMatrix4x4().Multiply4x4(w, ry, w)
    return w


def SetMeshColor(name):
    nc = vtk.vtkNamedColors()

    if name[0:4].lower() == "eles":
        color = nc.GetColor3d("Lime") #verde
    if name[0:4].lower() == "iles":
        color = nc.GetColor3d("DarkGreen")#verde scuro
    if name[0:4].lower() == "lfas":
        color = nc.GetColor3d("Blue") #blu
    if name[0:4].lower() == "rfas":
        color = nc.GetColor3d("Blue") #blu
    if name[0:4].lower() == "inte":
        color = nc.GetColor3d("Black")   #nero
    if name[0:4].lower() == "porz":
        color = nc.GetColor3d("Orange") #arancione
    if name[0:4].lower() == "pros":
        color = nc.GetColor3d("Gray") # grigio
    if name[0:4].lower() == "sfin":
        color = nc.GetColor3d("DarkOrange") #arancione scuro
    if name[0:4].lower() == "uret":
        color = nc.GetColor3d("Yellow") # giallo
    return color


# funzione che carica i file diversi file della prostata e li aggiunge al latey di renderizzazione
def load_models_and_create_actors():
    for (dirpath, dirnames, filenames) in walk(obj_path):
        modelfiles = [fi for fi in filenames if fi.endswith(".obj")]
        break

    idx = 0
    for f in modelfiles: #ogni mesh ha bisogno del suo actor e mapper
        print("Importing "+f)
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

    sceneRen.AddActor(fov_text_actor)
    sceneRen.AddActor(size_text_actor)
    sceneRen.AddActor(xrot_text_actor)
    sceneRen.AddActor(slide_text_actor)

    # Create the text representation. Used for positioning the text_actor
    text_representation1 = vtk.vtkTextRepresentation()
    text_representation1.GetPositionCoordinate().SetValue(0.01, 0.85)

    text_representation2 = vtk.vtkTextRepresentation()
    text_representation2.GetPositionCoordinate().SetValue(0.01, 0.75)

    text_representation3 = vtk.vtkTextRepresentation()
    text_representation3.GetPositionCoordinate().SetValue(0.01, 0.65)

    text_representation4 = vtk.vtkTextRepresentation()
    text_representation4.GetPositionCoordinate().SetValue(0.01, 0.55)

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
    ap.add_argument("-m", "--model", required=True, help="Select the model to use "
                                                         "(0 for MobileNet, 1 for VGG, 2 for ResNet, 3 for Unet")
    ap.add_argument("-v", "--video_source", required=True, help="Video source (0 for cam, 1-9 video input or filename)")

    args = vars(ap.parse_args())
    run_model = int(args["model"])
    models = ["MobileNet", "VGG", "ResNet", "U-Net"]

    video_source = 0
    if type(args["video_source"]) == int:
        video_source = int(args["video_source"])
    else:
        video_source = str(args["video_source"])

    print("Accessing video source:" + video_source)

    #creo gli attori leggendo i file obj
    load_models_and_create_actors()

    #parte di creazione finestra vtk

    renWin = vtk.vtkRenderWindow()
    renWin.SetNumberOfLayers(2)
    renWin.AddRenderer(backgr)
    renWin.AddRenderer(sceneRen)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    renWin.SetSize(dim)
    renWin.SetWindowName('VideoTest')

    imageActor = vtk.vtkImageActor()
    dataImporter = vtk.vtkImageImport()
    dataImporter.SetWholeExtent(0, dim[0]-1, 0, dim[1]-1, 0, 0)
    imageActor.SetInputData(dataImporter.GetOutput())
    backgr.AddActor(imageActor)

    #iren.AddObserver("KeyPressEvent", key_pressed_callback)
    iren.SetInteractorStyle(MyInteractorStyle())

    #################### il layer di backgound è flippato x via della direzione Y tra texture e 3D
    # se dovessero esserci errori, l'altro modo è:
    # vtkimg = cv2.flip(vtkimg, 0) #altrimenti l'immagine risulta capolvolta (solita question del Y-flip)
    #backgr.GetActiveCamera().SetViewUp(0, -1, 0)

    #aggiungo degli assi di riferimento
    refaxes = MakeAxesActor()
    om = vtk.vtkOrientationMarkerWidget()
    om.SetOrientationMarker(refaxes)
    om.SetInteractor(iren)
    om.EnabledOn()
    om.InteractiveOn()

    SetTextWidgets()

    iren.Initialize()
    #iren.Start() #non serve perchè abbiamo il nostro ciclo

    viewangle = sceneRen.GetActiveCamera().GetViewAngle()
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
    plot_main_lines = "Y"
    plot_center = "N"
    plot_apex = "N"
    plot_all_lines = "N"
    plot_intersection = "N"
    plot_contour = "N"
    plot_ellipse = "N"
    show_original = "N"
    show_segmentation = "N"
    show_contour = "N"
    show_output = "Y"

    # First we select the model to use, best results were obtained with mobilnet, so run the program with "-m 0"
    if run_model == 0:
        model = model_from_checkpoint_path("Checkpoints\\mobilenet_alltools")
    elif run_model == 1:
        model = model_from_checkpoint_path("..\\Checkpoints\\NewTool\\new_vgg_unet_tool")
    elif run_model == 2:
        model = model_from_checkpoint_path("..\\Checkpoints\\NewTool\\new_resnet_unet_tool")
    elif run_model == 3:
        model = model_from_checkpoint_path("..\\Checkpoints\\NewTool\\new_unet_tool")

    cap = cv2.VideoCapture(video_source)
    out_vid = cv2.VideoWriter('..\\OutputVideo\\Degree.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 3, (960, 540))

    # Uso il contatore perchè servono almeno 100 frames per permettere a beta di stabilizzarsi
    i = 0

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        if video_source.split("\\")[-1] == "New.MP4":
            frame = frame[20:20 + 600, int(1920 / 2) - int(1066 / 2): int(1920 / 2) + int(1066 / 2)]

        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_LINEAR)

        vtkimg = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #necessario perchè opencv ha le immagini in BGR, e vtk no.
        vtkimg = cv2.flip(vtkimg, 0)

        # Out è il risultato della segmentazione
        out = model.predict_segmentation(inp=frame)
        img = convert_np_to_mat(out)

        # Questa parte è commentata perchè migliora le prestazione ma rallenta molto. Si compie un'erosione,
        # seguita da una dilatazione e una successiva erosione per eliminare i punti sparsi
        # kernel = np.ones((11, 11), np.uint8)
        # kernel2 = np.ones((9, 9), np.uint8)
        # n_it = 3
        # img_first_erosion = cv2.erode(img, kernel, iterations=1)
        # img_dilated = cv2.dilate(img_first_erosion, kernel, iterations=n_it)
        # img = cv2.erode(img_dilated, kernel, iterations=n_it)

        # Salvo l'output della segmentazione per poterla plottare
        img_segmented = img

        (channel_b, channel_g, channel_r) = cv2.split(img)  # ciascuna con un solo canale

        # trovo i contorni per la segmentazione del catetere e degli strumenti
        contours_b, hierarchy_b = cv2.findContours(channel_b.astype('uint8'), cv2.RETR_TREE,
                                                   cv2.CHAIN_APPROX_SIMPLE)
        contours_g, hierarchy_g = cv2.findContours(channel_g.astype('uint8'), cv2.RETR_TREE,
                                                   cv2.CHAIN_APPROX_SIMPLE)

        # copio il frame dato che opencv va a disegnare sull'originale
        output = frame.copy()

        # qua andrei a lavorare con i contorni degli strumenti, ho commentato quasi tutto perchè ora non serve
        #if len(contours_b) != 0:
            # cv2.drawContours(output, contours_b, -1, (125,125,125), 1)
            # find the biggest countour (c) by the area
            # c = max(contours_b, key=cv2.contourArea)
            # x, y, w, h = cv2.boundingRect(c)

        # draw the biggest contour (c) in blue
        # cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)

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

            if plot_center == "Y":
                cv2.circle(output, (cX, cY), 7, (0, 0, 0), -1)

            # converto e faccio un dilatazione per rendere i contorni più spessi in modo che sia più facile
            # individuare le linee da HoughLines
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kernel_dilate = np.ones((7, 7), np.uint8)
            img = cv2.dilate(img, kernel_dilate, iterations=1)
            img_contours = img

            minLineLength = 20  # 50
            maxLineGap = 5  # 0
            lines = cv2.HoughLinesP(img, 1, np.pi / 180, 100, minLineLength, maxLineGap)

            if plot_all_lines == "Y":
                x = output.copy()
                if lines is not None:
                    for line in lines:
                        for x1, y1, x2, y2 in line:  # prendo i valori dei punti linea per linea
                            cv2.line(x, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.imshow("All Lines", x)
                            cv2.waitKey()

            if using_avg_lines:
                x = output.copy()
                lines_per_side = 3
                angle_between = 0
                threshold_min = 10
                threshold_max = 30
                m2 = None
                if lines is not None:  # se HL restituisce almeno una linea
                    n_line = 0
                    first_set = 0
                    second_set = 0
                    for line in lines:  # itero fra le linee
                        if first_set < lines_per_side or second_set < lines_per_side:
                            for x1, y1, x2, y2 in line:  # prendo i valori dei punti linea per linea
                                # cv2.line(x, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                y1 = 540 - y1
                                y2 = 540 - y2
                                #vcv2.imshow("All Lines", x)
                                # cv2.waitKey()
                                if n_line == 0:  # se è la prima la costruisco e la uso come riferimento, salvando i punti
                                    n_line += 1
                                    x1_line1 = x1
                                    x2_line1 = x2
                                    y1_line1 = y1
                                    y2_line1 = y2
                                    m1 = (y2 - y1) / (x2 - x1 + 0.00000001)
                                    first_set += 1
                                else:  # costruisco le successive linee finchè non ne trovo una ad una distanza
                                    # maggiore di 20, perchè spesso le prime linee trovate sono coincidenti
                                    # l_tmp = build_line([x1, y1], [x2, y2])
                                    avg_y = abs((abs(y2_line1 - y1_line1) - abs(y2 - 1))) / 2  #il punto medio delle y di ciascuna retta diviso 2
                                    # funziona che restituisce la distanza data un y fissa
                                    dist = abs(obtain_universalx(x1_line1, y1_line1, x2_line1, y2_line1, avg_y) -
                                               obtain_universalx(x1, y1, x2, y2, avg_y))
                                    if dist < threshold_min and first_set < lines_per_side:
                                        # faccio la media dei coefficenti angolari
                                        m1 = (m1 + ((y2 - y1) / (x2 - x1 + 0.00000001))) / 2
                                        first_set += 1
                                    elif dist > threshold_max and second_set < lines_per_side:
                                        second_set += 1
                                        if m2 is None:
                                            m2 = (y2 - y1) / (x2 - x1 + 0.00000001)
                                            # li salvo per poi ricostruire la linea
                                            x1_line2 = x1
                                            y1_line2 = y1
                                        else:
                                            m2 = (m2 + ((y2 - y1) / (x2 - x1 + 0.00000001))) / 2
                        else:
                            break

                    # calcolo l'angolo fra i due coefficenti e lo converto in gradi
                    # a volte trova solo uno dei due set, se non metto questo da errore
                    if m2 is None:
                        m2 = m1
                    print("m1 {} m2 {}".format(m1,m2))
                    angle_between = compute_angle_given_coeff(m1, m2)
                    angle_between = np.degrees(angle_between)
                    if angle_between > 90:
                        angle_between = abs(angle_between - 180)
                    # trovo due rette generiche con coefficente m1 e m2
                    p1, p2 = find_line_from_coef(m1, x1_line1, y1_line1)
                    p3, p4 = find_line_from_coef(m2, x1_line2, y1_line2)
                    cv2.line(x, p1, p2, (0, 0, 255), 2)
                    cv2.line(x, p3, p4, (0, 0, 255), 2)
                    cv2.imshow("All Lines", x)
                    # cv2.waitKey()


            # metodo senza media di linea, prende solo le prime due per lato
            else:
                if lines is not None:  # se restituisce almeno una linea
                    n_line = 0
                    for line in lines:  # itero fra le linee
                        for x1, y1, x2, y2 in line:  # prendo i valori dei punti linea per linea
                            if n_line == 0:  # se è la prima la costruisco e la uso come riferimento, salvando i punti
                                n_line += 1
                                l1 = build_line([x1, y1], [x2, y2])
                                x1_line1 = x1
                                x2_line1 = x2
                                y1_line1 = y1
                                y2_line1 = y2
                            if n_line > 0:  # costruisco le successive linee finchè non ne trovo una ad una distanza
                                            # maggiore di 20, perchè spesso le prime linee trovate sono coincidenti
                                l_tmp = build_line([x1, y1], [x2, y2])
                                dist1 = abs(x2 - x2_line1)
                                dist2 = abs(x1 - x1_line1)
                                if dist1 > 30 or dist2 > 30:
                                    l2 = l_tmp
                                    n_line += 1
                                    if plot_main_lines == "Y":
                                        cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                        cv2.line(output, (x1_line1, y1_line1), (x2_line1, y2_line1), (0, 255, 0), 2)
                                    break
                        if n_line == 2:
                            # trovo l'angolo tra le due linee
                            angle_between = compute_angle(x1_line1, y1_line1, x2_line1, y2_line1, x1, y1, x2, y2)
                            if angle_between > 90:
                                angle_between = abs(angle_between - 180)

                            # ridimensiono i valori in modo da plottare il punto di intersez all'interno della finestra
                            if plot_intersection == "Y":
                                inter_x, inter_y = intersection(l1, l2)
                                if inter_y < 0:
                                    inter_y = 0
                                if inter_y > 540:
                                    inter_y = 540
                                if inter_x < 0:
                                    inter_x = 0
                                if inter_x > 960:
                                    inter_x = 960
                                cv2.circle(output, (inter_x, inter_y), 10, (255, 0, 0), thickness=10)

                                if inter_y > 540 / 2:
                                    inters_pos = "Down"
                                elif inter_y <= 540 / 2:
                                    inters_pos = "Up"

                            break

            avg_buf = buf.avg()
            # trova l'angolo medio con due tecniche di pesatura, la prima calcola la media nel buffer di cui si puo
            # settare la lunghezza, nel secondo è il metodo di Pietro che prende sempre quello prima
            if avg_buf != 0:
                weight_with_Xprevious = (avg_buf * 0.9) + (angle_between * 0.1)
                weight_gradual = (previous * 0.9) + (angle_between * 0.1)
                previous = weight_gradual
                cv2.putText(output, "wgr: {:.8}".format(weight_gradual), (800, 300), cv2.FONT_HERSHEY_COMPLEX,
                            0.5,
                            (255, 255, 255), 1)
                cv2.putText(output, "wX: {:.8}".format(weight_with_Xprevious), (800, 400), cv2.FONT_HERSHEY_COMPLEX,
                            0.5,
                            (255, 255, 255), 1)
            else:
                previous = angle_between
            buf.add_to_buf(angle_between)
            # aspetto almeno 100 frame per riempire il buffer e per bilanciare il valore del peso
            beta = 0
            if i > len_buf:
                if w_gradual is True and 8 > weight_gradual > 4:
                    beta = 0.5
                elif w_gradual is False and 8 > weight_with_Xprevious > 4:
                    beta = 0.5
            print(beta)


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

                if plot_apex == "Y":
                    cv2.circle(output, (int(apex_point[0]), int(apex_point[1])), 2, (255, 0, 0), thickness=2)
                if plot_ellipse == "Y":
                    cv2.drawContours(output, [box], 0, (255, 255, 255), 2)
                #fine

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


                # Trovo l'altezza e la larghezza del catetere, serve per fare lo scalamento su vtk
                #height = sqrt((box[0][0] - box[1][0]) ** 2 + (box[0][1] - box[1][1]) ** 2)
                #width = sqrt((box[0][0] - box[3][0]) ** 2 + (box[0][1] - box[3][1]) ** 2)

            # plotto il frame originale, la segmentazione, il contorno e l'output finale
            MatToVtKImageRender(vtkimg)

            if show_original == "Y":
                cv2.imshow("Frame", frame)
            if show_segmentation == "Y":
                cv2.imshow("Segmentation", img_segmented)
            if show_contour == "Y":
                cv2.imshow("Contours", img_contours)

            if show_output == "Y":
                cv2.putText(output, "Frame {}".format(i), (800, 100), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(output, "a: {:.8}".format(float(angle_between)), (800, 200), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                            (255, 255, 255), 1)
                if 8 > weight_gradual > 4:
                    cv2.circle(output, (850, 330), 10, (0, 255, 0), thickness=20)
                if 8 > weight_with_Xprevious > 4:
                    cv2.circle(output, (850, 430), 10, (255, 0, 0), thickness=20)
                cv2.imshow("Output", output)

            if save_video == "Y":
                out_vid.write(output)

            # con questa funziona sovrappongo il modello 3D, però procede al frame successivo solo se premo 'q'
            #overlay_catheter(z_angle, x_angle, name_path, (cX, cY), height, width, i)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        i += 1
    cap.release()
    out_vid.release()
    cv2.destroyAllWindows()
