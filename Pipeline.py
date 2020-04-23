from __future__ import print_function
from keras_segmentation.predict import predict, model_from_checkpoint_path
import argparse
from math import sin, cos, sqrt
from myutils import *
from myutils3D import *

class_colors: List[Tuple[int, int, int]] = [(0, 0, 0), (255, 0, 0), (0, 255, 0)]

# gli angoli servono per la rotazione, il punto di centrale per la posizione (più semplice con questo metodo che col
# punto di apice, 'h' e 'w' sono le dimensioni originali del catetere per fare lo scalamento, 'path' serve per leggere le
# le immagini da importare, 'i' serve se si vogliono salvare i png e dare nomi diversi
def overlay_catheter(z_angle, x_angle, path, point, h, w, i):
    # leggo il frame su cui plotterò il catetere
    reader_img = vtkPNGReader()
    reader_img.SetFileName(path)
    reader_img.Update()
    image_data = reader_img.GetOutput()
    size = image_data.GetExtent()
    image_geometry_filter = vtkImageDataGeometryFilter()
    image_geometry_filter.SetInputConnection(reader_img.GetOutputPort())
    image_geometry_filter.Update()

    # associo un mapper
    mapper_img = vtkPolyDataMapper()
    mapper_img.SetInputConnection(image_geometry_filter.GetOutputPort())

    # creo l'actor corrispondente all'immagine
    actor_img = vtkActor()
    actor_img.SetMapper(mapper_img)

    # leggo il modello 3D
    path_obj = "..\\Germano\\Oriented\\CATETHER.obj"
    reader_obj = vtkOBJReader()
    reader_obj.SetFileName(path_obj)

    # associo un kapper
    mapper_obj = vtkPolyDataMapper()
    mapper_obj.SetInputConnection(reader_obj.GetOutputPort())

    # creo l'actor corrispondente al catetere
    actor_obj = vtkActor()
    actor_obj.SetMapper(mapper_obj)
    # prendo i punti di confine del modello 3D in modo da sapere quanto scalare altezza e larghezza
    bounds = actor_obj.GetBounds()
    original_h = abs(bounds[3] - bounds[2])
    original_w = abs(bounds[1] - bounds[0])
    # proietto il punto fi aggancio (centrale in questo caso) in 3D e sposto il modello
    point_3D = project_point_plane(point)
    actor_obj.SetPosition(point_3D[0], point_3D[1], point_3D[2])
    # scalo il modello
    sy = h / original_h
    sx = w / original_w
    actor_obj.SetScale(sx, sy, sx)
    # ruoto il modello
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

    render_window = vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(1000, 1000)

    render_window_interactor = vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)
    render_window_interactor.Initialize()
    render_window_interactor.Start()

    # se voglio esportare come png
    # get_screenshot(render_window, "output{}.png.format(i)", 1)


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True, help="Select the model to use "
                                                         "(0 for MobileNet, 1 for VGG, 2 for ResNet, 3 for Unet")
    ap.add_argument("-v", "--video", required=True, help="Path to the video.")
    args = vars(ap.parse_args())
    run_model = int(args["model"])
    video_file_path = args["video"]
    models = ["MobileNet", "VGG", "ResNet", "U-Net"]
    print("Running the {} model.".format(models[run_model]))

    # video_file_path = "..\\RealTime Video\\CV_2_Cropped.mp4"

    # First we select the model to use, best results were obtained with mobilnet, so run the program with "-m 0"
    if run_model == 0:
        model = model_from_checkpoint_path("..\\Checkpoints\\NewTool\\new_mobilenet_unet_tool")
    elif run_model == 1:
        model = model_from_checkpoint_path("..\\Checkpoints\\NewTool\\new_vgg_unet_tool")
    elif run_model == 2:
        model = model_from_checkpoint_path("..\\Checkpoints\\NewTool\\new_resnet_unet_tool")
    elif run_model == 3:
        model = model_from_checkpoint_path("..\\Checkpoints\\NewTool\\new_unet_tool")

    cap = cv2.VideoCapture(video_file_path)

    i = 0

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break
        # In questo modo si elabora 1 frame ogni 3, per ottimizzARE il tutto cap.read() non dovrebbe leggere tutti i
        # frame ma solamente quelli che andiamo poi ad elaborare
        frame_interval = 3
        if (i % frame_interval) == 0:

            dim = (960, 540)

            frame = cv2.resize(frame, dim, interpolation=cv2.INTER_LINEAR)

            # La cartella name_path serve per VTK quando va a leggere le immagini e le carica come oggetti
            name_path = "../OutputTMP/{}.png".format(i)
            cv2.imwrite(name_path, frame)

            # x_angle provvisoriamente può avere tre valori, 0, 1 se l'intersezione è sopra il catetere e -1 se è sotto
            x_angle = 0
            z_angle = 0

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

            # Salvo l'output della segmentazioneper poterla plottare
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
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.circle(output, (cX, cY), 7, (0, 0, 0), -1)

                # converto e faccio un dilatazione per rendere i contorni più spessi in modo che sia più facile
                # individuare le linee da HoughLines
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                kernel_dilate = np.ones((7, 7), np.uint8)
                img = cv2.dilate(img, kernel_dilate, iterations=1)
                img_contours = img

                minLineLength = 50
                maxLineGap = 0
                lines = cv2.HoughLinesP(img, 1, np.pi / 180, 100, minLineLength, maxLineGap)

                if lines is not None:  # se HL restituisce almeno una linea
                    n_line = 0
                    for line in lines:  # itero fra le linee
                        for x1, y1, x2, y2 in line:  # prendo i valori dei punti linea per linea
                            if n_line == 0:  # se è la prima la costruisco e la uso come riferimento, salvando i punti
                                n_line += 1
                                l1 = build_line([x1, y1], [x2, y2])
                                cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                x1_line1 = x1
                                x2_line1 = x2
                                y1_line1 = y1
                                y2_line1 = y2
                            if n_line > 0:  # costruisco le successive linee finchè non ne trovo una ad una distanza
                                            # maggiore di 20, perchè spesso le prime linee trovate sono coincidenti
                                l_tmp = build_line([x1, y1], [x2, y2])
                                dist1 = abs(x2 - x2_line1)
                                if dist1 > 20:
                                    l2 = l_tmp
                                    n_line += 1
                                    cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                    break
                        if n_line == 2:  # quando ho trovato le due linee, trovo l'intersezione
                            inter_x, inter_y = intersection(l1, l2)
                            # ridimensiono i valori in modo da plottare il punto di intersez all'interno della finestra
                            if inter_y < 0:
                                inter_y = 0
                            if inter_y > 540:
                                inter_y = 540
                            if inter_x < 0:
                                inter_x = 0
                            if inter_x > 960:
                                inter_x = 960
                            # QUESTA PARTE VA MODIFICATA, SERVE PER CAPIRE L'INCLINAZIONE DELL'ANGOLO X, PER ORA
                            # PRENDO SOLO QUELLI CHE STANNO AL'INTERNO DI UN RANGE DEFINITO A TENTATIVI
                            if cX - 100 < inter_x < cX + 100:
                                # cv2.circle(output, (inter_x, inter_y), 10, (255, 0, 0), thickness=10)
                                if inter_y > 270:
                                    x_angle = 1  # interseca sopra
                                elif inter_y <= 270:
                                    x_angle = -1  # interseca sotto
                            break
                # se il contorno almeno 4 vertici
                if len(max_c) > 3:
                    # trovo il rettangolo rotato che definisce un'ellisse
                    rot_rect = cv2.fitEllipse(max_c)
                    # l'angolo z corrisponde all'inclinazione di questo rettangolo
                    z_angle = rot_rect[2]
                    # trovo il punto di apice e lo disegno
                    box = cv2.boxPoints(rot_rect)
                    box = np.int0(box)
                    a = box[0]
                    apex_point = apex(box[0][0], box[0][1], box[3][0], box[3][1])
                    cv2.circle(output, apex_point, 2, (255, 0, 0), thickness=2)
                    # Trovo l'altezza e la larghezza del catetere, serve per fare lo scalamento su vtk
                    height = sqrt((box[0][0] - box[1][0]) ** 2 + (box[0][1] - box[1][1]) ** 2)
                    width = sqrt((box[0][0] - box[3][0]) ** 2 + (box[0][1] - box[3][1]) ** 2)

                    # converto l'angolo tra -180 e 180
                    if z_angle > 90:
                        z_angle = z_angle - 180
                    print("Z_Angle is {:.2f}".format(z_angle))

                print("X_angle is {:.2f}".format(x_angle))

                # plotto il frame originale, la segmentazione, il contorno e l'output finale
                cv2.imshow("Frame", frame)
                cv2.imshow("Segmentation", img_segmented)
                cv2.imshow("Contours", img_contours)
                cv2.imshow("Output", output)

                # con questa funziona sovrappongo il modello 3D, però procede al frame successivo solo se premo 'q'
                overlay_catheter(z_angle, x_angle, name_path, (cX, cY), height, width, i)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        i += 1

    cap.release()
    cv2.destroyAllWindows()
