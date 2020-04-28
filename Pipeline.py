from __future__ import print_function
from keras_segmentation.predict import predict, model_from_checkpoint_path
import argparse
from math import sin, cos, sqrt
from myutils import *
from overlaycatheter import *


class_colors: List[Tuple[int, int, int]] = [(0, 0, 0), (255, 0, 0), (0, 255, 0)]


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

    # video_file_path = "..\\RealTime Video\\Piazzolla CV.mp4"

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
    out_vid = cv2.VideoWriter('..\\OutputVideo\\Degree.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 3, (960, 540))

    i = 0

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break
        # In questo modo si elabora 1 frame ogni 3, per ottimizzare il tutto cap.read() non dovrebbe leggere tutti i
        # frame ma solamente quelli che andiamo poi ad elaborare

        if (i % frame_interval) == 0:

            frame = cv2.resize(frame, dim, interpolation=cv2.INTER_LINEAR)

            # La cartella name_path serve per VTK quando va a leggere le immagini e le carica come oggetti
            name_path = "../OutputTMP/{}.png".format(i)
            cv2.imwrite(name_path, frame)

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

                # disegno la convex hull in un'immagine bianca
                img = np.zeros([dim[1], dim[0], 3], dtype=np.uint8)
                img.fill(0)
                cv2.drawContours(img, [hull], 0, (255, 255, 255), 2)

                if plot_contour == "Y":
                    cv2.drawContours(output, [hull], 0, (255, 255, 255), 2)

                # trovo i quattro estremi e il centro
                ext_left = tuple(max_c[max_c[:, :, 0].argmin()][0])
                ext_right = tuple(max_c[max_c[:, :, 0].argmax()][0])
                ext_top = tuple(max_c[max_c[:, :, 1].argmin()][0])
                ext_bot = tuple(max_c[max_c[:, :, 1].argmax()][0])
                M = cv2.moments(max_c)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                if plot_center == "Y":
                    cv2.circle(output, (cX, cY), 7, (0, 0, 0), -1)

                # converto e faccio un dilatazione per rendere i contorni più spessi in modo che sia più facile
                # individuare le linee da HoughLines
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                kernel_dilate = np.ones((7, 7), np.uint8)
                img = cv2.dilate(img, kernel_dilate, iterations=1)
                img_contours = img

                minLineLength = 20
                maxLineGap = 5
                lines = cv2.HoughLinesP(img, 1, np.pi / 180, 100, minLineLength, maxLineGap)

                if plot_all_lines == "Y":
                    x = output.copy()
                    if lines is not None:
                        for line in lines:
                            for x1, y1, x2, y2 in line:  # prendo i valori dei punti linea per linea
                                cv2.line(x, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.imshow("All Lines", x)
                                cv2.waitKey()

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
                        if n_line == 2:  # quando ho trovato le due linee, trovo l'intersezione
                            inter_x, inter_y = intersection(l1, l2)
                            angle_between = compute_angle(x1_line1, y1_line1, x2_line1, y2_line1, x1, y1, x2, y2)

                            if angle_between > 90:
                                angle_between = abs(angle_between - 180)

                            print("Frame {}: intersection is ({}, {}), angle between lines {}".format(i, inter_x,
                                                                                                      inter_y,
                                                                                                      angle_between))
                            # ridimensiono i valori in modo da plottare il punto di intersez all'interno della finestra
                            if plot_intersection == "Y":
                                if inter_y < 0:
                                    inter_y = 0
                                if inter_y > 540:
                                    inter_y = 540
                                if inter_x < 0:
                                    inter_x = 0
                                if inter_x > 960:
                                    inter_x = 960
                                cv2.circle(output, (inter_x, inter_y), 10, (255, 0, 0), thickness=10)

                            if inter_y > 540/2:
                                inters_pos = "Down"
                            elif inter_y <= 540/2:
                                inters_pos = "Up"

                            break

                # almeno 5 punti per fare fitellipse
                if len(max_c) > 5:
                    # trovo il rettangolo rotato che definisce un'ellisse
                    rot_rect = cv2.fitEllipse(max_c)
                    # l'angolo z corrisponde all'inclinazione di questo rettangolo
                    z_angle = rot_rect[2]
                    # trovo il punto di apice e lo disegno
                    box = cv2.boxPoints(rot_rect)
                    box = np.int0(box)
                    a = box[0]
                    apex_point = apex(box[0][0], box[0][1], box[3][0], box[3][1])
                    if plot_apex == "Y":
                        cv2.circle(output, apex_point, 2, (255, 0, 0), thickness=2)
                    if plot_ellipse == "Y":
                        cv2.drawContours(output, [box], 0, (255, 255, 255), 2)
                    # Trovo l'altezza e la larghezza del catetere, serve per fare lo scalamento su vtk
                    height = sqrt((box[0][0] - box[1][0]) ** 2 + (box[0][1] - box[1][1]) ** 2)
                    width = sqrt((box[0][0] - box[3][0]) ** 2 + (box[0][1] - box[3][1]) ** 2)

                    # converto l'angolo tra -180 e 180
                    if z_angle > 90:
                        z_angle = z_angle - 180

                avg_buf = buf.avg()
                if avg_buf != 0:
                    weight_with_Xprevious = (avg_buf * 0.8) + (angle_between * 0.2)
                    weight_gradual = (previous * 0.8) + (angle_between * 0.2)
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

                # plotto il frame originale, la segmentazione, il contorno e l'output finale
                if show_original == "Y":
                    cv2.imshow("Frame", frame)
                if show_segmentation == "Y":
                    cv2.imshow("Segmentation", img_segmented)
                if show_contour == "Y":
                    cv2.imshow("Contours", img_contours)

                cv2.putText(output, "Frame {}".format(i), (800, 100), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(output, "a: {:.8}".format(angle_between), (800, 200), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                if 8 > weight_gradual > 4:
                    cv2.circle(output, (850, 330), 10, (0, 255, 0), thickness=20)
                if 8 > weight_with_Xprevious > 4:
                    cv2.circle(output, (850, 430), 10, (255, 0, 0), thickness=20)
                cv2.imshow("Output", output)

                if save_video == "Y":
                    out_vid.write(output)

                # con questa funziona sovrappongo il modello 3D, però procede al frame successivo solo se premo 'q'
                if plot_3d == "Y":
                    if i % frame_interval_for_3d == 0:
                        overlay_catheter(z_angle, weight_with_Xprevious, inters_pos, name_path, (cX, cY), height, width, i)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        i += 1

    cap.release()
    out_vid.release()
    cv2.destroyAllWindows()
