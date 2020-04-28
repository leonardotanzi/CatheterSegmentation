import cv2
import vtk


def main():
    dim = (960, 540)

    sceneRen = vtk.vtkRenderer()    #questo per il catetere e la prostata 3D
    sceneRen.SetLayer(1)
    backgr = vtk.vtkRenderer()      #questo per l'immagine di sfondo
    backgr.SetLayer(0)

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

    
    iren.Initialize()

    video_file_path = "..//RealTime Video//Cat_cropped.mp4"
    cap = cv2.VideoCapture(video_file_path)
    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_LINEAR)

        cv2.imshow("frame", frame) #finestra nativa di opencv x confrontare (di per se inutile, si potrebbe togliere per risparmiare mem)

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #necessario perchè opencv ha le immagini in BGR, e vtk no.
        img = cv2.flip(img, 0) #altrimenti l'immagine risulta capolvolta (solita question del Y-flip)

        #qui trasformiamo img in una immagine buona per VTK
        dataImporter.SetDataSpacing(1, 1, 1)
        dataImporter.SetWholeExtent(0, dim[0]-1, 0, dim[1]-1, 0, 2)
        dataImporter.SetDataExtentToWholeExtent()
        dataImporter.SetDataScalarTypeToUnsignedChar()
        dataImporter.SetNumberOfScalarComponents(3)
        dataImporter.SetImportVoidPointer(img)
        dataImporter.Update()
        #e la applichiamo all'attore...
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

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()