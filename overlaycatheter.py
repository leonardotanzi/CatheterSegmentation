from myutils3D import *

# gli angoli servono per la rotazione, il punto di centrale per la posizione (più semplice con questo metodo che col
# punto di apice, 'h' e 'w' sono le dimensioni originali del catetere per fare lo scalamento, 'path' serve per leggere le
# le immagini da importare, 'i' serve se si vogliono salvare i png e dare nomi diversi
def overlay_catheter(z_angle, weight, inters_pos, path, point, h, w, i):
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
    if 7 > weight > 3.5 and inters_pos == "Up":
        actor_obj.RotateX(-10)


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