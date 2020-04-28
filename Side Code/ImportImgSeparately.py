from __future__ import print_function
import sys
from vtk import (
    vtkPNGReader, vtkImageCanvasSource2D, vtkImageActor, vtkPolyDataMapper,
    vtkRenderer, vtkRenderWindow, vtkRenderWindowInteractor, vtkOBJReader, vtkActor, vtkImageDataGeometryFilter
)
import vtk

if __name__ == '__main__':
    #  Verify input arguments

    angle = -12.6
    area = 10162

    path_img = "..\\Output\\10.png"
    reader_img = vtkPNGReader()
    reader_img.SetFileName(path_img)
    reader_img.Update()

    image_geometry_filter = vtkImageDataGeometryFilter()
    image_geometry_filter.SetInputConnection(reader_img.GetOutputPort())
    image_geometry_filter.Update()

    mapper_img = vtkPolyDataMapper()
    mapper_img.SetInputConnection(image_geometry_filter.GetOutputPort())

    actor_img = vtkActor()
    actor_img.SetMapper(mapper_img)
    #actor_img.SetPosition(0, 0, 0)
    #actor_img.GetProperty().SetPointSize(3)

    path_obj = "..\\Germano\\Germano\\OBJ_FILES\\URETERE.obj"
    reader_obj = vtkOBJReader()
    reader_obj.SetFileName(path_obj)

    mapper_obj = vtkPolyDataMapper()
    mapper_obj.SetInputConnection(reader_obj.GetOutputPort())

    actor_obj = vtkActor()
    actor_obj.SetMapper(mapper_obj)
    scale_y = area / 10000
    actor_obj.SetScale(5, scale_y, 5)
    actor_obj.SetPosition(300, 40, 20)  # z è avanti indietro, x destra sinistra, y su giu
    actor_obj.RotateZ(-angle)  # destra sinitra
    actor_obj.RotateX(4) # metto in linea con img

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

    render_window_interactor.Start()