from __future__ import print_function
import sys
from vtk import (
    vtkPNGReader, vtkImageCanvasSource2D, vtkImageActor, vtkPolyDataMapper,
    vtkRenderer, vtkRenderWindow, vtkRenderWindowInteractor, vtkOBJReader, vtkActor
)
import vtk

if __name__ == '__main__':
    #  Verify input arguments

    area = 10162
    angle = -12.6

    # Read the image
    img_path = "..\\Output\\10.png"
    png_reader = vtkPNGReader()
    png_reader.SetFileName(img_path)
    png_reader.Update()
    image_data = png_reader.GetOutput()

    # Create an image actor to display the image
    image_actor = vtkImageActor()
    # image_actor.SetScale(3)
    # image_actor.SetOrigin(200,200,1)

    image_actor.SetInputData(image_data)

    # Create a renderer to display the image in the background
    background_renderer = vtkRenderer()

    obj_path = "..\\Germano\\Oriented\\CATHETER.obj"
    reader = vtkOBJReader()
    reader.SetFileName(obj_path)

    mapper = vtkPolyDataMapper()
    mapper.SetInputConnection(reader.GetOutputPort())

    actor = vtkActor()
    actor.SetMapper(mapper)
    actor.SetScale(20)
    actor.RotateZ(-angle) # destra sinitra
    # actor.RotateX(-10.0) # su giu
    actor.SetOrigin(0, 0, 0)

    scene_renderer = vtkRenderer()
    render_window = vtkRenderWindow()

    # Set up the render window and renderers such that there is
    # a background layer and a foreground layer
    background_renderer.SetLayer(0)
    background_renderer.InteractiveOn()
    scene_renderer.SetLayer(1)
    render_window.SetNumberOfLayers(2)
    render_window.AddRenderer(background_renderer)
    render_window.AddRenderer(scene_renderer)
    size = image_data.GetExtent()
    render_window.SetSize(size[1], size[3])

    render_window_interactor = vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

    # Add actors to the renderers
    scene_renderer.AddActor(actor)
    background_renderer.AddActor(image_actor)

    # Render once to figure out where the background camera will be
    render_window.Render()

    # Set up the background camera to fill the renderer with the image
    origin = image_data.GetOrigin()
    spacing = image_data.GetSpacing()
    extent = image_data.GetExtent()

    camera = background_renderer.GetActiveCamera()
    camera.ParallelProjectionOn()

    xc = origin[0] + 0.5*(extent[0] + extent[1]) * spacing[0]
    yc = origin[1] + 0.5*(extent[2] + extent[3]) * spacing[1]
    # xd = (extent[1] - extent[0] + 1) * spacing[0]
    yd = (extent[3] - extent[2] + 1) * spacing[1]
    d = camera.GetDistance()

    camera.SetParallelScale(0.5 * yd)
    camera.SetFocalPoint(xc, yc, 0.0)
    camera.SetPosition(xc, yc, d)

    # Render again to set the correct view
    render_window.Render()

    # Interact with the window
    render_window_interactor.Start()