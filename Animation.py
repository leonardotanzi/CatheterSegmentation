from __future__ import print_function
import vtk
import threading
import time

i = 0


def getScreenshot(self, fname, mag=10):
    r"""
    Generate a screenshot of the window and save to a png file

    Parameters
    ----------
    fname: str
        The file handle to save the image to
    mag: int, default 10
        The magnificaiton of the image, this will scale the resolution of
        the saved image by this face

    """
    self.SetAlphaBitPlanes(1)
    w2if = vtk.vtkWindowToImageFilter()
    w2if.SetInput(self)
    w2if.SetScale(mag)
    w2if.SetInputBufferTypeToRGBA()
    w2if.Update()

    writer = vtk.vtkPNGWriter()
    writer.SetFileName(fname)
    writer.SetInputConnection(w2if.GetOutputPort())
    writer.Write()


class vtkTimerCallback():
    def __init__(self):
        self.timer_count = 0

    def execute(self, obj, event):
        # print(self.timer_count)
        print("i = " + str(i))
        self.actor.SetPosition(self.timer_count, self.timer_count, 0)
        iren = obj
        iren.GetRenderWindow().Render()
        self.timer_count += 1
        getScreenshot(iren.GetRenderWindow(), "output{}.png".format(i), 1)


def render():
    # Create a sphere
    sphereSource = vtk.vtkSphereSource()
    sphereSource.SetCenter(0.0, 0.0, 0.0)
    sphereSource.SetRadius(5)

    # Create a mapper and actor
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(sphereSource.GetOutputPort())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    prop = actor.GetProperty()

    # Setup a renderer, render window, and interactor
    renderer = vtk.vtkRenderer()
    renderWindow = vtk.vtkRenderWindow()
    # renderWindow.SetWindowName("Test")

    renderWindow.AddRenderer(renderer)
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)

    # Add the actor to the scene
    renderer.AddActor(actor)
    renderer.SetBackground(1, 1, 1)  # Background color white

    # Render and interact
    renderWindow.Render()

    # Initialize must be called prior to creating timer events.
    renderWindowInteractor.Initialize()

    # Sign up to receive TimerEvent
    cb = vtkTimerCallback()
    cb.actor = actor
    renderWindowInteractor.AddObserver('TimerEvent', cb.execute)
    timerId = renderWindowInteractor.CreateRepeatingTimer(100)

    # start the interaction and timer
    renderWindowInteractor.Start()


if __name__ == '__main__':

    for i in range(100):
        if i == 0:
            # render()

            t = threading.Thread(target=render)
            t.start()

        time.sleep(0.1)
