import vtk

color_background = [0.0, 0.0, 0.0]

obj_path = "..\\Germano\\Germano\\OBJ_FILES\\URETERE.obj"

# this create a polygonal cylinder model with eight circum
reader = vtk.vtkOBJReader()
reader.SetFileName(obj_path)

reader.Update()

# the mapper is responsible for pushing the geometry into the graphics library.
mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(reader.GetOutputPort())

actor = vtk.vtkActor()
actor.SetMapper(mapper)
#actor.RotateZ(90.0)
actor.SetScale(1, 1, 1)

# Create a rendering window and renderer
# create the graphics structure. The rendere renders into the render window. The render window interactor captures
# mouse events and will perform appropriate camera and actors manipulations
ren = vtk.vtkRenderer()

renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(ren)

# Create a renderwindowinteractor
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)

# Assign actor to the renderer
ren.AddActor(actor)
ren.SetBackground(color_background)
renWin.SetSize(600, 600)

# ren.ResetCamera()
# ren.GetActiveCamera().Zoom(1.5)
# Enable user interface interactor

iren.Initialize()

renWin.Render()

iren.Start()