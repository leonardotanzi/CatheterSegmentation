import numpy as np
import math
from vtk import *

# 540 è l'altezza della finestra
def project_point_plane(pointxy, point_z=1.0, origin=[0, 0, 0], normal=[0, 0, 1]):
    projected_point = np.zeros(3)
    p = [pointxy[0], 540 - pointxy[1], point_z]
    vtkPlane.ProjectPoint(p, origin, normal, projected_point)
    return projected_point


def get_screenshot(ren_win, fname, mag=10):
    """
    fname: str
        The file handle to save the image to
    mag: int, default 10
        The magnificaiton of the image, this will scale the resolution of
        the saved image by this face

    """
    ren_win.SetAlphaBitPlanes(1)
    w2if = vtk.vtkWindowToImageFilter()
    w2if.SetInput(ren_win)
    w2if.SetScale(mag)
    w2if.SetInputBufferTypeToRGBA()
    w2if.Update()

    writer = vtk.vtkPNGWriter()
    writer.SetFileName(fname)
    writer.SetInputConnection(w2if.GetOutputPort())
    writer.Write()

def multiDimenDist(point1,point2):
   #find the difference between the two points, its really the same as below
   deltaVals = [point2[dimension]-point1[dimension] for dimension in range(len(point1))]
   runningSquared = 0
   #because the pythagarom theorm works for any dimension we can just use that
   for coOrd in deltaVals:
       runningSquared += coOrd**2
   return runningSquared**(1/2)


def findVec(point1,point2,unitSphere = False):
  #setting unitSphere to True will make the vector scaled down to a sphere with a radius one, instead of it's orginal length
  finalVector = [0 for coOrd in point1]
  for dimension, coOrd in enumerate(point1):
      #finding total differnce for that co-ordinate(x,y,z...)
      deltaCoOrd = point2[dimension]-coOrd
      #adding total difference
      finalVector[dimension] = deltaCoOrd
  if unitSphere:
      totalDist = multiDimenDist(point1,point2)
      unitVector =[]
      for dimen in finalVector:
          unitVector.append( dimen/totalDist)
      return unitVector
  else:
      return finalVector

def AngleBetween(v1, v2):
    x = v2[0] - v1[0]
    y = v2[1] - v1[1]

    return math.atan2(x, y)
