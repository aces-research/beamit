# trace generated using paraview version 5.10.0-RC2

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()
import glob
import os

# width and height of the images
layoutWidth = 994
layoutHeight = 820

# get the details of the output directory
current_dir = os.getcwd()
files = glob.glob(f"{current_dir}/VTK/output-*.vtu")
files.sort(key=lambda x: int(x.split("output-")[-1].split(".")[0]))

# create a new 'XML Unstructured Grid Reader'
output = XMLUnstructuredGridReader(registrationName='output-*', FileName=files)
output.PointArrayStatus = ['displacements', 'forces', 'moments', 'damage']

# get animation scene
animationScene = GetAnimationScene()

# get active view
renderView = GetActiveViewOrCreate('RenderView')

#changing interaction mode based on data extents
renderView.InteractionMode = '2D'
renderView.CameraPosition = [0.0006670661759719098, -0.000465113366067217, 0.0031419952492217277]
renderView.CameraFocalPoint = [0.0006670661759719098, -0.000465113366067217, 0.0]
renderView.CameraParallelScale = 0.0008132082101202242
renderView.CameraViewUp = [0.0, 1.0, 0.0]

# create a new 'Warp By Vector'
warpByVector = WarpByVector(registrationName='WarpByVector', Input=output)
warpByVector.Vectors = ['POINTS', 'displacements']

# show data in view
warpByVectorDisplay = Show(warpByVector, renderView, 'UnstructuredGridRepresentation')
warpByVectorDisplay.Representation = 'Surface'

# Properties modified on warpByVectorDisplay
warpByVectorDisplay.LineWidth = 10.0

# Properties modified on warpByVectorDisplay
warpByVectorDisplay.RenderLinesAsTubes = 1

# hide data in view
Hide(output, renderView)

# update the view to ensure updated data information
renderView.Update()

# go to the end of the simulation
animationScene.GoToLast()

# get layout
layout = GetLayout()

# layout/tab size in pixels
layout.SetSize(layoutWidth, layoutHeight)

# create or clear directory for storing results
if not os.path.isdir('results'):
    os.mkdir('results')
else:
    for item in os.listdir('results'):
        os.remove(os.path.join('results', item))

# generate images of all the results
for i in range(0, len(output.PointArrayStatus)):

    # set scalar coloring of the current result
    ColorBy(warpByVectorDisplay, ('POINTS', output.PointArrayStatus[i]))

    # show color bar/color legend of the current result
    warpByVectorDisplay.SetScalarBarVisibility(renderView, True)

    # save screenshot of the current result
    SaveScreenshot(current_dir+'/results/'+output.PointArrayStatus[i]+'.png', renderView, ImageResolution=[layoutWidth, layoutHeight])

    # clear color bar/color legend of the previous result
    warpByVectorDisplay.SetScalarBarVisibility(renderView, False)