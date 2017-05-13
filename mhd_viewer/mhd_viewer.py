#!/usr/bin/env python

import sys
import os
import glob

from PyQt4 import uic, QtCore, QtGui
from PyQt4.QtGui import QMainWindow
import pyqtgraph as pg
from pyqtgraph.parametertree import ParameterTree, Parameter

import numpy as np
from math import sqrt
import SimpleITK as sitk

from exceptions import NotImplementedError


def getFilepathFromLocalFileID(localFileID):  # get real filepath from POSIX file in mac
  import CoreFoundation as CF
  import objc
  localFileQString = QtCore.QString(localFileID.toLocalFile())
  relCFStringRef = CF.CFStringCreateWithCString(
           CF.kCFAllocatorDefault,
           localFileQString.toUtf8(),
           CF.kCFStringEncodingUTF8
           )
  relCFURL = CF.CFURLCreateWithFileSystemPath(
         CF.kCFAllocatorDefault,
         relCFStringRef,
         CF.kCFURLPOSIXPathStyle,
         False   # is directory
         )
  absCFURL = CF.CFURLCreateFilePathURL(
         CF.kCFAllocatorDefault,
         relCFURL,
         objc.NULL
         )
  return QtCore.QUrl(str(absCFURL[0])).toLocalFile()


class MHDViewer(QMainWindow):
  def __init__(self, parent=None):
    super(MHDViewer, self).__init__(parent)
    dir_ = os.path.dirname(os.path.abspath(__file__))
    uic.loadUi(dir_ + '/' + 'layout.ui', self)
    # accept drop mhd files
    self.setAcceptDrops(True)
    self.acceptedFileTypes = ('mhd',)

    self.filepath = None  # mhd filepath
    self.data = None  # mhd 3d array
    self.image = None  # display 2d image
    self.origin = None  # x,y,z  Origin in world coordinates (mm)
    self.spacing = None  # spacing of voxels in world coor. (mm)

    self.axis = 'z'
    self.frame = 0
    
    # nodule circle
    self.showNodule = False  # plot nodule circle or not
    self.noduleX = 0.  # nodule x coordinate in mm
    self.noduleY = 0.  # nodule y coordinate in mm
    self.noduleZ = 0.  # nodule z coordinate in mm
    self.noduleDiameter = 0.  # nodule diameter in mm
    self.noduleItem = pg.ScatterPlotItem()
    self.imageView.getView().addItem(self.noduleItem)

    # parameter tree
    params_list = [
            {'name': 'File Info', 'type': 'group', 'children': [
              {'name': 'Filename', 'type': 'str', 'value': 'not set', 'readonly': True},
              {'name': 'Shape', 'type': 'str', 'value': 'unknown', 'readonly': True},
              {'name': 'Origin', 'type': 'str', 'value': 'unknown', 'readonly': True},
              {'name': 'Spacing', 'type': 'str', 'value': 'unknown', 'readonly': True},
            ]},
            {'name': 'Basic Operation', 'type': 'group', 'children': [
              {'name': 'Axis', 'type': 'list', 'values': ['x','y','z'], 'value': self.axis},
              {'name': 'Frame', 'type': 'int', 'value': self.frame},
            ]},
            {'name': 'Nodule', 'type': 'group', 'children': [
              {'name': 'Show', 'type': 'bool', 'value': self.showNodule},
              {'name': 'coorX', 'type': 'float', 'value': self.noduleX},
              {'name': 'coorY', 'type': 'float', 'value': self.noduleY},
              {'name': 'coorZ', 'type': 'float', 'value': self.noduleZ},
              {'name': 'diameter', 'type': 'float', 'value': self.noduleDiameter},
              {'name': 'Go to nodule', 'type': 'action'},
            ]},
            ]
    self.params = Parameter.create(name='params', 
      type='group', children=params_list)
    self.parameterTree.setParameters(self.params, showTop=False)

    # signal and slot
    self.fileList.itemDoubleClicked.connect(self.changeFileSlot)
    self.imageView.scene.sigMouseMoved.connect(self.mouseMoved)
    self.params.param('Basic Operation', 
      'Axis').sigValueChanged.connect(self.axisChangedSlot)
    self.params.param('Basic Operation', 
      'Frame').sigValueChanged.connect(self.frameChangedSlot)
    self.params.param('Nodule', 
      'Show').sigValueChanged.connect(self.showNoduleSlot)
    self.params.param('Nodule', 
      'coorX').sigValueChanged.connect(self.coorXChangedSlot)
    self.params.param('Nodule', 
      'coorY').sigValueChanged.connect(self.coorYChangedSlot)
    self.params.param('Nodule', 
      'coorZ').sigValueChanged.connect(self.coorZChangedSlot)
    self.params.param('Nodule', 
      'diameter').sigValueChanged.connect(self.diameterChangedSlot)
    self.params.param('Nodule', 
      'Go to nodule').sigActivated.connect(self.goToNoduleSlot)

    # disable old ROI and create new ROI
    self.imageView.roi.sigRegionChanged.disconnect(self.imageView.roiChanged)
    self.imageView.roi = pg.RectROI((0, 0), (10, 10), 0)
    self.imageView.roi.setZValue(20)
    self.imageView.view.addItem(self.imageView.roi)
    self.imageView.roi.hide()
    self.imageView.roi.sigRegionChanged.connect(self.roiChanged)

  def roiChanged(self):
    if self.data is None:
      return 
    data, coords = self.imageView.roi.getArrayRegion(
      self.imageView.image, self.imageView.imageItem, 
      (0, 1), returnMappedCoords=True)
    bins = np.arange(self.data.min(), self.data.max(), 2)
    y, x = np.histogram(data, bins=bins)
    self.imageView.roiCurve.setData(y=y, x=x[:len(bins)-1])

  def dragEnterEvent(self, event):
    urls = event.mimeData().urls()
    for url in urls:
      # mac posix filename
      if QtCore.QString(url.toLocalFile()).startsWith('/.file/id='):
        dropFile = getFilepathFromLocalFileID(url)
      else:
        dropFile = url.toLocalFile()
      fileInfo = QtCore.QFileInfo(dropFile)
      ext = fileInfo.suffix()
      if ext in self.acceptedFileTypes:
        event.accept()
        return 
    event.ignore()
    return 

  def dropEvent(self, event):
    urls = event.mimeData().urls()
    for url in urls:
      if QtCore.QString(url.toLocalFile()).startsWith('/.file/id='):
        dropFile = getFilepathFromLocalFileID(url)
      else:
        dropFile = url.toLocalFile()
      fileInfo = QtCore.QFileInfo(dropFile)
      ext = fileInfo.suffix()
      if ext in self.acceptedFileTypes:
        self.addFile(dropFile)

  def addFile(self, filepath):
    ext = str(QtCore.QFileInfo(filepath).suffix())
    item = FileItem(filepath=filepath)
    self.fileList.insertItem(0, item)

  def updateDisplay(self):
    if self.data is  None:
      return
    if self.axis == 'z':
      image = self.data[self.frame,:,:]
    elif self.axis == 'y':
      image = self.data[:,self.frame,:]
    elif self.axis == 'x':
      image = self.data[:,:,self.frame]
    self.image = image.T
    self.imageView.setImage(self.image, 
      autoRange=False, autoLevels=False, autoHistogramRange=True)
    if self.showNodule:
      nodule_mm = np.asarray([self.noduleX, self.noduleY, self.noduleZ])  # in mm
      nodule_voxel = (nodule_mm - self.origin) / self.spacing  # in voxel
      if self.axis == 'z':
        offset_z_voxel = abs(nodule_voxel[2] - self.frame)
        offset_z_mm = offset_z_voxel * self.spacing[2]
        if offset_z_mm >= self.noduleDiameter / 2.:
          self.noduleItem.clear()
          return
        circle_radius_mm = sqrt((self.noduleDiameter / 2.) ** 2. - offset_z_mm ** 2.)
        circle_radius_voxel = circle_radius_mm / self.spacing[0]
        self.noduleItem.setData([int(nodule_voxel[0])], [int(nodule_voxel[1])], 
          size=2. * circle_radius_voxel, symbol='o', 
          brush=(255,255,255,0), pen='r', pxMode=False)

  def changeFileSlot(self, item):
    # load mhd file and update display
    self.filepath = item.filepath
    itk_img = sitk.ReadImage(str(self.filepath))
    self.data = sitk.GetArrayFromImage(itk_img)  # indexes are z, y, x
    self.origin = np.array(itk_img.GetOrigin())
    self.spacing = np.array(itk_img.GetSpacing())
    self.updateDisplay()
    self.roiChanged()

    # update file info
    basename = os.path.basename(self.filepath)
    self.params.param('File Info', 
      'Filename').setValue(basename)
    shape = self.data.shape
    self.params.param('File Info', 
      'Shape').setValue('x: %d, y: %d, z: %d' %
      (shape[2], shape[1], shape[0]))
    self.params.param('File Info', 
      'Origin').setValue('x0: %.2f, y0: %.2f, z0: %.2f' %
      (self.origin[0], self.origin[1], self.origin[2]))
    self.params.param('File Info', 
      'Spacing').setValue('xs: %.2f, ys: %.2f, zs: %.2f' %
      (self.spacing[0], self.spacing[1], self.spacing[2]))

  def axisChangedSlot(self, _, axis):
    self.axis = axis
    self.updateDisplay()
    self.roiChanged()

  def frameChangedSlot(self, _, frame):
    self.frame = int(frame)
    self.updateDisplay()
    self.roiChanged()

  def showNoduleSlot(self, _, showNodule):
    self.showNodule = showNodule
    self.updateDisplay()
    self.roiChanged()

  def coorXChangedSlot(self, _, coorX):
    self.noduleX = coorX
    self.updateDisplay()
    self.roiChanged()

  def coorYChangedSlot(self, _, coorY):
    self.noduleY = coorY
    self.updateDisplay()
    self.roiChanged()

  def coorZChangedSlot(self, _, coorZ):
    self.noduleZ = coorZ
    self.updateDisplay()
    self.roiChanged()

  def diameterChangedSlot(self, _, diameter):
    self.noduleDiameter = diameter
    self.updateDisplay()
    self.roiChanged()

  def goToNoduleSlot(self):
    if self.axis == 'z':
      z = int((self.noduleZ - self.origin[2]) / self.spacing[2])
      self.params.param('Basic Operation', 
        'Frame').setValue(z)
    else:
      raise NotImplementedError('not finished...')

  def mouseMoved(self, pos):
    if self.data is None:
        return
    mouse_point = self.imageView.view.mapToView(pos)
    x, y = int(mouse_point.x()), int(mouse_point.y())
    shape = self.image.shape
    if 0 <= x < shape[0] and 0 <= y < shape[1]:
      self.statusbar.showMessage("x: %d  y:%d  Value:%.2E" % 
        (x, y, self.image[x, y]), 5000)


class FileItem(QtGui.QListWidgetItem):
  """docstring for FileItem"""
  def __init__(self, parent=None, filepath=None):
    super(FileItem, self).__init__(parent)
    self.filepath = str(filepath)
    basename = os.path.basename(self.filepath)
    self.setText(basename)
    self.setToolTip(self.filepath)


if __name__ == '__main__':
  # add signal to enable CTRL-C
  import signal
  signal.signal(signal.SIGINT, signal.SIG_DFL)

  app = QtGui.QApplication(sys.argv)
  viewer = MHDViewer()
  viewer.resize(900, 600)
  viewer.setWindowTitle("MHD Viewer")
  viewer.show()
  app.exec_()
