import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pyqtgraph as pg
import tifffile as tif

from PyQt5 import uic
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QWidget, QTableWidgetItem, QHeaderView
    
from ScopeFoundry import Measurement
from ScopeFoundry.helper_funcs import sibling_path, load_qt_ui_file

from HexSimProcessor.SIM_processing.hexSimProcessor import HexSimProcessor
from HexSIM_Analyser.image_decorr import ImageDecorr
from HexSIM_Analyser.get_h5_data import get_h5_dataset, get_h5_attr


def add_timer(function):
    """Function decorator to mesaure the execution time of a method.
    To avoid conflicts with QtObjects, it assumes that the method takes no arguments except self 
    """ 
    def inner(cls):
        print(f'\nStarting method "{function.__name__}" ...') 
        start_time = time.time() 
        result = function(cls) 
        end_time = time.time() 
        print(f'Execution time for method "{function.__name__}": {end_time-start_time:.6f} s') 
        return result
    inner.__name__ = function.__name__
    return inner 

    
def add_update_display(function):
    """Function decorator to to update display at the end of the execution
    To avoid conflicts with QtObjects, it assumes that the method takes no arguments except self 
    """ 
    def inner(cls):
        result = function(cls)
        cls.update_display()
        return result
    inner.__name__ = function.__name__
    return inner  


class HexSimAnalysis(Measurement):
    name = 'HexSIM_Analysis'

    def setup(self):
        """
        Initializes the setttings (aka logged quantities)
        """
        # load ui file
        self.ui_filename = sibling_path(__file__, "hexsim_analysis.ui")
        self.ui = load_qt_ui_file(self.ui_filename)
       
        self.settings.New('debug', dtype=bool, initial=False,
                          hardware_set_func = self.setReconstructor) 
        self.settings.New('cleanup', dtype=bool, initial=False, 
                          hardware_set_func = self.setReconstructor) 
        self.settings.New('gpu', dtype=bool, initial=False, 
                          hardware_set_func = self.setReconstructor) 
        self.settings.New('compact', dtype=bool, initial=False, 
                          hardware_set_func = self.setReconstructor) 
        self.settings.New('axial', dtype=bool, initial=False, 
                          hardware_set_func = self.setReconstructor) 
        self.settings.New('usemodulation', dtype=bool, initial=True, 
                          hardware_set_func = self.setReconstructor) 
        self.settings.New('magnification', dtype=float, initial=63,  spinbox_decimals=2,
                          hardware_set_func = self.setReconstructor)
        self.settings.New('NA', dtype=float, initial=0.75,  spinbox_decimals=2,
                          hardware_set_func = self.setReconstructor)
        self.settings.New('n', dtype=float, initial=1.0,  spinbox_decimals=2,
                          hardware_set_func = self.setReconstructor)
        self.settings.New('wavelength', dtype=float, initial=0.532,  spinbox_decimals=3,
                          hardware_set_func = self.setReconstructor)
        self.settings.New('pixelsize', dtype=float, initial=5.85,  spinbox_decimals=3, unit = 'um',
                          hardware_set_func = self.setReconstructor)
        self.settings.New('alpha', dtype=float, initial=0.500,  spinbox_decimals=3, description='0th att width',
                          hardware_set_func = self.setReconstructor)
        self.settings.New('beta', dtype=float, initial=0.950,  spinbox_decimals=3,description='0th width',
                          hardware_set_func = self.setReconstructor)
        self.settings.New('w', dtype=float, initial=5.00, spinbox_decimals=2, description='wiener parameter',
                          hardware_set_func = self.setReconstructor)
        self.settings.New('eta', dtype=float, initial=0.70, spinbox_decimals=2, 
                          description='must be smaller than the sources radius normalized on the pupil size',
                          hardware_set_func = self.setReconstructor)
        self.settings.New('find_carrier', dtype=bool, initial=True,
                          hardware_set_func = self.setReconstructor)
        self.settings.New('selectROI', dtype=bool, initial=True) 
        self.settings.New('ROI_size', dtype=int, initial=512, vmin=1, vmax=4096) 
        
        
    def setup_figure(self):
        
        self.display_update_period = 0.5 
        
        self.imvRaw = pg.ImageView()
        self.imvRaw.ui.roiBtn.hide()
        self.imvRaw.ui.menuBtn.hide()

        self.imvSIM = pg.ImageView()
        self.imvSIM.ui.roiBtn.hide()
        self.imvSIM.ui.menuBtn.hide()

        self.imvWF = pg.ImageView()
        self.imvWF.ui.roiBtn.hide()
        self.imvWF.ui.menuBtn.hide()

        self.ui.rawImageLayout.addWidget(self.imvRaw)
        self.ui.simImageLayout.addWidget(self.imvSIM)
        self.ui.wfImageLayout.addWidget(self.imvWF)

        # Image initialization
        self.imageRawShape = (1, self.settings['ROI_size'], self.settings['ROI_size']) # TODO update to 4D case
        self.imageSIMShape = [self.imageRawShape[-2]*2, self.imageRawShape[-1]*2]
        self.imageWFShape = [self.imageRawShape[-2],self.imageRawShape[-1]]
        self.imageRaw = np.zeros(self.imageRawShape, dtype=np.uint16) 
        self.imageSIM = np.zeros(self.imageSIMShape, dtype=np.uint16) 
        self.imageWF = np.zeros(self.imageWFShape, dtype=np.uint16)

        self.imvRaw.setImage(self.imageRaw, autoRange=False, autoLevels=True, autoHistogramRange=True)
        self.imvWF.setImage(self.imageWF, autoRange=False, autoLevels=True, autoHistogramRange=True)
        self.imvSIM.setImage(self.imageSIM, autoRange=False, autoLevels=True, autoHistogramRange=True)

        # Toolbox
        self.ui.loadFileButton.clicked.connect(self.loadFile)
        self.ui.resetButton.clicked.connect(self.reset)
        self.ui.calibrationResult.clicked.connect(self.showMessageWindow)

        self.ui.calibrationSave.clicked.connect(self.saveMeasurements) #TODO change UI names
        self.ui.calibrationLoad.clicked.connect(self.loadCalibrationResults)

        self.ui.standardSimButton.clicked.connect(self.standard_reconstruction)
        self.ui.calibrationButton.clicked.connect(self.calibration)

        self.ui.batchSimButton.clicked.connect(self.batch_recontruction)
        
        self.ui.resolutionEstimateButton.clicked.connect(self.estimate_resolution)
        
        self.start_sim_processor()


    def update_display(self):
        """
        This function runs repeatedly and automatically during the measurement run,
        its update frequency is defined by self.display_update_period.
        """
        self.display_update_period = 0.5 #unused
        self.settings['progress'] = 0.0 #unused

        if hasattr(self, 'showCalibrationResult'):
            if self.showCalibrationResult:
                self.showMessageWindow()
                self.showCalibrationResult = False
                self.isCalibrationSaved = False
                
        self.imvRaw.setImage(self.imageRaw,
                                 autoRange=True, autoLevels=True, autoHistogramRange=True)
        self.imvWF.setImage(self.imageWF,
                                 autoRange=True, autoLevels=True, autoHistogramRange=True) 
        self.imvSIM.setImage(self.imageSIM,
                                 autoRange=True, autoLevels=True, autoHistogramRange=True)


    def start_sim_processor(self):
        self.isCalibrated = False
        self.kx_input = np.zeros((3, 1), dtype=np.single)
        self.ky_input = np.zeros((3, 1), dtype=np.single)
        self.p_input = np.zeros((3, 1), dtype=np.single)
        self.ampl_input = np.zeros((3, 1), dtype=np.single)
        
        if not hasattr(self, 'h'):
            self.h = HexSimProcessor()  # create reconstruction object
            self.h.opencv = False
            self.setReconstructor()
            
    def stop_sim_processor(self):
        if hasattr(self, 'h'):
            delattr(self, 'h')
     
        
    def run(self):
        pass
   
    
    def load_h5_file(self,filename):
            self.imageRaw = get_h5_dataset(filename) #TODO read h5 info
            self.imageRawShape = self.imageRaw.shape
            self.raw2WideFieldImage()
            self.enableROIselection()
            
            #load some settings, if present in the h5 file
            for key in ['magnification','n','NA','pixelsize','wavelength']:
                val = get_h5_attr(filename, key)
                if len(val)>0:
                    new_value = val[0]
                    self.settings[key] = new_value
                    self.show_text(f'Updated {key} to: {new_value} ')
            
            
    def enableROIselection(self):
        """
        If the image has not the size specified in ROIsize,
        listen for click event on the pg.ImageView imvRaw.
        """
        def click(event):
            """
            Resizes imageRaw on click event, to the specified size 'ROI_size'
            around the clicked point.
            """
            if self.settings['selectROI']:
                ROIsize = self.settings['ROI_size']
                Ly =self.imageRaw.shape[-1]
                Lx =self.imageRaw.shape[-2]
                event.accept()  
                pos = event.pos()
                x = int(pos.x()) #pyqtgraph is transposed
                y = int(pos.y())
                x = max(min(x, Lx-ROIsize//2 ),ROIsize//2 )
                y = max(min(y, Ly-ROIsize//2 ),ROIsize//2 )
                xmin = x - ROIsize//2
                ymin = y - ROIsize//2
                self.imageRaw = self.imageRaw [:,xmin:xmin+ROIsize, ymin:ymin+ROIsize] 
                self.imageRawShape = self.imageRaw.shape
                self.raw2WideFieldImage()
                self.update_display()
                self.settings['selectROI'] = False
                self.show_text(f'ROI set to shape: {self.imageRawShape}')
            
        
        self.imvRaw.getImageItem().mouseClickEvent = click
        self.imvWF.getImageItem().mouseClickEvent = click
        self.settings['selectROI'] = True

   
    def load_tif_file(self,filename):
        self.imageRaw = np.single(tif.imread(filename))
        self.imageRawShape = self.imageRaw.shape
        self.raw2WideFieldImage()
        try:
            # get file name of txt file
            for file in os.listdir(self.filepath):
                if file.endswith(".txt"):
                    configFileName = os.path.join(self.filepath, file)

            configFile = open(configFileName, 'r')
            configSet = json.loads(configFile.read())

            self.kx_input = np.asarray(configSet["kx"])
            self.ky_input = np.asarray(configSet["ky"])
            self.p_input = np.asarray(configSet["phase"])
            self.ampl_input = np.asarray(configSet["amplitude"])

            # set value
            self.settings['magnification'] = configSet["magnification"]
            self.settings['NA'] = configSet["NA"]
            self.settings['n'] = configSet["refractive index"]
            self.settings['wavelength'] = configSet["wavelength"]
            self.settings['pixelsize']  = configSet["pixelsize"]

            try:
                self.exposuretime = configSet["camera exposure time"]
            except:
                self.exposuretime = configSet["exposure time (s)"]

            try:
                self.laserpower = configSet["laser power (mW)"]
            except:
                self.laserpower = 0

            txtDisplay = "File name:\t {}\n" \
                         "Array size:\t {}\n" \
                         "Wavelength:\t {} um\n" \
                         "Exposure time:\t {:.3f} s\n" \
                         "Laser power:\t {} mW".format(self.filetitle, self.imageRawShape, \
                                                                configSet["wavelength"], \
                                                                self.exposuretime, self.laserpower)
            self.show_text(txtDisplay)

        except:
            self.show_text("No information about this measurement.")
    
    
    @add_update_display
    def loadFile(self):
        
            try:
                filename, _ = QFileDialog.getOpenFileName(directory = self.app.settings['save_dir'])
                
                if filename.endswith('.tif') or filename.endswith('.tiff'):
                    self.load_tif_file(filename)
                elif filename.endswith('.h5') or filename.endswith('.hdf5'):
                    self.load_h5_file(filename)
                else:
                    raise OSError('Invalid file type')
    
                self.filetitle = Path(filename).stem
                self.filepath = os.path.dirname(filename)
                self.isFileLoad = True
            except:
                self.isFileLoad = False
    
            if self.isFileLoad:
                self.isCalibrated = False
                self.setReconstructor()
                self.h._allocate_arrays()
            else:
                txtDisplay = "File is not loaded."
                self.show_text(txtDisplay)
                

    def raw2WideFieldImage(self):
        self.imageWF = np.mean(self.imageRaw, axis=0)
        self.imageWFShape = self.imageWF.shape
    
        
    @add_update_display        
    def reset(self):
        self.isFileLoad = False
        self.isCalibrated = False
        self.stop_sim_processor()
        self.start_sim_processor()
        self.imageSIM = np.zeros(self.imageSIMShape, dtype=np.uint16) 
        self.imageRaw = np.zeros(self.imageRawShape, dtype=np.uint16)
        self.imageWF = np.zeros(self.imageWFShape, dtype=np.uint16)
        
      
    @add_timer
    def calibration(self):  
        try:
            self.setReconstructor()
            if self.settings['gpu']:
                self.h.calibrate_cupy(self.imageRaw, self.isFindCarrier)       
            else:
                self.h.calibrate(self.imageRaw,self.isFindCarrier)          
            self.isCalibrated = True
            self.find_phaseshifts()
            self.show_text('Calibration completed')
         
        except Exception as e:
            self.isCalibrated = False
            txtDisplay = f'Calibration encountered an error \n{e}'
            self.show_text(txtDisplay)
             
        
    @add_update_display
    @add_timer  
    def standard_reconstruction(self):
        try:
            self.setReconstructor()
            if self.isCalibrated:
                
                if self.settings['gpu']:
                    self.imageSIM = self.h.reconstruct_cupy(self.imageRaw)
    
                elif not self.settings['gpu']:
                    self.imageSIM = self.h.reconstruct_rfftw(self.imageRaw)
            else:
                self.calibration()
                if self.isCalibrated:
                    self.standard_reconstruction()
                    
            self.imageSIMShape = self.imageSIM.shape
        
        except Exception as e:
            txtDisplay = f'Reconstruction encountered an error \n{e}'
            self.show_text(txtDisplay)
        
    @add_update_display
    @add_timer    
    def batch_recontruction(self): # TODO fix this reconstruction with  multiple batches (multiple planes)
        self.setReconstructor()
        if self.isCalibrated:
            # Batch reconstruction
            if self.settings['gpu']:
                if self.settings['compact']:
                    self.imageSIM = self.h.batchreconstructcompact_cupy(self.imageRaw)
                elif not self.settings['compact']:
                    self.imageSIM = self.h.batchreconstruct_cupy(self.imageRaw)

            elif not self.settings['gpu']:
                if self.settings['compact']:
                    self.imageSIM = self.h.batchreconstructcompact(self.imageRaw)
                elif not self.settings['compact']:
                    self.imageSIM = self.h.batchreconstruct(self.imageRaw)
            
        elif not self.isCalibrated:
            nStack = len(self.imageRaw)
            # calibrate & reconstruction
            if self.settings['gpu']:
                self.h.calibrate_cupy(self.imageRaw[int(nStack // 2):int(nStack // 2 + 7), :, :], self.isFindCarrier)
                self.isCalibrated = True
                
                if self.settings['compact']:
                    self.imageSIM = self.h.batchreconstructcompact_cupy(self.imageRaw)
                elif not self.settings['compact']:
                    self.imageSIM = self.h.batchreconstruct_cupy(self.imageRaw)
                

            elif not self.settings['gpu']:
                self.h.calibrate(self.imageRaw[int(nStack // 2):int(nStack // 2 + 7), :, :], self.isFindCarrier)
                self.isCalibrated = True
                
                if self.settings['compact']:
                    self.imageSIM = self.h.batchreconstructcompact(self.imageRaw)
                elif not self.settings['compact']:
                    self.imageSIM = self.h.batchreconstruct(self.imageRaw)
        self.imageSIMShape = self.imageSIM.shape

    def setReconstructor(self,*args):
        self.isFindCarrier = self.settings['find_carrier']
        self.h.debug = self.settings['debug']
        self.h.cleanup = self.settings['cleanup']
        self.h.axial = self.settings['axial']
        self.h.usemodulation = self.settings['usemodulation']
        self.h.magnification = self.settings['magnification']
        self.h.NA = self.settings['NA']
        self.h.n = self.settings['n']
        self.h.wavelength = self.settings['wavelength']
        self.h.pixelsize = self.settings['pixelsize']
        self.h.alpha = self.settings['alpha']
        self.h.beta = self.settings['beta']
        self.h.w = self.settings['w']
        self.h.eta = self.settings['eta']
        if not self.isFindCarrier:
            self.h.kx = self.kx_input
            self.h.ky = self.ky_input
            

    def saveMeasurements(self):
        t0 = time.time()
        timestamp = datetime.fromtimestamp(t0)
        timestamp = timestamp.strftime("%Y%m%d%H%M")
        pathname = self.filepath + '/reprocess'
        Path(pathname).mkdir(parents=True,exist_ok=True)
        simimagename = pathname + '/' + self.filetitle + timestamp + f'_reprocessed' + '.tif'
        wfimagename = pathname + '/' + self.filetitle + timestamp + f'_widefield' + '.tif'
        txtname =      pathname + '/' + self.filetitle + timestamp + f'_reprocessed' + '.txt'
        tif.imwrite(simimagename, np.single(self.imageSIM))
        tif.imwrite(wfimagename,np.uint16(self.imageWF))

        savedictionary = {
            #"exposure time (s)":self.exposuretime,
            #"laser power (mW)": self.laserpower,
            # "z stepsize (um)":  self.
            # System setup:
            "magnification" :   self.h.magnification,
            "NA":               self.h.NA,
            "refractive index": self.h.n,
            "wavelength":       self.h.wavelength,
            "pixelsize":        self.h.pixelsize,
            # Calibration parameters:
            "alpha":            self.h.alpha,
            "beta":             self.h.beta,
            "Wiener filter":    self.h.w,
            "eta":              self.h.eta,
            "cleanup":          self.h.cleanup,
            "axial":            self.h.axial,
            "modulation":       self.h.usemodulation,
            "kx":               self.h.kx,
            "ky":               self.h.ky,
            "phase":            self.h.p,
            "amplitude":        self.h.ampl
            }
        f = open(txtname, 'w+')
        f.write(json.dumps(savedictionary, cls=NumpyEncoder,indent=2))
        self.isCalibrationSaved = True


    @add_update_display
    @add_timer   
    def estimate_resolution(self): #TODO : consider to add QT timers
            pixelsizeWF = self.h.pixelsize / self.h.magnification
            ciWF = ImageDecorr(self.imageWF, square_crop=True,pixel_size=pixelsizeWF)
            optimWF, resWF = ciWF.compute_resolution()
            ciSIM = ImageDecorr(self.imageSIM, square_crop=True,pixel_size=pixelsizeWF/2)
            optimSIM, resSIM = ciSIM.compute_resolution()
            txtDisplay = f"Wide field image resolution:\t {ciWF.resolution:.3f} um \
                  \nSIM image resolution:\t {ciSIM.resolution:.3f} um\n"
            self.show_text(txtDisplay)
        
        

    def loadCalibrationResults(self):
        try:
            filename, _ = QFileDialog.getOpenFileName(caption="Open file", directory=self.app.settings['save_dir'], filter="Text files (*.txt)")
            file = open(filename,'r')
            loadResults = json.loads(file.read())
            self.kx_input = np.asarray(loadResults["kx"])
            self.ky_input = np.asarray(loadResults["ky"])
            self.show_text("Calibration results are loaded.")
        except:
            self.show_text("Calibration results are not loaded.")
            
    
    def find_phaseshifts(self):
        self.h.phaseshift = np.zeros((4,7))
        self.h.expected_phase = np.zeros((4,7))
    
        for i in range (3):
            phase, _ = self.h.find_phase(self.h.kx[i],self.h.ky[i],self.imageRaw)
            self.h.expected_phase[i,:] = np.arange(7) * 2*(i+1) * np.pi / 7
            self.h.phaseshift[i,:] = np.unwrap(phase - self.h.expected_phase[i,:]) + self.h.expected_phase[i,:] - phase[0]
    
        self.h.phaseshift[3] = self.h.phaseshift[2]-self.h.phaseshift[1]-self.h.phaseshift[0]

    def show_text(self, text):
        self.ui.MessageBox.insertPlainText(text+'\n')
        self.ui.MessageBox.ensureCursorVisible()
        print(text)    
        
    def showMessageWindow(self):
        messageWindow = MessageWindow(self.h, self.kx_input, self.ky_input)
        messageWindow.show()


class MessageWindow(QWidget):

    """
    This window display the Winier filter and other debug data
    """

    def __init__(self, h, kx, ky):
        super().__init__()
        self.ui = uic.loadUi(sibling_path(__file__, "calibration_results.ui"),self)
        self.h = h
        self.kx = kx
        self.ky = ky
        self.setWindowTitle('Calibration results')
        self.showCurrentTable()
        self.show_images()
        
    def show_images(self):
        if all(hasattr(self.h, attr) for attr in ["wienerfilter",
                                                  "phaseshift",
                                                  "expected_phase"]):
            widgets =[pg.ImageView(),
                      pg.PlotWidget(),
                      pg.ImageView()]
            layouts = ['wienerLayout',
                       'phaseLayout',
                       'otfLayout']
            data_to_show = [self.h.wienerfilter,
                            [self.h.phaseshift, self.h.expected_phase],
                            self.h.wienerfilter]
            for widget, layout_name, data in zip(widgets,layouts,data_to_show):
                layout = getattr(self.ui, layout_name)
                layout.addWidget(widget)
                if isinstance(widget, pg.ImageView):
                        self.show_image_data(widget,data)
                if isinstance(widget, pg.PlotWidget):
                        self.show_plot_data(widget,data)
    

    def showCurrentTable(self):

        self.ui.currentTable.setItem(0, 0, QTableWidgetItem(str(self.kx[0]).lstrip('[').rstrip(']')))
        self.ui.currentTable.setItem(0, 1, QTableWidgetItem(str(self.kx[1]).lstrip('[').rstrip(']')))
        self.ui.currentTable.setItem(0, 2, QTableWidgetItem(str(self.kx[2]).lstrip('[').rstrip(']')))
        #
        self.ui.currentTable.setItem(1, 0, QTableWidgetItem(str(self.ky[0]).lstrip('[').rstrip(']')))
        self.ui.currentTable.setItem(1, 1, QTableWidgetItem(str(self.ky[1]).lstrip('[').rstrip(']')))
        self.ui.currentTable.setItem(1, 2, QTableWidgetItem(str(self.ky[2]).lstrip('[').rstrip(']')))

        self.ui.currentTable.setItem(2, 0, QTableWidgetItem(str(self.h.kx[0]).lstrip('[').rstrip(']')))
        self.ui.currentTable.setItem(2, 1, QTableWidgetItem(str(self.h.kx[1]).lstrip('[').rstrip(']')))
        self.ui.currentTable.setItem(2, 2, QTableWidgetItem(str(self.h.kx[2]).lstrip('[').rstrip(']')))
        #
        self.ui.currentTable.setItem(3, 0, QTableWidgetItem(str(self.h.ky[0]).lstrip('[').rstrip(']')))
        self.ui.currentTable.setItem(3, 1, QTableWidgetItem(str(self.h.ky[1]).lstrip('[').rstrip(']')))
        self.ui.currentTable.setItem(3, 2, QTableWidgetItem(str(self.h.ky[2]).lstrip('[').rstrip(']')))
        #
        self.ui.currentTable.setItem(4, 0, QTableWidgetItem(str(self.h.p[0]).lstrip('[').rstrip(']')))
        self.ui.currentTable.setItem(4, 1, QTableWidgetItem(str(self.h.p[1]).lstrip('[').rstrip(']')))
        self.ui.currentTable.setItem(4, 2, QTableWidgetItem(str(self.h.p[2]).lstrip('[').rstrip(']')))
        #
        self.ui.currentTable.setItem(5, 0, QTableWidgetItem(str(self.h.ampl[0]).lstrip('[').rstrip(']')))
        self.ui.currentTable.setItem(5, 1, QTableWidgetItem(str(self.h.ampl[1]).lstrip('[').rstrip(']')))
        self.ui.currentTable.setItem(5, 2, QTableWidgetItem(str(self.h.ampl[2]).lstrip('[').rstrip(']')))

        # Table will fit the screen horizontally
        self.currentTable.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)


    def show_image_data(self, widget, image_to_show):
        widget.aspectRatioMode = Qt.KeepAspectRatio
        widget.ui.roiBtn.hide()
        widget.ui.menuBtn.hide()
        widget.ui.histogram.hide()
        widget.setImage(image_to_show, autoRange=True, autoLevels=True)
        widget.adjustSize()
    
    
    def show_plot_data(self, widget, plot_to_show):
        widget.aspectRatioMode = Qt.KeepAspectRatio
        for idx in range(len(plot_to_show[0])):
            widget.plot(plot_to_show[0][idx], symbol = '+', color ='r')
            widget.plot(plot_to_show[1][idx])
        widget.adjustSize()


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

    
if __name__ == "__main__" :
    from ScopeFoundry import BaseMicroscopeApp
    import sys
    class testApp(BaseMicroscopeApp):
        def setup(self):
            self.add_measurement(HexSimAnalysis)
            self.ui.show()
            self.ui.activateWindow()

    app = testApp(sys.argv)
    
    app.settings_load_ini(".\\Settings\\HexSIM_Analysis.ini")
    sys.exit(app.exec_())