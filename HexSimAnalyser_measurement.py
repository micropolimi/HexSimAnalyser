import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pyqtgraph as pg
import tifffile as tif
from numpy.fft import fft2, fftshift

# from PyQt5.QtWidgets import QFileDialog, QTableWidgetItem
from qtpy.QtWidgets import QFileDialog, QTableWidgetItem   

from ScopeFoundry import Measurement
from ScopeFoundry.helper_funcs import sibling_path, load_qt_ui_file

from HexSimProcessor.SIM_processing.hexSimProcessor import HexSimProcessor 
from HexSimProcessor.SIM_processing.simProcessor import SimProcessor 
from HexSimAnalyser.image_decorr import ImageDecorr
from HexSimAnalyser.get_h5_data import get_h5_dataset, get_multiple_h5_datasets, get_h5_attr, get_datasets_index_by_name, get_group_name


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


class HexSimAnalysis(Measurement):
    name = 'HexSIM_Analysis'

    def setup(self):
        """
        Initializes the setttings (aka logged quantities)
        """
        # load ui file
        self.ui_filename = sibling_path(__file__, "hexsim_analysis.ui")
        self.ui = load_qt_ui_file(self.ui_filename)
        
        self.settings.New('phases_number', dtype=int, initial=7,
                          hardware_set_func = self.reset_processor)
        # self.settings.phases_number.add_listener(self.reset_processor)
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
        self.settings.New('eta', dtype=float, initial=0.15, spinbox_decimals=2, 
                          description='must be smaller than the sources radius normalized on the pupil size',
                          hardware_set_func = self.setReconstructor)
        self.settings.New('find_carrier', dtype=bool, initial=True,                         
                          hardware_set_func = self.setReconstructor)
        self.settings.New('selectROI', dtype=bool, initial=False) 
        self.settings.New('roiX', dtype=int, initial=600)
        self.settings.New('roiY', dtype=int, initial=1200)
        self.settings.New('ROI_size', dtype=int, initial=512, vmin=1, vmax=2048) 
        self.settings.New('dataset_index', dtype = int, initial=0, vmin = 0,
                          hardware_set_func = self.set_dataset)
        self.settings.New('frame_index', dtype = int, initial=0, vmin = 0,
                          hardware_set_func = self.set_frame)
        
    def setup_figure(self):
        #self.display_update_period = 0.5 
        
        self.imvRaw = pg.ImageView()
        self.imvRaw.ui.roiBtn.hide()
        self.imvRaw.ui.menuBtn.hide()

        self.imvSIM = pg.ImageView()
        self.imvSIM.ui.roiBtn.hide()
        self.imvSIM.ui.menuBtn.hide()
        
        self.imvFft = pg.ImageView()
        self.imvFft.ui.roiBtn.hide()
        self.imvFft.ui.menuBtn.hide()
        
        self.imvSimFft = pg.ImageView()
        self.imvSimFft.ui.roiBtn.hide()
        self.imvSimFft.ui.menuBtn.hide()
        
        self.imvWF = pg.ImageView()
        self.imvWF.ui.roiBtn.hide()
        self.imvWF.ui.menuBtn.hide()
        #self.imvWF......add_listener(self.set_frame) #TODO add listener

        self.ui.rawImageLayout.addWidget(self.imvRaw)
        self.ui.simImageLayout.addWidget(self.imvSIM)
        self.ui.wfImageLayout.addWidget(self.imvWF)
        self.ui.fftLayout.addWidget(self.imvFft)
        self.ui.simFftLayout.addWidget(self.imvSimFft)

        self.settings.dataset_index.connect_to_widget(self.ui.datasetSpinBox)
        self.settings.frame_index.connect_to_widget(self.ui.frameSpinBox)
        self.ui.loadFileButton.clicked.connect(self.loadFile)
        self.ui.resetButton.clicked.connect(self.reset_processor)
        self.ui.cutRoiButton.clicked.connect(self.cutRoi)

        self.ui.calibrationSave.clicked.connect(self.saveMeasurements) 
        self.ui.calibrationLoad.clicked.connect(self.loadCalibrationResults)
        
        self.ui.standardSimButton.clicked.connect(self.standard_reconstruction)
        self.ui.calibrationButton.clicked.connect(self.calibration)
        self.ui.batchSimButton.clicked.connect(self.batch_recontruction)
        self.ui.resolutionEstimateButton.clicked.connect(self.estimate_resolution)
        
        self.start_sim_processor()
        
    def update_display(self):
        pass
     
    def run(self):
        pass
        
    @property
    def imageRaw(self):
        """
        self.imageRaw is the original acquired image. It is organized in a 4D numpy array with
        phases, z, y, x
        """
        return self._imageRaw
            
    @imageRaw.setter
    def imageRaw(self, img_raw): 
        """
        Setter function: selectes the frame to choose, generates and sets the
        widefield stack and the power spectrum
        """
        size_z = img_raw.shape[1] 
        frame_index = min(self.settings.frame_index.val, size_z-1)
        self._imageRaw = img_raw 
        self.imvRaw.setImage(img_raw[:,frame_index,:,:], autoRange=True, autoLevels=True, autoHistogramRange=True)
        imageWF = self.imageWF = self.calculate_WF_image(img_raw) 
        self.imvWF.setImage(imageWF, autoRange=True, autoLevels=True, autoHistogramRange=True) 
        self.imvWF.setCurrentIndex(frame_index) 
        self.settings.frame_index.change_min_max(vmin = 0, vmax = size_z-1)
        spectrum = self.calculate_spectrum(img_raw[:,frame_index,:,:]) # calculates the power spectrum of imageRaw for the first phase
        self.imvFft.setImage(spectrum, autoRange=True, autoLevels=True, autoHistogramRange=True) 
        

    @property
    def imageSIM(self):
        return self._imageSIM
            
    @imageSIM.setter
    def imageSIM(self, img_sim): 
        self._imageSIM = img_sim 
        self.imvSIM.setImage(img_sim, autoRange=True, autoLevels=True, autoHistogramRange=True)
        spectrum = self.calculate_spectrum(img_sim)
        self.imvSimFft.setImage(spectrum, autoRange=True, autoLevels=True, autoHistogramRange=True) 
        
    def start_sim_processor(self):
        self.isCalibrated = False
        self.kx_input = np.zeros((3, 1), dtype=np.single)
        self.ky_input = np.zeros((3, 1), dtype=np.single)
        self.p_input = np.zeros((3, 1), dtype=np.single)
        self.ampl_input = np.zeros((3, 1), dtype=np.single)
        
        if hasattr(self, 'h'):
            self.reset()
            self.start_sim_processor()
        else:
            if self.settings['phases_number'] == 7: 
                self.h = HexSimProcessor()  # create reconstruction object
                self.h.opencv = False
                self.setReconstructor()        
            elif self.settings['phases_number'] == 3:
                self.h = SimProcessor()  # create reconstruction object
                self.h.opencv = False
                self.setReconstructor()           
            else: 
                raise(ValueError("Invalid number of phases"))
            
    def stop_sim_processor(self):
        if hasattr(self, 'h'):
            delattr(self, 'h')
      
    def load_h5_file(self):
        filename = self.filename
        measurement_names,_ = get_group_name(filename, 'measurement')
        measurement_name = measurement_names[0]
        self.measurement_name = measurement_name
        
        if 'FLIR_NI' in measurement_name:
            available_datasets = self.open_FLIR_NI_h5_file() # sd is the number of available datasets
        
        elif 'PROCHIP' in measurement_name:
            available_datasets = self.open_PROCHIP_h5_file()
             
        else:
            raise(ValueError('Wrong measurement type'))
     
        sp,sz,sy,sx = self.imageRaw.shape
        
        self.settings['phases_number'] = sp 
        self.settings['dataset_index'] = 0
        #self.settings['frame_index'] = sz//2
        
        self.show_text(f'\nCorrectly opened dataset {0}/{available_datasets} \
                             \nwith {sp} phases and {sz} images')
        # self.settings.frame_index.change_min_max(vmin = 0, vmax = sz-1)
        self.settings.dataset_index.change_min_max(vmin = 0, vmax= available_datasets-1)
            
        
        
        # allow selection of the ROI clicking the mouse
        self.enableROIselection()
        #load measurement settings, if present in the h5 file
        for key in ['magnification','n','NA','pixelsize','wavelength']:
            val = get_h5_attr(filename, key, group = measurement_name) # Checks if the key is in the Scopefoundry measurement settings
            if len(val)>0 and hasattr(self.settings,key):
                new_value = val[0]
                self.settings[key] = new_value
                self.show_text(f'Updated {key} to: {new_value} ')        
    
    
    def open_FLIR_NI_h5_file(self):
        """Specific for ScopeFoudry FLIR NI measurement data
        """
        im, available_datasets = get_h5_dataset(self.filename)
        self.imageRaw = im[:,np.newaxis,...]
        return available_datasets
    
    
    def open_PROCHIP_h5_file(self):
        """Specific for ScopeFoudry PROCHIP measurement data
        with multichannel data in different datasets
        """
        available_datasets = self.set_dataset(idx = 0)
        return available_datasets

    
    def set_dataset(self,idx):
        """
        Opens the dataset specified in idx
        Specific for ScopeFoudry PROCHIP measurement data
        """
        if not hasattr(self,'measurement_name'):
            return 0
        
        if 'FLIR_NI' in self.measurement_name:
            available_datasets = 1
        
        elif 'PROCHIP' in self.measurement_name:   
            
            t_idx = f'/t{idx:04d}/'
            
            index_list, names = get_datasets_index_by_name(self.filename, t_idx)
            if len(index_list) <1:
                t_idx = f'/t{idx:01d}/' # fix for not ROI acquisitions before June 1st 2021. TODO remove
                index_list, names = get_datasets_index_by_name(self.filename, t_idx)
            stack,found = get_multiple_h5_datasets(self.filename, index_list)
            shape = stack.shape
            size_phases = shape[0]
            size_z = shape[1]
            if size_phases != 3 and size_phases != 7: # ==7 is for the future in PROCHIP  measurements 
                self.show_text(f'\nUnable to open dataset {idx}.\n')
                raise(ValueError)
            else:
                self.show_text(f'\nCorrectly opened dataset {idx}/{(found//size_phases)-1} \
                             \nwith {size_phases} phases and {size_z} images')
            self.imageRaw = stack
            self.settings['frame_index'] = size_z//2 # sets image to the central frame of the stack
            available_datasets = found//size_phases
        return available_datasets
    
    
    def set_frame(self,idx):
        if hasattr(self, 'imageRaw'):
            self.imageRaw = self.imageRaw # frame update is done by the imageRaw setter 
        
            
    def cutRoi(self):        
        if hasattr(self, 'imageRaw'):
            Ly =self.imageRaw.shape[-1]
            Lx =self.imageRaw.shape[-2]
            x = self.settings['roiX']
            y = self.settings['roiY']
            ROIsize = self.settings['ROI_size']
            x = max(min(x, Lx-ROIsize//2 ),ROIsize//2 )
            y = max(min(y, Ly-ROIsize//2 ),ROIsize//2 )
            xmin = x - ROIsize//2
            ymin = y - ROIsize//2    
            self.imageRaw = self.imageRaw [:,:,xmin:xmin+ROIsize, ymin:ymin+ROIsize] 
            self.settings['selectROI'] = False
            self.show_text(f'ROI set to shape: {self.imageRaw.shape}')
      
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
            ROIsize = self.settings['ROI_size']
            Ly =self.imageRaw.shape[-1]
            Lx =self.imageRaw.shape[-2]
                
            if self.settings['selectROI'] and (Lx,Ly)!=(ROIsize,ROIsize):
                event.accept()  
                pos = event.pos()
                x = int(pos.x()) #pyqtgraph is transposed
                y = int(pos.y())
                x = max(min(x, Lx-ROIsize//2 ),ROIsize//2 )
                y = max(min(y, Ly-ROIsize//2 ),ROIsize//2 )
                self.settings['roiX']= x
                self.settings['roiY']= y
                self.cutRoi()
            self.settings['selectROI'] = False
        
        self.imvRaw.getImageItem().mouseClickEvent = click
        self.imvWF.getImageItem().mouseClickEvent = click
        self.settings['selectROI'] = False

    def load_tif_file(self):
        filename = self.filename
        im = np.single(tif.imread(filename))
        self.imageRaw = im[:,np.newaxis,...]
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
                         "Laser power:\t {} mW".format(self.filetitle, self.imageRaw.shape, \
                                                                configSet["wavelength"], \
                                                                self.exposuretime, self.laserpower)
            self.show_text(txtDisplay)

        except:
            self.show_text("No information about this measurement.")
       
    def loadFile(self):
           
        filename, _ = QFileDialog.getOpenFileName(directory = self.app.settings['save_dir'])
        self.filetitle = Path(filename).stem
        self.filepath = os.path.dirname(filename)
        self.filename = filename
        
        if filename.endswith('.tif') or filename.endswith('.tiff'):
            self.load_tif_file()
        elif filename.endswith('.h5') or filename.endswith('.hdf5'):
            self.load_h5_file()
        else:
            raise(OSError('Invalid file type'))

        
        self.setReconstructor()
        self.h._allocate_arrays()
                
    def calculate_WF_image(self, img):
        imageWF = np.mean(img, axis=0)
        return imageWF
        
    def calculate_spectrum(self, img):
        """
        Calculates power spectrum of the image
        """
        epsilon = 1e-6
        ps = np.log((np.abs(fftshift(fft2(img))))**2+epsilon) 
        return ps 
        
    def reset_processor(self,*args):
        self.isCalibrated = False
        self.stop_sim_processor()
        self.start_sim_processor()
            
    @add_timer
    def calibration(self):    
        self.setReconstructor()
        frame_index = self.settings['frame_index']
        if self.settings['gpu']:
            self.h.calibrate_cupy(self.imageRaw[:,frame_index,:,:], self.isFindCarrier)       
        else:
            self.h.calibrate(self.imageRaw[:,frame_index,:,:],self.isFindCarrier)          
        self.isCalibrated = True
        self.find_phaseshifts() 
        self.show_text('Calibration completed')
        # update wiener filter image
        if not hasattr(self, 'wienerImv'):
            self.wienerImv = pg.ImageView()
            self.wienerImv.ui.roiBtn.hide()
            self.wienerImv.ui.menuBtn.hide()
            self.ui.wienerLayout.addWidget(self.wienerImv)
        self.wienerImv.setImage(self.h.wienerfilter, autoRange=True, autoLevels=True, autoHistogramRange=True)
        self.showCalibrationTable() 
             
    @add_timer  
    def standard_reconstruction(self):
        frame_index = self.settings['frame_index']
        self.setReconstructor()
        if self.isCalibrated:
            
            if self.settings['gpu']:
                self.imageSIM = self.h.reconstruct_cupy(self.imageRaw[:,frame_index,:,:])

            elif not self.settings['gpu']:
                self.imageSIM = self.h.reconstruct_rfftw(self.imageRaw[:,frame_index,:,:])
        else:
            self.calibration()
            if self.isCalibrated:
                self.standard_reconstruction()
          
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
        if not hasattr(self, 'phasesPlot'):
            self.phasesPlot = pg.PlotWidget()
            self.ui.phasesLayout.addWidget(self.phasesPlot)
            
        if self.settings.phases_number.val == 7:
            self.find_7phaseshifts()
        if self.settings.phases_number.val == 3:
            self.find_3phaseshifts()
        
    
    def find_7phaseshifts(self):    
        self.phaseshift = np.zeros((4,7))
        self.expected_phase = np.zeros((4,7))
        frame_index = self.settings['frame_index']
    
        for i in range (3):
            phase, _ = self.h.find_phase(self.h.kx[i],self.h.ky[i],self.imageRaw[:,frame_index,:,:])
            self.expected_phase[i,:] = np.arange(7) * 2*(i+1) * np.pi / 7
            self.phaseshift[i,:] = np.unwrap(phase - self.expected_phase[i,:]) + self.expected_phase[i,:] - phase[0]
    
        self.phaseshift[3] = self.phaseshift[2]-self.phaseshift[1]-self.phaseshift[0]
        
        self.phasesPlot.clear()
        for idx in range(len(self.phaseshift)):
            self.phasesPlot.plot(self.phaseshift[idx], symbol = '+')
            pen = pg.mkPen(color='r')
            self.phasesPlot.plot(self.expected_phase[idx], pen = pen)
            
    def find_3phaseshifts(self):
        frame_index = self.settings['frame_index']
        phase, _ = self.h.find_phase(self.h.kx,self.h.ky,self.imageRaw[:,frame_index,:,:])
        expected_phase = np.arange(0,2*np.pi ,2*np.pi / 3)
        phaseshift= np.unwrap(phase - expected_phase) + expected_phase - phase[0]
        error = phaseshift-expected_phase
        self.phasesPlot.clear()
        self.phasesPlot.plot(phaseshift, symbol = '+')
        pen = pg.mkPen(color='r')
        self.phasesPlot.plot(expected_phase, pen=pen)
        self.phasesPlot.plot(error, symbol = 'o')
        self.show_text(f"\nExpected phases: {expected_phase}\
                         \nMeasured phases: {phaseshift}\
                         \nError          : {error}\n")
     
    def show_text(self, text):
        self.ui.MessageBox.insertPlainText(text+'\n')
        self.ui.MessageBox.ensureCursorVisible()
        print(text)    
        
    def showCalibrationTable(self):
        if self.settings.phases_number.val == 3:
            self.show3CalibrationTable()
        elif self.settings.phases_number.val == 7:
            self.show7CalibrationTable()
    
    def show3CalibrationTable(self):
        def table_item(element):
            return QTableWidgetItem(str(element).lstrip('[').rstrip(']'))
        
        table = self.ui.currentTable
        table.setColumnCount(2)
        table.setRowCount(6)
        table.setItem(0, 0, table_item('[kx_in]')) #TODO check if this is necessary
        table.setItem(0, 1, table_item(self.kx_input[0]))
        
        table.setItem(1, 0, table_item('[ky_in]'))              
        table.setItem(1, 1, table_item(self.ky_input[0]))
        
        table.setItem(2, 0, table_item('[kx]'))             
        table.setItem(2, 1, table_item(self.h.kx))
        
        #
        table.setItem(3, 0, table_item('[ky]'))              
        table.setItem(3, 1, table_item(self.h.ky))
        
        #
        table.setItem(4, 0, table_item('[phase]'))  
        table.setItem(4, 1, table_item(self.h.p))
        
        #
        table.setItem(5, 0, table_item('[amplitude]'))  
        table.setItem(5, 1, table_item(self.h.ampl))
          
    
    def show7CalibrationTable(self):
        def table_item(element):
            return QTableWidgetItem(str(element).lstrip('[').rstrip(']'))
            
        table = self.ui.currentTable
        table.setColumnCount(4)
        table.setRowCount(6)
        
        table.setItem(0, 0, table_item('[kx_in]'))
        table.setItem(0, 1, table_item(self.kx_input[0])) #TODO check if this is necessary
        table.setItem(0, 2, table_item(self.kx_input[1]))
        table.setItem(0, 3, table_item(self.kx_input[2]))
        
        table.setItem(1, 0, table_item('[ky_in]'))    #TODO check if this is necessary          
        table.setItem(1, 1, table_item(self.ky_input[0]))
        table.setItem(1, 2, table_item(self.ky_input[1]))
        table.setItem(1, 3, table_item(self.ky_input[2]))

        table.setItem(2, 0, table_item('[kx]'))             
        table.setItem(2, 1, table_item(self.h.kx[0]))
        table.setItem(2, 2, table_item(self.h.kx[1]))
        table.setItem(2, 3, table_item(self.h.kx[2]))
        #
        table.setItem(3, 0, table_item('[ky]'))              
        table.setItem(3, 1, table_item(self.h.ky[0]))
        table.setItem(3, 2, table_item(self.h.ky[1]))
        table.setItem(3, 3, table_item(self.h.ky[2]))
        #
        table.setItem(4, 0, table_item('[phase]'))  
        table.setItem(4, 1, table_item(self.h.p[0]))
        table.setItem(4, 2, table_item(self.h.p[1]))
        table.setItem(4, 3, table_item(self.h.p[2]))
        #
        table.setItem(5, 0, table_item('[amplitude]'))  
        table.setItem(5, 1, table_item(self.h.ampl[0]))
        table.setItem(5, 2, table_item(self.h.ampl[1]))
        table.setItem(5, 3, table_item(self.h.ampl[2]))

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
    
    app.settings_load_ini(".\\Settings\\PROCHIP_Analysis.ini")
    sys.exit(app.exec_())