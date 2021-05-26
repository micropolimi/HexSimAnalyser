"""
Written by Andrea Bassi (Politecnico di Milano) 10 August 2018
to find the location of datasets in a h5 file and to extact attributes.

"""

import h5py
import numpy as np

def get_h5_dataset(fname, dataset_index=0):
        """
        Finds the datasets in HDF5 file.
        Returns the dataset specified by the dataset_index.
        """
        try:
            f = h5py.File(fname,'r')
            name,shape,found = _get_h5_dataset(f, name=[], shape=[], found=0)    
            #assert found > 0, "Specified h5 file does not exsist or have no datasets"    
            if dataset_index >= found:    
                dataset_index = 0
            data = np.single(f[name[dataset_index]])
        
        finally:
            f.close()
        return data

def _get_h5_dataset(g, name, shape, found) :
        """
        Extracts the dataset location (and its shape).
        It is operated recursively in the h5 file.
        """
       
        if isinstance(g,h5py.Dataset):   
            found += 1
            name.append(g.name)
            shape.append(g.shape)
            
        if isinstance(g, h5py.File) or isinstance(g, h5py.Group):
            for key,val in dict(g).items() :
                
                name,shape,found = _get_h5_dataset(val,name,shape,found)
                 
        return name,shape,found 
    

def get_h5_attr(fname, attr_name):
    """
    Finds an attribute in with the specified names, in a h5 file
    Returns a dictionary with name as key and the attribute value
    Raise a Warning if more that one attribute with the same name is found.
    
    """
    try:
        f = h5py.File(fname,'r')
        attr_dict, found = _get_h5_attr(f, attr_name, value=[], found=0)
        if found > 1:
            print(f'Warning: more than one attribute with name {attr_name} found in h5 file')
    finally:
        f.close()        
    return attr_dict  
        
        
def _get_h5_attr(g, attr_name='some_name', value=[], found=0):
    """
    Returns the attribute's key and value in a dictionary.
    It is operated recursively in the h5 file. 
    """
    if isinstance(g, h5py.File) or isinstance(g, h5py.Group):
        
        for key,val in dict(g).items() :
            for attr_key, attr_val in val.attrs.items():
                if attr_key == attr_name:
                    found +=1
                    value.append(attr_val)
            value, found = _get_h5_attr(val, attr_name, value, found)
    
    return value, found
                  
            
"""The following is only to test the functions.

"""     
if __name__ == "__main__" :
    
        import sys
        import pyqtgraph as pg
        import qtpy.QtCore
        from qtpy.QtWidgets import QApplication
        
        # this h5 file must contain a dataset composed by an array or an image
        file_name='D:\\data\\PROCHIP\\temp\\test1.h5'
        
        stack = get_h5_dataset(file_name)    
    
        #attr= get_h5_attr(file_name,'n')
        for key in ['magnification','n','NA','pixelsize','wavelength']:
                val = get_h5_attr(file_name, key)
                print(val)
    
        pg.image(stack, title="Stack of images")        
               
        #keeps the window open running a QT application
        if sys.flags.interactive != 1 or not hasattr(qtpy.QtCore, 'PYQT_VERSION'):
            QApplication.exec_()
                          
   
        sys.exit ( "End of test")