"""
Written by Andrea Bassi (Politecnico di Milano) 10 August 2018
to find the location of datasets in a h5 file and to extact attributes.

"""

import h5py
import numpy as np

 
def get_datasets_index_by_name(fname, match="/t0000/"):
        f = h5py.File(fname,'r')
        names,shapes,found = _get_h5_datasets(f, name=[], shape=[], found=0)    
        assert found > 0, "Specified h5 file does not exsist or have no datasets"    
        
        index_list = []
        names_list = []
        for idx,name in enumerate(names):
            if match in name:
                index_list.append(idx)
                names_list.append(name)
        f.close()        
        return index_list, names_list
        


def get_multiple_h5_datasets(fname, idx_list):
        """
        Finds datasets in HDF5 file.
        Returns the datasets specified by the dataset_index in a 16bit, n-dimensional numpy array
        If the size of the first dimension of the stack is different, between the datasets, the minimum size is choosen 
        """
        f = h5py.File(fname,'r')
        names,shapes,found = _get_h5_datasets(f, name=[], shape=[], found=0)    
        assert found > 0, "Specified h5 file does not exsist or have no datasets"    
        assert max(idx_list) < found, "Specified h5 file have less datasets than requested"    
        
        data_shape = shapes[idx_list[0]] 
        size0 = data_shape[0]
        for idx in idx_list[1::]:
             size0 = min(size0,shapes[idx][0])   
        data = np.zeros([len(idx_list), size0, *data_shape[1::]])
        for key,idx in enumerate(idx_list):
            stack = np.single(f[names[idx]])
            data [key,...] = stack[0:size0,...]
        f.close()
        return data, found


def get_h5_dataset(fname, dataset_index=0):
        """
        Finds the datasets in HDF5 file.
        Returns the 3D dataset specified by the dataset_index as a 16bit array.
        """
        f = h5py.File(fname,'r')
        name,shape,found = _get_h5_datasets(f, name=[], shape=[], found=0)    
        assert found > 0, "Specified h5 file does not exsist or have no datasets"    
        assert dataset_index < found, "Specified h5 datset does not exist"    
        data = np.single(f[name[dataset_index]])
        f.close()
        return data, found
    
    
def _get_h5_datasets(g, name, shape, found) :
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
                
                name,shape,found = _get_h5_datasets(val,name,shape,found)
                 
        return name,shape,found 
    

def get_h5_attr(fname, attr_key = 'magnification', group = None):
    """
    Finds an attribute with the specified names, in a specified group of a h5 file
    Returns a list attribute values with the specified key
    Raise a Warning if more that one attribute with the same name is found.
    
    """
    attr_dict =[]
    try:
        f = h5py.File(fname,'r')
        attr_dict, found = _get_h5_attr(f, attr_key, group, value=[], found=0)
        if found > 1:
            print(f'Warning: more than one attribute with name {attr_key} found in h5 file')
    finally:
        f.close()        
    return attr_dict  
        
        
def _get_h5_attr(g, attr_key ='magnification', group=None, value=[], found=0):
    """
    Returns the attribute's key and value in a list .
    It is operated recursively in the h5 file. 
    """
    if isinstance(g, h5py.File) or isinstance(g, h5py.Group):
        
        for key,val in dict(g).items() :
            
            for sub_key, sub_val in val.attrs.items():
                if sub_key == attr_key:
                    if group is None or group in val.name:
                        found +=1
                        value.append(sub_val)
                        #print(key,val.name)
                    
            value, found = _get_h5_attr(val, attr_key, group, value, found)
    
    return value, found
 


def get_group_name(fname, group_key = 'measurement'):
    """
    Returns the keys of the subgroups for a specified group
    Used to get ScopeFoundry measurement name
    
    """
    groups = []
    found=0
    try :
        f = h5py.File(fname,'r')
        groups, found = _get_h5_groups(f, group_key, groups, found)
    finally:
        f.close() 
    
    if found > 1:
            print(f'Warning: more than one group with name {group_key} found in h5 file')
        
    return groups, found 
   
 
    
def _get_h5_groups(g, group_key = 'measurement', groups=[], found=0):
     if isinstance(g, h5py.File) or isinstance(g, h5py.Group):
        
        for key,val in dict(g).items() :
            if key == group_key:
                for subkey,subval in dict(val).items() :
                    
                    found +=1
                    groups.append(subval.name)
                                            
            groups, found = _get_h5_groups(val, group_key, groups, found)
    
     return groups, found   



               
            
"""The following is only to test the functions.

"""     
if __name__ == "__main__" :
    
        import sys
        import pyqtgraph as pg
        import qtpy.QtCore
        from qtpy.QtWidgets import QApplication
        import time
        # this h5 file must contain a dataset composed by an array or an image
        file_name='D:\\Data\\PROCHIP\\TEMP\\test_hex_sim.h5'
        
        stack = get_h5_dataset(file_name)    
        idx = 1
        
        index_list, index_names = get_datasets_index_by_name(file_name, '/t0002/')
        #print(index_list)
        #print(index_names)
        
        # hyperstack,found = get_multiple_h5_datasets(file_name, index_list)    
        
        # print(found)
        # print(hyperstack.shape)
        
        
        # attr= get_h5_attr(file_name,'n')
        # for key in ['magnification','n','NA','pixelsize','wavelength']:
        #         val = get_h5_attr(file_name, key, group = None)
        #         print(val)
            
        # key = 'NA'    
        # val = get_h5_attr(file_name,key)
        # print(val)
        
        
        groups,found = get_group_name(file_name, group_key = 'measurement')
        print(groups)
        
        
        #pg.image(hyperstack[0,:,:,:], title="Stack of images")        
               
        #keeps the window open running a QT application
        #if sys.flags.interactive != 1 or not hasattr(qtpy.QtCore, 'PYQT_VERSION'):
        #    QApplication.exec_()
                          
   
        #sys.exit ( "End of test")