import numpy as np
import os
import sys
import h5py

class writer(object):
    
    def __init__(self,prefix,Lx,Ly,base_output_folder=None,suffix=None):
        self.prefix = prefix
        self.Lx = Lx
        self.Ly = Ly
        
        self.path = self.get_file_path(base_output_folder,suffix)

        self.groups = ['ic', 'avg_conc', 'energy', 'order_parameter']
        
    def get_file_path(self, base_output_folder=None,suffix=None):
        if base_output_folder == None:
            os.chdir('/home/ray/git-projects/modified_ch/')
            base_output_folder = "output"
        if suffix != None:
            base_output_folder += "/" + suffix
        base_output_folder += "/"
        base_output_name = "%s_Lx=%.4f_Ly=%.4f" %(self.prefix,self.Lx,self.Ly)
        hdf5_format = '.h5'
        return base_output_folder + base_output_name + hdf5_format
        
        
    def create_output_file(self, options):

        if os.path.exists(self.path):
            os.rename(self.path, self.path+'_old')

        file = h5py.File(self.path, 'a')

        for group in self.groups:
            # check if groups have been created
            # if not created, create empty groups
            if not (group in file):
                file.create_group(group,track_order=True)

        for key in options:
            if type(options[key]) != str:
                file.attrs.create(key,options[key])
        file.close()
        
    def write_data(self, it, data):
        file = h5py.File(self.path , 'r+')
        
        for group, datum in zip(self.groups,data):
            name = group + '_%i' %it
            if isinstance(datum, (list, tuple, np.ndarray)):
                file.create_dataset(group + '/' + name, data=datum ,chunks=True, compression='gzip', compression_opts=4, dtype=np.float64)
            else:
                file.create_dataset(group + '/' + name, data=datum , dtype=np.float32)
        file.close()
        
    def get_dataset(self,group,it):
        file = h5py.File(self.path, 'r')
        
        arr = np.copy(file[group][group + '_' + str(it)])
        
        file.close()
        return arr