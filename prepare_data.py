import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import zipfile
import numpy as np
from urllib.request import urlretrieve

def expmap_to_quaternion(exp_map):

    """
    Convert axis-angle rotations (exponential maps) to quaternions.
    Formula from "Practical Parameterization of Rotations Using the Exponential Map".
    """
    assert exp_map.shape[-1] == 3
    
    original_shape = exp_map.shape
    exp_map = exp_map.reshape(-1, 3)
    
    # angle / norm
    theta = np.linalg.norm(exp_map, axis = 1).reshape(-1, 1)

    # quaternion
    w = np.cos( 0.5 * theta ).reshape(-1, 1)
    xyz = 0.5 * np.sinc(0.5 * theta/np.pi) * exp_map

    return np.concatenate((w, xyz), axis = 1).reshape( original_shape[0] , -1, 4)


def prepare_data():

    output_directory = 'dataset'
    h36m_dataset_url = 'http://www.cs.stanford.edu/people/ashesh/h3.6m.zip'
    
    if output_directory not in os.listdir():
        
        # create output directory
        os.makedirs(output_directory)
        
        # download data
        print('Downloading Human3.6M dataset (it may take a while)...')
        zip_path = output_directory + '/h3.6m.zip'
        urlretrieve(h36m_dataset_url, zip_path)
        
        # extract fata from zip file
        print('Extracting Human3.6M dataset ...')
        with zipfile.ZipFile(zip_path, 'r') as archive:
            archive.extractall(output_directory)
        
        
        # remove zip file
        os.remove(zip_path)

    data_path = output_directory + '/h3.6m/dataset'
    subjects_list = sorted( os.listdir(data_path) )

    data = {}
    for subject in subjects_list:
        
        # add subject to data
        if subject not in data:
            data[subject] = {}

        files_list = sorted( os.listdir(data_path + '/' + subject ) )
        
        for file in files_list:

            action = file[:-4]

            # read data
            df = pd.read_csv( data_path + '/' + subject + '/' + file, header = None)
            arr = df.to_numpy()
            
            # trajectory
            trajectory = arr[:,:3]

            # transform to quaternions
            exp_map = arr[:,3:]
            exp_map = exp_map.reshape( exp_map.shape[0], -1 , 3)
            quaternions = expmap_to_quaternion( exp_map )

            # add action to subject
            if action not in data[subject]:
                data[subject][action] = {}
            
            data[subject][action]['trajectory'] = trajectory
            data[subject][action]['quaternions'] = quaternions

    return data

data = prepare_data()