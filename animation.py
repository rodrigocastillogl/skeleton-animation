from prepare_data import data
import numpy as np
import matplotlib.pyplot as plt

def quaterion_rotation(q, v):
    """
    Hamilton product of two quaternions.
    Input
    -----
        * q : rotation quaternion
        * v : vector
    Output
    ------
        *  v_rot : rotated vector
    """

    qvec = q[:,1:]
    qw = q[:,0]

    print(qvec.shape)
    print(qw.shape)

    qv = np.cross( qvec, v)
    qqv = np.cross( qvec, qv)

    return v + 2 * ( qw * qv + qqv ) 


class Skeleton():

    def __init__( self, data, offsets, parents, joints_left, joints_right ):
        """
        Constructor
        Input
        -----
            * data: rotations data.
            * offsets: skeleton offsets.
            * parents: skeleton hierarchical structure.
            * joints_left: left joints indices.
            * joints_right: rights joints inidices.
        Output
        ------
            None
        """

        self.data = data
        self.offsets = offsets
        self.parents = parents
        self.joints_left = joints_left
        self.joints_right = joints_right

    def compute_positions(self):
        """
        Compute world positions with forward kinematics.
        Input
        -----
            None
        Output
        ------
            None
        """

        n_frames = self.data.shape[0]
        n_joints = self.data.shape[1]

        self.positions = np.zeros( n_frames, n_joints , 3)
        self.positions[:,0,:] = data['trajectory']

        for frame in range(n_frames):
            
            for joint in range(n_joints):
                
                parent  = self.parents[joint]

                while parent != -1:

                    parent = -1

            



offsets = [ [   0.      ,    0.      ,    0.      ],
            [-132.948591,    0.      ,    0.      ],
            [   0.      , -442.894612,    0.      ],
            [   0.      , -454.206447,    0.      ],
            [   0.      ,    0.      ,  162.767078],
            [   0.      ,    0.      ,   74.999437],
            [ 132.948826,    0.      ,    0.      ],
            [   0.      , -442.894413,    0.      ],
            [   0.      , -454.20659 ,    0.      ],
            [   0.      ,    0.      ,  162.767426],
            [   0.      ,    0.      ,   74.999948],
            [   0.      ,    0.1     ,    0.      ],
            [   0.      ,  233.383263,    0.      ],
            [   0.      ,  257.077681,    0.      ],
            [   0.      ,  121.134938,    0.      ],
            [   0.      ,  115.002227,    0.      ],
            [   0.      ,  257.077681,    0.      ],
            [   0.      ,  151.034226,    0.      ],
            [   0.      ,  278.882773,    0.      ],
            [   0.      ,  251.733451,    0.      ],
            [   0.      ,    0.      ,    0.      ],
            [   0.      ,    0.      ,   99.999627],
            [   0.      ,  100.000188,    0.      ],
            [   0.      ,    0.      ,    0.      ],
            [   0.      ,  257.077681,    0.      ],
            [   0.      ,  151.031437,    0.      ],
            [   0.      ,  278.892924,    0.      ],
            [   0.      ,  251.72868 ,    0.      ],
            [   0.      ,    0.      ,    0.      ],
            [   0.      ,    0.      ,   99.999888],
            [   0.      ,  137.499922,    0.      ],
            [   0.      ,    0.      ,    0.      ] ]

parents = [ -1,  0,  1,  2,  3,  4,  0,  6,  7,  8, 
            9,  0, 11, 12, 13, 14, 12, 16, 17, 18,
            19, 20, 19, 22, 12, 24, 25, 26, 27, 28,
            27, 30 ]

joints_left = [ 1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31 ]
joints_right = [ 6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23 ]

"""
for subject in data.keys():

    print('subject: ' + subject)

    for action in data[subject].keys():
        print('  action: ' + action )
        print( '    trajectory: ', data[subject][action]['trajectory'].shape   ,
               '    quaternions: ', data[subject][action]['quaternions'].shape )

    print()
"""

x = data['S9']['directions_1']['quaternions'][:,0,:]
v = np.zeros( (x.shape[0], 3) )
v = np.zeros( (x.shape[0], 3) )

print(x.shape)
print(v.shape)
