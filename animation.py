from prepare_data import data
import numpy as np
import matplotlib.pyplot as plt
from quaternion import *

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

        # save data
        self.quaternions = data['quaternions'].copy()
        self.trajectory = data['trajectory'].copy()

        # - number of frames / number of joints -
        self.n_frames = self.quaternions.shape[0]
        self.n_joints = self.quaternions.shape[1]
        # ---------------------------------------

        # skeleton parameters
        self.offsets = offsets.copy()
        self.parents = parents
        self.joints_left = joints_left
        self.joints_right = joints_right
        self.compute_has_children()
    
    
    def compute_has_children(self):
        """
        Compute has_children object attribute.
        """
        self.has_children = [False] * self.n_joints
        for joint in self.parents:
            self.has_children[joint] = True
    

    def compute_positions(self):
        """
        Compute world positions with forward kinematics.
        """

        n_frames = self.quaternions.shape[0]
        n_joints = self.quaternions.shape[1]

        # world positions
        p = np.zeros( (n_frames, n_joints, 3) )

        # the first joint is the root 
        p[:,0,:] = self.trajectory

        for joint in range(1,n_joints):

            parent  = self.parents[joint]
            l = []
            while parent != -1:
                l.append(parent)
                parent = self.parents[parent]
            l = l[:-1]

            r = self.quaternions[:,0,:]
            if len(r):
                for i in reversed(l):
                    r = quaternion_product( r, self.quaternions[:,i,:] )

            p[:,joint,:] = p[:,self.parents[joint],:] + quaterion_rotation( r,
                           np.repeat( np.expand_dims( self.offsets[joint,:], axis = 0),
                           n_frames, axis = 0) )
            print(self.parents[joint], joint)



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
offsets = np.array(offsets)

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

s = Skeleton( data    = data['S1']['walking_1'],
              offsets = offsets                   ,
              parents = parents                   ,
              joints_left  = joints_left          ,
              joints_right = joints_right         )

# s.compute_positions()

# pp = s.positions[500,:,:]

"""
fig = plt.figure( figsize = (9,6) )
ax = plt.axes( projection = '3d' )
ax.set_xlabel('x'), ax.set_ylabel('-z'), ax.set_zlabel('y')
ax.set_xlim([-1000, 1000]), ax.set_ylim([-1000, 1000]), ax.set_zlim([-1000, 1000])

plt.plot(pp[0,0], -pp[0,2], pp[0,1], 'o', c = 'k', markersize = 3)

for i in range(1, pp.shape[0]):
    plt.plot( pp[i,0], -pp[i,2],
	          pp[i,1], 'o', c  = 'k', markersize = 3)
    plt.plot( [pp[parents[i], 0], pp[i, 0]]    ,
	          [-pp[parents[i], 2], -pp[i, 2] ] ,
	          [pp[parents[i], 1], pp[i, 1] ]   ,
	          c  = 'k', linewidth = 0.5        )

plt.tight_layout()
plt.show()
"""