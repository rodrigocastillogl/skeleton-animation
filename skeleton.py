import numpy as np
from quaternion import quaternion_product, quaterion_rotation

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
        self.compute_children()

        # joints world postion
        self.positions = np.zeros( (self.n_frames, self.n_joints, 3) )
        self.compute_positions( 0, self.quaternions[:,0,:], self.trajectory )


    def compute_children(self):
        """
        Compute children object attribute
        """
        self.children = []
        for j in range(self.n_joints):
            self.children.append([])
        for j in range(self.n_joints):
            if self.parents[j] != -1:
                self.children[self.parents[j]].append(j)
    
    
    def compute_positions(self, root, q, p):
        """
        Recursive function to compute world positions.
        Input
        -----
            * root : root index.
            * q : root rotation as quaternion.
            * p : root position as vector.
        Output
        ------
            None
        """

        for j in self.children[root]:
            new_p = p + quaterion_rotation( q, np.repeat( np.expand_dims( self.offsets[j,:], axis = 0),
                                                      self.n_frames, axis = 0 ) )
            new_q = quaternion_product( q, self.quaternions[:,j,:] )
            self.positions[:,j,:] = new_p
            self.compute_positions(j, new_q, new_p)