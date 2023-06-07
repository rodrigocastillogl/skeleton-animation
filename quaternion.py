import numpy as np


def expmap_to_quaternion(exp_map):

    """
    Convert axis-angle rotations (exponential maps) to quaternions.
    Formula from "Practical Parameterization of Rotations Using the Exponential Map".
    Input
    -----
        * exp_map : axis-angle rotations array (Frames, Joints, 3)
    Output
    ------
        * quaternions : quaternions array (Frames, Joints, 4)
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


def quaternion_fix(q):
    """
    Enforce quaternion continuity by selecting the representation (q or -q) with minimal
    euclidean distance (equivalently, maximal dot product) between two consecutive frames.
    Input
    -----
        * q : quaternions array (Frames, Joints, 4). 
    Output
    ------
        * q_fix : quaternions array (Frames, Joints, 4).
    """
    
    assert q.shape[-1] == 4
    
    result = q.copy()
    dot_products = np.sum( q[1:] * q[:-1], axis = 2 )
    mask = ( np.cumsum( (dot_products < 0) , axis = 0) % 2 ).astype(bool)
    result[1:][mask] *= -1

    return result

def quaternion_product(q1, q2):
    """
    Hamilton product of two quaternion arrays q1 and q2.
    Input
    -----
        * q1, q2 : quaternions array (Frames, 4).
    Output
    ------
        * prod : hamilton product (Frames, 4).
    """

    assert q1.shape == q2.shape
    assert q1.shape[-1] == 4

    # outer product between q1 and q2
    M = q1.reshape((-1, 4, 1)) @ q2.reshape(-1, 1, 4)
    
    prod = np.zeros( (q1.shape[0], 4) )
    prod[:,0] = M[:,0,0] - M[:,1,1] - M[:,2,2] - M[:,3,3]
    prod[:,1] = M[:,0,1] + M[:,1,0] - M[:,2,3] + M[:,3,2]
    prod[:,2] = M[:,0,2] + M[:,1,3] + M[:,2,0] - M[:,3,1]
    prod[:,3] = M[:,0,3] - M[:,1,2] + M[:,2,1] + M[:,3,0]
    
    return prod


def quaterion_rotation(q, v):
    """
    Rotation of a vectors array v by a quaternions array q. 
    Input
    -----
        * q : quaternions array (Frames, 4)
        * v : vectors array (Frames, 3)
    Output
    ------
        *  v_rot : rotated vectors (Frames, 3)
    """

    qvec = q[:,1:]
    qw = q[:,0].reshape(-1,1)

    qv = np.cross( qvec, v)
    qqv = np.cross( qvec, qv)

    return v + 2 * ( qw * qv + qqv )