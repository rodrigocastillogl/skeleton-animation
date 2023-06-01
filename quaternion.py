import numpy as np

def quaternion_product(q1, q2):
    """
    Hamilton product of two quaternion arrays q1 and q2.
    Input
    -----
        * q1, q2 : quaternions
    Output
    ------
        * prod : quaterion product.
    """

    assert q1.shape == q2.shape
    assert q1.shape[-1] == 4

    M = q1.reshape((-1, 4, 1)) @ q2.reshape(-1, 1, 4)
    
    prod = np.zeros( (q1.shape[0], 4) )
    prod[:,0] = M[:,0,0] - M[:,1,1] - M[:,2,2] - M[:,3,3]
    prod[:,1] = M[:,0,1] + M[:,1,0] - M[:,2,3] + M[:,3,2]
    prod[:,2] = M[:,0,2] + M[:,1,3] + M[:,2,0] - M[:,3,1]
    prod[:,3] = M[:,0,3] - M[:,1,2] + M[:,2,1] + M[:,3,0]
    
    return prod

def quaterion_rotation(q, v):
    """
    Rotation of a vector array v by a quaternion array q. 
    Input
    -----
        * q : rotation quaternion
        * v : vector
    Output
    ------
        *  v_rot : rotated vector
    """

    qvec = q[:,1:]
    qw = q[:,0].reshape(-1,1)

    qv = np.cross( qvec, v)
    qqv = np.cross( qvec, qv)

    return v + 2 * ( qw * qv + qqv )