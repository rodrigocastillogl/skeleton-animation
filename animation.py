import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
from mpl_toolkits.mplot3d import Axes3D

from skeleton import Skeleton
from prepare_data import data


def plot_frame(skeleton, frame, name = 'show'):
    """
    Plot skeleton configuration in a frame.
    Input
    -----
        * skeleton : Skeleton object
        * frame    : frame index
        * name     : name to save figure (if 'show' figure displayed).
    Output
    ------
        None
    """
    
    # heuristic from https://github.com/facebookresearch/QuaterNet.
    radius = np.max( skeleton.offsets.squeeze() ) * 10

    print(radius)

    # create figure
    fig = plt.figure( figsize = (6,6) )
    ax = plt.axes( projection = '3d' )

    # parameters
    ax.view_init( elev = 20.0 , azim = 30.0 )
    ax.set_xlim3d( [ -radius/2, radius/2 ] )
    ax.set_zlim3d( [ -radius/2, radius/2 ] )
    ax.set_ylim3d( [ -radius/2, radius/2 ] )
    ax.set_aspect('equal')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    
    for j in range( 1, skeleton.n_joints ):
        
        if skeleton.children[j]:
            
            # select color
            c = 'r' if j in joints_right else 'k'
            
            # plot joint
            plt.plot( skeleton.positions[frame, j,0], skeleton.positions[frame, j,1] ,
                      skeleton.positions[frame, j,2], 'o', c  = c, markersize = 1 , zdir = 'y' )

            # plot segment
            plt.plot( [ skeleton.positions[ frame, skeleton.parents[j], 0], skeleton.positions[ frame, j, 0] ] ,
	                  [ skeleton.positions[ frame, skeleton.parents[j], 1], skeleton.positions[ frame, j, 1] ] ,
                      [ skeleton.positions[ frame, skeleton.parents[j], 2], skeleton.positions[ frame, j, 2] ] ,
                      c  = c , linewidth = 1.0 , zdir = 'y' )
            
    plt.tight_layout()

    if name == 'show':
        plt.show()
    else :
        plt.savefig()


def animation(skeleton, fps, name = 'show', bitrate = 1000):
    """
    Render or show an animation.
    Input
    -----
        * skeleton : Skeleton object.
        * fps  : frame rate (frames per second).
        * name : output file name
                 The supported output modes are:
                    - 'show' : display an interactive figure.
                    - 'html' : HTML5 video.
                    - 'mp4'  : h264 video (requires ffmpeg).
                    - 'gif'  : gif file (requires imagemagick).
        * bitrate : ...
    Output
    ------
        None
    """

    x = 0
    y = 1
    z = 2

    
    # heuristic from https://github.com/facebookresearch/QuaterNet.
    radius = np.max( skeleton.offsets.squeeze() ) * 10
    
    skeleton_parents = skeleton.parents

    # Figure parameters
    plt.ioff()

    fig = plt.figure( figsize = (6, 6) )
    ax = fig.add_subplot( 1, 1, 1, projection = '3d' )
    ax.view_init( elev = 20.0, azim = 30.0 )
    ax.set_xlim3d([-radius/2, radius/2])
    ax.set_zlim3d([0, radius])
    ax.set_ylim3d([-radius/2, radius/2])
    ax.set_aspect('equal')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    # ax.dist = 7.5

    lines = []
    markers = []
    initialized = False
    
    camera_pos = skeleton.trajectory
    data = skeleton.positions.copy()
    height_offset = np.min( data[:, :, 1].squeeze() ) 
    data = skeleton.positions.copy()
    data[:, :, 1] -= height_offset
    
    def update(frame):
        nonlocal initialized
        
        ax.set_xlim3d([-radius/2 + camera_pos[frame, 0], radius/2 + camera_pos[frame, 0]])
        ax.set_ylim3d([-radius/2 + camera_pos[frame, 1], radius/2 + camera_pos[frame, 1]])

        positions_world = data[frame]
        
        for j in range(positions_world.shape[0]):
            
            if skeleton_parents[j] == -1:
                continue

            if not initialized:
                
                c = 'r' if j in skeleton.joints_right else 'b'

                # plot joint
                #markers.append( ax.plot( positions_world[j,x], positions_world[j,y] ,
                #                         positions_world[j,z], 'o', c  = c, markersize = 1 , zdir = 'y' ) )
                
                # plot segment
                lines.append( ax.plot( [ positions_world[j, x], positions_world[skeleton_parents[j], x] ],
                                       [ positions_world[j, y], positions_world[skeleton_parents[j], y] ],
                                       [ positions_world[j, z], positions_world[skeleton_parents[j], z] ],
                                       zdir = 'y', c = c) )

            else:
                lines[j-1][0].set_xdata([positions_world[j, x], positions_world[skeleton_parents[j], x]])
                lines[j-1][0].set_ydata([positions_world[j, y], positions_world[skeleton_parents[j], y]])
                lines[j-1][0].set_3d_properties( [ positions_world[j, z], positions_world[skeleton_parents[j],z] ], zdir='y')

        #l = max(frame - draw_offset, 0)
        #r = min(frame+draw_offset, trajectory.shape[0])
        #spline_line.set_xdata(trajectory[l:r, 0])
        #spline_line.set_ydata(np.zeros_like(trajectory[l:r, 0]))
        #spline_line.set_3d_properties(trajectory[l:r, 1], zdir='y')
        
        initialized = True

        if name == 'show' and frame == data.shape[0] - 1:
            plt.close('all')
    
    
    fig.tight_layout()
    anim = FuncAnimation(fig, update, frames = np.arange(0, data.shape[0]), interval = 1000/fps, repeat = False)
    if name == 'show':
        plt.show()
        return anim
    """
    elif output == 'html':
        return anim.to_html5_video()
    elif output.endswith('.mp4'):
        Writer = writers['ffmpeg']
        writer = Writer(fps=fps, metadata={}, bitrate=bitrate)
        anim.save(output, writer=writer)
    elif output.endswith('.gif'):
        anim.save(output, dpi=80, writer='imagemagick')
    else:
        raise ValueError('Unsupported output format (only html, .mp4, and .gif are supported)')
    plt.close()
    """


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


s = Skeleton( data    = data['S1']['walking_2'],
              offsets = offsets                   ,
              parents = parents                   ,
              joints_left  = joints_left          ,
              joints_right = joints_right         )

if __name__ == '__main__':

    #plot_frame(s, 500)
    animation(s, fps = 60 )