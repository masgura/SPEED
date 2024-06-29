import shutil
import os
import math
import numpy as np
from numpy import array as npa
import matplotlib as mpl

colab_drive_dir = '/content/gdrive/My Drive'
if os.path.exists(colab_drive_dir):
    homeDir = colab_drive_dir
else:
    homeDir = os.path.curdir
print(homeDir)
originalSPEED_dir = homeDir + '/speed'
mySPEED_dir = homeDir + '/speed_MS'
mySPEED_RoI_dir = homeDir + '/speed_MS_roi'
earthBGStart_ID = 7500


RENDERING_TAG = '_rendering'


# rgb_color_sequence = [plt.rcParams['axes.prop_cycle'].by_key()['color'][0],
#                       'darkred',
#                       'olivedrab']
rgb_color_sequence = ['#249cd8',  # B
                      '#e2415a',  # R
                      '#92d849']  # G

# Create custom colormaps with: hex_to_rgb(), rgb_to_dec(), get_continuous_cmap()
def hex_to_rgb(value):
    """
    Converts hex to rgb colours
    value: string of 6 characters representing a hex colour.
    Returns: list length 3 of RGB values
    """
    value = value.strip("#") # removes hash symbol if present
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
def rgb_to_dec(value):
    """
    Converts rgb to decimal colours (i.e. divides each value by 256)
    value: list (length 3) of RGB values
    Returns: list (length 3) of decimal values
    """
    return [v/256 for v in value]
def get_continuous_cmap(hex_list, float_list=None):
    """ creates and returns a color map that can be used in heat map figures.
        If float_list is not provided, colour map graduates linearly between each color in hex_list.
        If float_list is provided, each color in hex_list is mapped to the respective location in float_list.

        Parameters
        ----------
        hex_list: list of hex code strings
        float_list: list of floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1.

        Returns
        ----------
        colormap
    """

    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0, 1, len(rgb_list)))

    cdict = dict()
    for num, col in enumerate(['red', 'green', 'blue']):
        col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
        cdict[col] = col_list
    cmp = mpl.colors.LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
    return cmp
rgb_cmap = get_continuous_cmap(hex_list= [rgb_color_sequence[i] for i in [1,0,2]] )  # RBG order
rb_cmap  = get_continuous_cmap(hex_list= [rgb_color_sequence[i] for i in [0,1]] )


def createDirectory(myDir):
    if os.path.exists(myDir):
        shutil.rmtree(myDir)
    os.mkdir(myDir)

setOrder = ['train', 'dev', 'test']

def get_coords_from_landmark_label(landmarks, point_label):
    is_my_label_here = [point['label'] == point_label for point in landmarks]
    my_label_idx = [idx for idx, myBool in enumerate(is_my_label_here) if myBool][0]
    return landmarks[my_label_idx]['r_B']

class Wireframe:
    """"
    Utility class that defines landmarks and coordinates
    of other points from the wireframe model.
    """

    # Define 11 landmark points (body coordinates)
    landmarks = [
    {'label': 'B1', 'r_B': [-0.37, 0.304, 0]},
    {'label': 'B2', 'r_B': [-0.37, -0.264, 0]},
    {'label': 'B3', 'r_B': [0.37, -0.264, 0]},
    {'label': 'B4', 'r_B': [0.37, 0.304, 0]},
    {'label': 'S1', 'r_B': [-0.37, 0.385, 0.3215]},
    {'label': 'S2', 'r_B': [-0.37, -0.385, 0.3215]},
    {'label': 'S3', 'r_B': [0.37, -0.38, 0.3215]},
    {'label': 'S4', 'r_B': [0.37, 0.385, 0.3215]},
    {'label': 'A1', 'r_B': [-0.5427, 0.4877, 0.2535]},
    {'label': 'A2', 'r_B': [0.3050, -0.5790, 0.2515]},
    {'label': 'A3', 'r_B': [0.5427, 0.4877, 0.2591]}
        ]

    landmark_mat = np.column_stack( [point['r_B'] for point in landmarks] )

    # Top of the main body (not used as landmarks)
    topMainBody = [
    {'label': 'T1', 'r_B': [-0.37, 0.304, 0.305]},
    {'label': 'T2', 'r_B': [-0.37, -0.264, 0.305]},
    {'label': 'T3', 'r_B': [0.37, -0.264, 0.305]},
    {'label': 'T4', 'r_B': [0.37, 0.304, 0.305]}
        ]
    topMainBody_mat = np.column_stack( [point['r_B'] for point in topMainBody] )

    # Antenna clamps
    antClamps = [
    {'label': 'Ac1', 'r_B': [-0.23, 0.3, 0.2535]},
    {'label': 'Ac2', 'r_B': [0.31, -0.26, 0.2515]},
    {'label': 'Ac3', 'r_B': [0.23, 0.3, 0.2591]}
        ]
    antClamps_mat = np.column_stack( [point['r_B'] for point in antClamps] )

    body_center = [0, 0, (0+get_coords_from_landmark_label(landmarks, 'S1')[2])/2]

    # We compute L_c as the diagonal length of the solar panel
    k1 = 1.05  # a constant empirically tuned, by testing on outliers
    charact_length = k1 * np.linalg.norm( npa(get_coords_from_landmark_label(landmarks, 'S1'))
                                     -
                                     npa(get_coords_from_landmark_label(landmarks, 'S3')) )




# reference points @ body frame (for drawing axes)
p_axes = np.array([[0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])


BB_enlarge = .1  # i.e. 10% larger than minimum rectangle
max_ROI_size = 320  # [px]


class Camera:
    """" Utility class for accessing camera parameters. """

    fx = 0.0176  # focal length [m]
    fy = 0.0176  # focal length [m]
    nu = 1920  # no. horizontal pixels
    nv = 1200  # no. of vertical pixels
    ppx = 5.86e-6  # horizontal pixel pitch [m/px]
    ppy = ppx      # vertical pixel pitch [m/px]
    fpx = fx / ppx  # horizontal focal length [px]
    fpy = fy / ppy  # vertical focal length [px]
    k = [[fpx,   0, nu / 2], # camera intrinsic matrix
         [0,   fpy, nv / 2],
         [0,     0,      1]]
    K = npa(k)

    # angular size of the camera's FoV, considering the diagonal aperture [deg]
    FOV_diagonal_deg = 180*math.pi * 2 * math.atan( ppx * math.sqrt(nu**2 + nv**2) / (2*fx) )


def project3Dto2D(dcm_CB, t_CB, r_B_mat):
    """ Projecting points to image frame.
        q_CB:       quaternion representing rotation: camera_frame --> Tango princ. axes
        t_CB:       camera2body_translation
        r_B_mat:    body coordinates of SC points (stacked column by column)
    """

    points_body = np.concatenate( ( r_B_mat, np.ones((1,r_B_mat.shape[1])) ), axis=0 )

    # transformation to camera frame
    pose_mat = np.hstack( ( dcm_CB.T, np.expand_dims(t_CB, 1) ) )
    p_cam = np.dot(pose_mat, points_body)

    # getting homogeneous coordinates
    points_camera_frame = p_cam / p_cam[2]

    # projection to image plane
    points_image_plane = Camera.K.dot(points_camera_frame)

    x, y = (points_image_plane[0], points_image_plane[1])
    return x, y



def quat2dcm(q):

    """ Convert quaternion [q4 q1 q2 q3] to Direction Cosine Matrix. """

    # normalizing quaternion
    q = q/np.linalg.norm(q)

    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]

    dcm = np.zeros((3, 3))

    dcm[0, 0] = 2 * q0**2 - 1 + 2 * q1**2
    dcm[1, 1] = 2 * q0**2 - 1 + 2 * q2**2
    dcm[2, 2] = 2 * q0**2 - 1 + 2 * q3**2

    dcm[0, 1] = 2 * q1 * q2 + 2 * q0 * q3
    dcm[0, 2] = 2 * q1 * q3 - 2 * q0 * q2

    dcm[1, 0] = 2 * q1 * q2 - 2 * q0 * q3
    dcm[1, 2] = 2 * q2 * q3 + 2 * q0 * q1

    dcm[2, 0] = 2 * q1 * q3 + 2 * q0 * q2
    dcm[2, 1] = 2 * q2 * q3 - 2 * q0 * q1

    return dcm

def dcm2quat(dcm):
    """
    Based on the method described here:
    http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
    """
    if dcm[2, 2] < 0:
        if dcm[0, 0] > dcm[1, 1]:
            t = 1 + dcm[0, 0] - dcm[1, 1] - dcm[2, 2]
            q = [dcm[1, 2] - dcm[2, 1], t, dcm[0, 1] + dcm[1, 0], dcm[2, 0] + dcm[0, 2]]
        else:
            t = 1 - dcm[0, 0] + dcm[1, 1] - dcm[2, 2]
            q = [dcm[2, 0] - dcm[0, 2], dcm[0, 1] + dcm[1, 0], t, dcm[1, 2] + dcm[2, 1]]
    else:
        if dcm[0, 0] < -dcm[1, 1]:
            t = 1 - dcm[0, 0] - dcm[1, 1] + dcm[2, 2]
            q = [dcm[0, 1] - dcm[1, 0], dcm[2, 0] + dcm[0, 2], dcm[1, 2] + dcm[2, 1], t]
        else:
            t = 1 + dcm[0, 0] + dcm[1, 1] + dcm[2, 2]
            q = [t, dcm[1, 2] - dcm[2, 1], dcm[2, 0] - dcm[0, 2], dcm[0, 1] - dcm[1, 0]]

    q = np.array(q)
    q *= 0.5 / math.sqrt(t)
    return q





# def euler2dcm(euler):
#     R_x = npa([[1, 0, 0],
#                     [0, math.cos(euler[0]), math.sin(euler[0])],
#                     [0, -math.sin(euler[0]), math.cos(euler[0])]
#                     ])
#
#     R_y = npa([[math.cos(euler[1]), 0, math.sin(euler[1])],
#                     [0, 1, 0],
#                     [math.sin(euler[1]), 0, math.cos(euler[1])]
#                     ])
#
#     R_z = npa([[math.cos(euler[2]), -math.sin(euler[2]), 0],
#                     [-math.sin(euler[2]), math.cos(euler[2]), 0],
#                     [0, 0, 1]
#                     ])
#
#     dcm = np.dot(R_z, np.dot(R_y, R_x))
#
#     return dcm

def dcm2euler(dcm):
    """
    Converts Direction Cosine Matrix to corresponding Euler angles representation.

    th_x, th_y, th_z [deg] are computed as the rotation angles
    about the x,y,x axes, respectively, whose sign is given by the SCREW rule
    """
    # assert(isRotationMatrix(dcm))

    # N.B. I used math instead of numpy since it is a little faster

    sy = math.sqrt(dcm[0, 0] * dcm[0, 0] + dcm[1, 0] * dcm[1, 0])
    singular = sy < 1e-6

    if not singular:
       th_x = math.atan2( dcm[2, 1], dcm[2, 2])
       th_y = math.atan2(-dcm[2, 0], sy)
       th_z = math.atan2( dcm[1, 0], dcm[0, 0])

    else:
        th_x = math.atan2(-dcm[1, 2], dcm[1, 1])
        th_y = math.atan2(-dcm[2, 0], sy)
        th_z = 0

    return -npa([th_x, th_y, th_z]) * 180/math.pi  # [deg]



