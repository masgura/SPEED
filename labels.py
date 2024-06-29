import numpy as np
from numpy import array as npa
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import patches  # for drawing rectangular BB

import json
import os
import shutil
import pandas as pd

# import module from parent directory
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from utils_various import *


##

# Actual dataset
createDirectory(os.path.join(mySPEED_dir, 'labels'))
createDirectory(os.path.join(mySPEED_dir, 'labels/train'))
createDirectory(os.path.join(mySPEED_dir, 'labels/dev'))
createDirectory(os.path.join(mySPEED_dir, 'labels/test'))


# Dataset of RoIs (used for training landmark regression)
createDirectory(os.path.join(mySPEED_RoI_dir, 'images'))
createDirectory(os.path.join(mySPEED_RoI_dir, 'images/train'))
createDirectory(os.path.join(mySPEED_RoI_dir, 'images/dev'))
createDirectory(os.path.join(mySPEED_RoI_dir, 'images/test'))
createDirectory(os.path.join(mySPEED_RoI_dir, 'labels'))
createDirectory(os.path.join(mySPEED_RoI_dir, 'labels/train'))
createDirectory(os.path.join(mySPEED_RoI_dir, 'labels/dev'))
createDirectory(os.path.join(mySPEED_RoI_dir, 'labels/test'))

# HRNet labels folder (contains: train,dev,test.json)
createDirectory('./sharedData')


with open('./sharedData/train.json') as jFile:
    train = json.load(jFile)
with open('./sharedData/dev.json') as jFile:
    dev = json.load(jFile)
with open('./sharedData/test.json') as jFile:
    test = json.load(jFile)



# Save landmarks list of dicts as a JSON file
with open('./sharedData/landmarks.json', 'w') as fp:
    json.dump(Wireframe.landmarks, fp)


##

def project3Dto2D(q_CB, t_CB, r_B_mat):
    """ Projecting points to image frame.
        q_CB:       quaternion representing rotation: camera_frame --> Tango princ. axes
        t_CB:       camera2body_translation
        r_B_mat:    body coordinates of SC points (stacked column by column)
    """

    points_body = np.concatenate( ( r_B_mat, np.ones((1,r_B_mat.shape[1])) ), axis=0 )

    # transformation to camera frame
    pose_mat = np.hstack( ( quat2dcm(q_CB).T, np.expand_dims(t_CB, 1) ) )
    p_cam = np.dot(pose_mat, points_body)

    # getting homogeneous coordinates
    points_camera_frame = p_cam / p_cam[2]

    # projection to image plane
    points_image_plane = Camera.K.dot(points_camera_frame)

    x, y = (points_image_plane[0], points_image_plane[1])
    return x, y



class DatasetImage:

    """ Class for dataset inspection:
        provides access to individual images and corresponding ground truth labels.
    """

    def __init__(self, partition='train', idx=0):
        self.partition = partition
        self.idx = idx
        self.width = self.get_image().size[0]
        self.height = self.get_image().size[1]

    def get_pose(self):
        """ Get pose label for current image: quaternion rotation and translation. """
        if self.partition == 'train':
            q_CB = npa(train[self.idx]['q_vbs2tango'])
            t_CB = npa(train[self.idx]['r_Vo2To_vbs_true'])
        elif self.partition == 'dev':
            q_CB = npa(dev[self.idx]['q_vbs2tango'])
            t_CB = npa(dev[self.idx]['r_Vo2To_vbs_true'])
        elif self.partition == 'test':
            q_CB = npa(test[self.idx]['q_vbs2tango'])
            t_CB = npa(test[self.idx]['r_Vo2To_vbs_true'])
        else:
            q_CB, t_CB = (None, None)
        return q_CB, t_CB  # numpy arrays

    def get_image(self):
        """ Load image as PIL image. """
        img_dir = os.path.join(mySPEED_dir, 'images', self.partition)
        if self.partition == 'train':
            img_dir = os.path.join(img_dir, train[self.idx]['filename'])
        elif self.partition == 'dev':
            img_dir = os.path.join(img_dir, dev[self.idx]['filename'])
        elif self.partition == 'test':
            img_dir = os.path.join(img_dir, test[self.idx]['filename'])
        else:
            img_dir = os.path.join(img_dir, 'ERROR')

        img = Image.open(img_dir).convert('RGB')
        return img

    def get_image_name(self):
        """ Retrieve the file name of the image """
        if self.partition == 'train':
            filename = train[self.idx]['filename']
        elif self.partition == 'dev':
            filename = dev[self.idx]['filename']
        elif self.partition == 'test':
            filename = test[self.idx]['filename']
        else:
            filename = None
        return filename

    def show_image(self):
        Image._show(self.get_image())
        return
    def get_bb(self, enlargement=BB_enlarge):
        """ Retrieve Bounding Box coordinates and size """
        q, r = self.get_pose()

        # get original landmarks
        xl, yl = project3Dto2D(q,r,Wireframe.landmark_mat)

        # adjust landmarks whose projections lies outside image bounds
        xl[xl > self.width] = self.width
        xl[xl < 0] = 0
        yl[yl > self.height] = self.height
        yl[yl < 0] = 0

        x_min = np.min(xl)
        x_max = np.max(xl)
        y_min = np.min(yl)
        y_max = np.max(yl)

        # we slightly relax the minimum rectangle enclosing all landmarks;
        w = (x_max - x_min)
        h = (y_max - y_min)
        avg_side = (w+h)/2
        w += enlargement * avg_side
        h += enlargement * avg_side

        # top-left corner (referred to axes having (0,0) @ TL, i.e. ↓⟶  )
        xTL = x_min - enlargement * avg_side/2
        yTL = y_min - enlargement * avg_side/2

        # we enforce that BB fits within the image frame
        if xTL < 0:
            w = w-xTL
            xTL = 0
        if yTL < 0:
            h = h-yTL
            yTL = 0
        w = np.min([self.width-xTL, w])
        h = np.min([self.height-yTL, h])

        return (xTL, yTL), w, h

    def get_projected_landmarks(self):
        q, r = self.get_pose()
        xl, yl = project3Dto2D(q, r, Wireframe.landmark_mat)
        visibility_mask = []
        for xli, yli in zip(xl, yl):
            if xli < 0 or xli > Camera.nu or yli < 0 or yli > Camera.nv:
                visibility_mask.append(0)
            else:
                visibility_mask.append(2)
            # i.e. is landmark within image frame? (bool)

        return xl, yl, list(visibility_mask)


    def landmarks_roi_frame(self):
        xl, yl, visibility_mask = self.get_projected_landmarks()
        TL, w, h  = self.get_bb()

        xl_roi_norm = (xl - TL[0]) / w
        yl_roi_norm = (yl - TL[1]) / h

        return xl_roi_norm, yl_roi_norm, visibility_mask


    def visualize_pose(self, ax=None):
        """ Visualizing image pose labels. """

        plt.figure()
        if ax is None:
            ax = plt.gca() # Plot using Current Axes
        img = self.get_image()
        ax.imshow(img)

        plt.xlim([0, Camera.nu])
        plt.ylim([Camera.nv, 0])
        plt.xlabel('$P_x$ [px]')
        plt.ylabel('$P_y$ [px]')

        # Plot Bounding Box
        # BB label
        bb_TL, w, h = self.get_bb()
        rect = patches.Rectangle(bb_TL, w, h,
                                 linewidth=1, edgecolor='y', facecolor='none')
        ax.add_patch(rect)
        # minimum rectangle
        bb_TL, w, h = self.get_bb(0)
        rect = patches.Rectangle(bb_TL, w, h,
                                 linestyle='--', linewidth=0.4, edgecolor='y', facecolor='none')
        ax.add_patch(rect)



        q, r = self.get_pose()


        # plot axes
        xa, ya = project3Dto2D(q,r,p_axes)
        ax.arrow(xa[0], ya[0], xa[1] - xa[0], ya[1] - ya[0], head_width=24, color='r')
        ax.arrow(xa[0], ya[0], xa[2] - xa[0], ya[2] - ya[0], head_width=24, color='r')
        ax.arrow(xa[0], ya[0], xa[3] - xa[0], ya[3] - ya[0], head_width=24, color='r')

        # axis labels
        plt.text(xa[1] + 60*np.sign(xa[1] - xa[0]), ya[1], 'x', color='r',
                 horizontalalignment='center',verticalalignment = 'center')
        plt.text(xa[2] + 60*np.sign(xa[2] - xa[0]), ya[2], 'y', color='r',
                 horizontalalignment='center', verticalalignment='center')
        plt.text(xa[3] + 60*np.sign(xa[3] - xa[0]), ya[3], 'z', color='r',
                 horizontalalignment='center', verticalalignment='center')


        # plot landmarks
        xl, yl = project3Dto2D(q,r,Wireframe.landmark_mat)
        plt.plot(xl, yl, 'co', markersize=2)

        # plot wireframe
        wfWidth = 0.5;
        # bottom of main body
        plt.plot(np.hstack((xl[0:4], xl[0])),
                 np.hstack((yl[0:4], yl[0])),
                 'aqua', lw=wfWidth)
        # solar panel
        plt.plot(np.hstack((xl[4:8], xl[4])),
                 np.hstack((yl[4:8], yl[4])),
                 'aqua', lw=wfWidth)
        # top of main body
        x_top, y_top = project3Dto2D(q, r, Wireframe.topMainBody_mat)
        plt.plot(np.hstack((x_top[0:4], x_top[0])),
                 np.hstack((y_top[0:4], y_top[0])),
                 'aqua', lw=wfWidth)
        # corners
        for k in np.arange(0,4):
            plt.plot(np.hstack((xl[k], x_top[k])),
                     np.hstack((yl[k], y_top[k])),
                     'aqua', lw=wfWidth)
        # antennas
        x_clamp, y_clamp = project3Dto2D(q, r, Wireframe.antClamps_mat)
        for k in np.arange(0,3):
            plt.plot(np.hstack((x_clamp[k], xl[8+k])),
                     np.hstack((y_clamp[k], yl[8+k])),
                     'aqua', lw=wfWidth)

    def save_roi_img(self, img_path):

        # Crop RoI
        img = self.get_image()
        TL, w, h = self.get_bb()
        x1 = TL[0]
        y1 = TL[1]
        x2 = x1 + w
        y2 = y1 + h

        img_ROI = img.crop((x1, y1, x2, y2))

        # Save .jpg
        img_ROI.save(img_path)

        return

## Assign Bounding Box and Landmark labels
# (write to 3 global JSON files, 3 HRNet JSON files, 12000 TXT files)

count = -1 # used for looping over train,dev,test
for mySet in [train, dev, test]:
    count += 1

    for idx,image_fromList in enumerate(mySet):
        img = DatasetImage(setOrder[count], idx)

        # Bounding Box
        # For global .json file
        TL, w, h = img.get_bb()
        mySet[idx]['bounding_box'] = {
            'TL': (TL[0]/img.width, TL[1]/img.height),
            'w': w/img.width,
            'h': h/img.height
        }
   
        img.save_roi_img(
                os.path.join(mySPEED_RoI_dir, 'images', setOrder[count], img.get_image_name())
            )

        # For HRNet .json file
        x1 = TL[0];     x2 = TL[0]+w;
        y1 = TL[1];     y2 = TL[1]+h;


        # Landmarks' normalized coordinates in RoI frame
        xl_roi_norm, yl_roi_norm, visibility_mask = img.landmarks_roi_frame()
        mySet[idx]['landmarks_roi'] = {
            'x': [float("{:.6f}".format(x_i)) for x_i in list(xl_roi_norm)],
            'y': [float("{:.6f}".format(y_i)) for y_i in list(yl_roi_norm)],
            'visibility': visibility_mask
        }
        # For HRNet .json file
        xl, yl, visibility_mask = img.get_projected_landmarks()


        # update imgxxxxxx.txt files for YOLOv5 training
        xc_norm = (TL[0] + w / 2)/img.width
        yc_norm = (TL[1] + h / 2)/img.height
        wnorm = w/img.width
        hnorm = h/img.height
        txt_dir = os.path.join(mySPEED_dir, 'labels', setOrder[count], os.path.splitext(mySet[idx]['filename'])[0] + '.txt')
        with open(txt_dir, 'w') as text_file:
            text_file.write('0 %.6f %.6f %.6f %.6f' % (xc_norm, yc_norm, wnorm, hnorm) )
        txt_roi_dir = os.path.join(mySPEED_RoI_dir, 'labels', setOrder[count], os.path.splitext(mySet[idx]['filename'])[0] + '.txt')
        with open(txt_roi_dir, "w") as file:
            # Write the initial values
            file.write("0 0.5 0.5 1 1 ")
            
            # Iterate through the arrays and write the values to the file
            for xi, yi, vi in zip(xl, yl, visibility_mask):
                file.write(f"{xi} {yi} {vi} ")

        # print progress
        if (idx % 100) == 0:
            print('%s BB-labels: %i%% (%i of %i)'
                  % (setOrder[count], int(idx / len(mySet) * 100), count+1, len(setOrder)), end='\r')
            # last argument to delete the previously printed line at each progress update


    # Update train,dev,test.json

    # A workaround to avoid 'TypeError: Object of type int64 is not JSON serializable'
    # You should always use this whenever storing a LIST as a dictionary key to be saved as a JSON file
    def convert(o):
        if isinstance(o, np.int64):
            return int(o)

    with open('./sharedData/' + setOrder[count] + '.json', 'w') as fp:
        json.dump(mySet, fp, default=convert)




## Pose visualization examples
def get_distr_and_idx(name):
    idx = None
    distr = None
    count = 0
    for mySet in [train, dev, test]:
        for currentIdx, image in enumerate(mySet):
            if image['filename'] == name:
                idx = currentIdx
                break
        if idx is not None:
            break
        count += 1


    if idx is None:
        print('No such filename found in this dictionary')
    else:
        distr = setOrder[count]
    return distr, idx


createDirectory('pose_visualization')
for partition, idx in [get_distr_and_idx('img005020.jpg'),
                       get_distr_and_idx('img012706.jpg'),
                       get_distr_and_idx('img003976.jpg'),
                       get_distr_and_idx('img000002.jpg'),
                       get_distr_and_idx('img013927.jpg'),
                       get_distr_and_idx('img013256.jpg'),
                       get_distr_and_idx('img001971.jpg'),
                       get_distr_and_idx('img010239.jpg')
                       ]:
    img = DatasetImage(partition, idx)
    img.visualize_pose()
    plt.savefig(os.path.join('pose_visualization',
                             partition + '_' + img.get_image_name() + '.pdf'),
                bbox_inches='tight')

