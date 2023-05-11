from skimage.measure import regionprops
from scipy import ndimage
from scipy.ndimage.morphology import distance_transform_edt
from matplotlib.colors import LinearSegmentedColormap
import colorcet as cc
import numpy as np


# Methods to fill holes on label image
# Converts a single label into a binary mask and fills its holes
def fill_mask(image, label_id):   
    binary_label_id = np.where(image == label_id, 1, 0) # crates binary image containing only the specified label
    filled = ndimage.binary_fill_holes(binary_label_id) # fills the binary mask
    filled_label_id = np.where(filled == 1, label_id, 0) # creates label image containing only the specified label after filling

    return filled_label_id

# Fill the holes on all the masks contained in a label image
def fill_labels(image):
    filled_labels_list = [] # creates empty list to save images of the filled labels
    regions = regionprops(image)

    # fills every label and stores them as individual images
    for i in range(len(regions)):
        label_id = regions[i].label # gets the label id of the current region
        filleded_label = fill_mask(image, label_id) # creates label image containing only the specified label after filling
        filled_labels_list.append(filleded_label) # stores the image on the list

    # reconstructs the label image, now containing the filled labels
    filleded_labels_stack = np.stack(filled_labels_list) # creates stack from list of images (numpy arrays)
    image_filled = np.max(filleded_labels_stack, axis = 0) # calculates the maximum projection to get back a 2D, labelled image

    return image_filled

# Methods to compute distance transforms on label image
# Converts a single label into a binary mask and gets the distance transform
def dist_trans_mask(image, label_id):   
    binary_label_id = np.where(image == label_id, 1, 0) # crates binary image containing only the specified label
    dist_trans = distance_transform_edt(binary_label_id) # gets the distance transform of the binary mask
    
    return dist_trans

# get the distance transforms from all the masks contained in a label image
def dist_trans_labels(image):
    dist_trans_list = [] # creates empty list to save images of individual distance transforms
    regions = regionprops(image)

    # gets the image transform of every label and stores them as individual images
    for i in range(len(regions)):
        label_id = regions[i].label # gets the label id of the current region
        dist_trans_label = dist_trans_mask(image, label_id) # creates image containing only the specified distance transform
        dist_trans_list.append(dist_trans_label) # stores the image on the list

    # generates a new image containing all the distance transforms
    dist_trans_stack = np.stack(dist_trans_list) # creates stack from list of images (numpy arrays)
    image_dist_trans = np.max(dist_trans_stack, axis = 0) # calculates the maximum projection to get back a 2D image

    return image_dist_trans

# Methods to erode labels
def boolean_kernel(crop):
    unique_labels = np.unique(crop)
    if unique_labels.size > 1:
      return 0
    else:
       return crop[1,1]

def erode_labels(image):
    image_eroded = np.zeros([image.shape[0]-2,image.shape[1]-2])
    for x in range(image.shape[1]-2):
      for y in range(image.shape[0]-2):
        crop = image[y:y+3, x:x+3]
        #if(y==100):
        #  print(crop.shape)
        current_result = boolean_kernel(crop)
        image_eroded[y, x] = current_result
  
    return image_eroded

def erode_labels2(image):
    image_eroded = []
    for x in range(image.shape[1]-2):
      for y in range(image.shape[0]-2):
        crop = image[y:y+3, x:x+3]
        #if(y==100):
        #  print(crop.shape)
        image_eroded.append(boolean_kernel(crop))
    image_eroded = np.array(image_eroded)
  
    return np.reshape(image_eroded, (image.shape[0]-2,image.shape[1]-2), order='F')

# Get glasbey cmap
def get_glasbey_cmap():
    l=cc.cm.glasbey_bw_minc_20_minl_30_r.colors            
    l[0]=[0,0,0]
    cmap_glasbey = LinearSegmentedColormap.from_list('my_list', l, N=1000)
    return cmap_glasbey



