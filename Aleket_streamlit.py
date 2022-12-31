# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 15:59:10 2022

@author: User
"""
import streamlit as st
from PIL import Image
import numpy as np
import cv2
from PIL import Image
import skimage.io as io
from skimage import color
from skimage.filters import median
from skimage.measure import find_contours, label
from skimage.filters import threshold_otsu
from scipy.ndimage import binary_closing, binary_opening
import scipy
import matplotlib.pyplot as plt
import stardist
from stardist.models import StarDist2D 
from csbdeep.utils import normalize

###################### Functions ######################
@st.cache
# function for k mean
def segment_image_kmeans(img, k=3, attempts=10): 

    # Convert MxNx3 image into Kx3 where K=MxN
    pixel_values  = img.reshape((-1,3))  #-1 reshape means, in this case MxN

    #We convert the unit8 values to float as it is a requirement of the k-means method of OpenCV
    pixel_values = np.float32(pixel_values)

    # define stopping criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    
    _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS)
    
    # convert back to 8 bit values
    centers = np.uint8(centers)

    # flatten the labels array
    labels = labels.flatten()
    
    # convert all pixels to the color of the centroids
    segmented_image = centers[labels.flatten()]
    
    # reshape back to the original image dimension
    segmented_image = segmented_image.reshape(img.shape)
    
    return segmented_image


def process_image(image):
    # Convert to PIE-LaB format:
    image_lab = color.rgb2lab(image)

    # Median filter on L channel to clean Noise
    L_channel = image_lab[:,:,2]
    L_channel = scipy.ndimage.median_filter(L_channel, footprint=np.ones((2,2)))
    
    return image_lab, L_channel

def mask_and_clean(L_channel,threshold):

    # Create a mask using the threshold
    mask = np.where(L_channel < threshold, 0, 255)

    # Clean up the binary image using morphological operations
    cleaned_image = binary_closing(mask, structure=np.ones((3,3)))
    cleaned_image = binary_opening(cleaned_image, structure=np.ones((3,3)))

    # Identify seed boundaries
    contours = find_contours(cleaned_image, .5)

    return mask, cleaned_image, contours

# StarDist function
def StarDist_prediction(image):

    # Define a pretrained model to segment nuclei in fluorescence images (download from pretrained)
    # Display a spinner while the model is making predictions
    with st.spinner('Making predictions on image...'):
        # Define the model from stardist
        model = StarDist2D.from_pretrained("Versatile (fluorescent nuclei)")
        axis_norm = (0,1)
        
        # Use the model to predict object instances in the image
        instances, labels = model.predict_instances(normalize(image.astype(int), 1,99.8, axis=axis_norm))
        
        # Plot the instances on top of the original image
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(image, cmap="gray")
        ax.imshow(instances, alpha=0.4)
        
        time.sleep(2)
        
        return fig
      

    


    # Plot the instances on top of the original image
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image, cmap="gray")
    ax.imshow(instances, alpha=0.4)

    return fig
# Allow the user to select a TIF image file
uploaded_file = st.file_uploader("Choose image file")

if uploaded_file is not None:
    # Open the TIF image using io amd present image:
    image = io.imread(uploaded_file)
    st.subheader('Original image:')
    st.image(image,use_column_width=True)
    # Expanded bottom fore analysis:
    with st.expander("Filters"):
               
        image_lab, L_channel = process_image(image)
        
        # Calculate the Otsu threshold and create masked image
        st.write('3. Mask with threshold')
        
        # Otsu thresh for recommendation:
        otsu_thresh = threshold_otsu(L_channel)
        st.info(f'Otsu threshold is {round(otsu_thresh,2)}')
        
        # Set threshold by user choice and apply mask
        threshold = st.slider('Threshold Value', min_value=0, max_value=255)
        mask, cleaned_image, contours = mask_and_clean(L_channel,threshold)

        st.write('4. Clean image and count seeds')
        # Plot seed contours
        fig, ax = plt.subplots()
        ax.imshow(L_channel*cleaned_image, cmap='gray')
        ax.set_title('Clean Image')
        for contour in contours:
           ax.plot(contour[:, 1], contour[:, 0], '-r', linewidth=1.5)
        # Use Streamlit to display the Matplotlib plot
        st.pyplot(fig)
        
        # Print number of seeds in image
        st.write('Image contains',len(contours),'seeds')

        # StarDist:
        st.write('StarDist prediction')
        StarDist_prediction(L_channel*cleaned_image)
        
        
        ###################### Kmens ######################

    # slider for choosing K:
    k_value = st.slider('Insert K value (number of clusters):', 2,10,3) # asks for input from the user
    attempts_value_slider = st.slider('Number of attempts', value = 7, min_value = 1, max_value = 10) # slider example
    segmented_image = segment_image_kmeans(image, k=k_value, attempts=attempts_value_slider)
    
    # Display the result
    st.subheader('Output Image')
    st.image(segmented_image, use_column_width=True)
