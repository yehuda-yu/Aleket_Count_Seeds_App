# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 15:59:10 2022

@author: User
"""
import streamlit as st
from PIL import Image
import numpy as np
import cv2
import skimage.io as io


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

###### Streamlit ######
st.set_page_config(page_icon = "bar_chart", layout = 'wide')
st.title("Image KMeans Clustering")

# Allow the user to select a TIF image file
uploaded_file = st.file_uploader("Choose image file")

if uploaded_file is not None:
    
    with st.expander("See explanation"):
    st.write("the chart above shows some numbers I picked for you.rolled actual dice for these, so they're *guaranteed* to be random.")
    
    
    # Open the TIF image using io amd present image:
    image = io.imread(uploaded_file)
    st.subheader('Original image:')
    st.image(image,use_column_width=True)
    
    # slider for choosing K:
    k_value = st.slider('Insert K value (number of clusters):', 2,10,3) # asks for input from the user
    attempts_value_slider = st.slider('Number of attempts', value = 7, min_value = 1, max_value = 10) # slider example
    segmented_image = segment_image_kmeans(image, k=k_value, attempts=attempts_value_slider)
    
    # Display the result
    st.subheader('Output Image')
    st.image(segmented_image, use_column_width=True)
