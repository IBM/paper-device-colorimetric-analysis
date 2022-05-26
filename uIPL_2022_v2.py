###############################################################################################################
## uPad Image Processing Library - uIPL
## Last Updated: May, 2022
## Industrial Technology & Science
## IBM Research Brazil
###############################################################################################################
#
# Copyright 2020- IBM Inc. All rights reserved
# SPDX-License-Identifier: BSD-3-Clause
#
###############################################################################################################
##Importing Packages

import numpy as np
import matplotlib as mpl 
import matplotlib.pyplot as plt 
import cv2 
import sklearn.decomposition as sd
import skimage 
import scipy as spy
from PIL import Image
from skimage.filters import try_all_threshold
from skimage.filters import threshold_isodata
from skimage.filters import threshold_mean
from skimage.filters import threshold_yen
from skimage.filters import threshold_li
from skimage.color import rgb2hsv
import skimage.morphology as morpho
import pandas as pd
from skimage.color import rgb2hsv, hsv2rgb, rgb2lab, lab2rgb
import glob
import os
import json

#################################################################################################################
##Classes

class uPad:
    
    def __init__(self, image, GlobalRef):
        self.image = image #Loads the Test Image
        self.GlobalRef = GlobalRef #Loads the color reference from the calibration data; 
        self.TotalGlobalRef = np.stack((self.GlobalRef[0,:3],self.GlobalRef[0,3:6]), axis=1) #to format to [[channelMean, Channelstd]...]
        self.GlobalRef_cmtx = self.GlobalRef[1:,:3]
        
    def set_card_output(self, croplim):
        self.output_position = croplim
        self.card_output_image_original = self.image[croplim[0]:croplim[1], croplim[2]:croplim[3]]
        self.mask_original, self.masked_output_original, self.spots_original = extract_spots(self.card_output_image_original, 
                                                                  (0, self.card_output_image_original.shape[0], 0, self.card_output_image_original.shape[1]))
        self.stats_original = extract_spot_statistics(self.spots_original)
    
    def set_card_ref(self, croplim, offset=[0,0]): #croplim = [Xmin, Xmax, Ymin, Ymax]
        self.card_ref_image = self.image[croplim[0]:croplim[1], croplim[2]:croplim[3]] #Cropping the reference region
        
        self.card_ref_mask = threshold(PCA_mask(self.card_ref_image)) #creating a mask by applying the PCA method
        self.card_ref_mask_total, self.white_mask = create_white_ref(self.card_ref_mask, offset=offset)
        
        self.card_ref_total = mask_img(self.card_ref_image, self.card_ref_mask_total)
        self.card_ref_colors = mask_img(self.card_ref_image, self.card_ref_mask)
        self.card_ref_white = mask_img(self.card_ref_image, self.white_mask)

        self.card_ref_cmtx = colormatrix_total(self.card_ref_mask, self.white_mask,
                                             self.card_ref_total)
    def load_ml_models(self, model_prefix):
        self.ML_models = (cv2.ml.LogisticRegression_load(model_prefix+'_L0.xml'),
                          cv2.ml.LogisticRegression_load(model_prefix+'_L1.xml'),
                          cv2.ml.LogisticRegression_load(model_prefix+'_L2.xml'),
                          cv2.ml.LogisticRegression_load(model_prefix+'_L3.xml'),
                          cv2.ml.LogisticRegression_load(model_prefix+'_L4.xml'))
        
            
    def correct_image(self, colorspace='RGB', reconstruction='linear', degree=3):
        
        if colorspace == 'RGB':
            self.card_ref_colortransfer = color_transf(self.card_ref_total.astype('uint8'), 
                                                    self.TotalGlobalRef, self.card_ref_mask_total)
            
            self.card_ref_colortransfer_spots = mask_img(self.card_ref_colortransfer, 
                                                        self.card_ref_mask_total)
            
            self.card_ref_colortransfer_cmtx = colormatrix_total(self.card_ref_mask, 
                                                    self.white_mask, self.card_ref_colortransfer_spots)
            
            self.card_ref_colortransfer_trn = transform_matrix(self.card_ref_colortransfer_cmtx, 
                                                            self.GlobalRef_cmtx)
            
            self.card_ref_corrected = apply_transform(self.card_ref_colortransfer, 
                                                    self.card_ref_colortransfer_trn)
            
            self.card_ref_corrected_cmtx = colormatrix_total(self.card_ref_mask, self.white_mask, 
                                                self.card_ref_corrected)
            
            if reconstruction == 'linear':
            
                self.correction_transform, out = linear_transform(self.image, 
                                                            self.card_ref_cmtx, 
                                                            self.card_ref_corrected_cmtx, raw_out=True)
            
                self.corrected_image = np.clip(out, 0, 255).astype('uint8')

            if reconstruction == 'poly':
            
                self.coeff = color_polyfit(self.card_ref_cmtx, self.card_ref_corrected_cmtx, degree=degree)
                
                out = img_polyfit(self.image, self.coeff)
                
                self.corrected_image = np.clip(out, 0, 255).astype('uint8')
                
                
        if colorspace == 'LAB':
            
            self.card_ref_total_lab = mask_img(cv2.cvtColor(self.card_ref_total.astype('uint8'), 
                                                            cv2.COLOR_RGB2LAB), self.card_ref_mask_total)
            
            self.card_ref_cmtx_lab = colormatrix_total(self.card_ref_mask, self.white_mask,
                                                self.card_ref_total_lab)
            
            
            self.card_ref_colortransfer = color_transf(self.card_ref_total.astype('uint8'), 
                                                    self.TotalGlobalRef, self.card_ref_mask_total)
            
            self.card_ref_colortransfer = cv2.cvtColor(self.card_ref_colortransfer.astype('uint8'), 
                                                            cv2.COLOR_RGB2LAB)
            
            self.card_ref_colortransfer_spots = mask_img(self.card_ref_colortransfer, 
                                                        self.card_ref_mask_total)
            
            self.card_ref_colortransfer_cmtx = colormatrix_total(self.card_ref_mask, 
                                                    self.white_mask, self.card_ref_colortransfer_spots)
            
            self.card_ref_colortransfer_trn = transform_matrix(self.card_ref_colortransfer_cmtx, 
                                                            self.GlobalRef_cmtx)
            
            self.card_ref_corrected = apply_transform(self.card_ref_colortransfer, 
                                                    self.card_ref_colortransfer_trn)
            
            self.card_ref_corrected_cmtx = colormatrix_total(self.card_ref_mask, self.white_mask, 
                                                self.card_ref_corrected)
            
            self.image_lab = cv2.cvtColor(self.image, cv2.COLOR_RGB2LAB)
            
            if reconstruction == 'linear':
            
                self.correction_transform, self.out = linear_transform(self.image_lab, 
                                                                self.card_ref_cmtx_lab, 
                                                                self.card_ref_corrected_cmtx)


                self.corrected_image = cv2.cvtColor(np.clip(self.out, 0, 255).astype('uint8'), 
                                                    cv2.COLOR_LAB2RGB)
                
            if reconstruction == 'poly':
            
                self.coeff = color_polyfit(self.card_ref_cmtx_lab, self.card_ref_corrected_cmtx, degree=degree)
                
                self.out = img_polyfit(self.image_lab, self.coeff)
                
                self.corrected_image = cv2.cvtColor(np.clip(self.out, 0, 255).astype('uint8'), 
                                                    cv2.COLOR_LAB2RGB)
                
        self.card_output_image_corrected = self.corrected_image[self.output_position[0]:self.output_position[1], self.output_position[2]:self.output_position[3]]

        self.mask_corrected, self.masked_output_corrected, self.spots_corrected = extract_spots(self.card_output_image_corrected , 
                                                              (0, self.card_output_image_corrected.shape[0], 0, self.card_output_image_corrected.shape[1]))
        self.stats_corrected = extract_spot_statistics(self.spots_corrected)
                
    def analyze_card(self):
        self.analysis_results = AgroPad_analysis(self.ML_models, self.stats_corrected)
        
        
                
###############################################################################################
def AgroPad_analysis(spot_models, stats, labels=['L0', 'L1', 'L2', 'L3', 'L4']):
    results = dict()
    for i in range(len(spot_models)):
        results[labels[i]] = run_ML_model(spot_models[i], stats[labels[i]][0])
    return results

def extract_spot_statistics(spots, labels=['L0', 'L1', 'L2', 'L3', 'L4']):
    stats = dict()
    for i in range(len(spots)):
        stats[labels[i]] = non_zero_analysis(spots[i])
    return stats

def run_ML_model(model, rgb):
    # cv_lr = cv2.ml.LogisticRegression_create()
    # cv_lr = cv_lr.load(model_path)
    rgb = rgb.reshape(1,3).astype('float32')
    return model.predict(rgb)

def load_IllumCal(json_path, lot):
    calib_file = open(json_path)
    calibration = json.load(calib_file)[lot]
    illum_ref = np.empty((5,6))
    reference_spots = ['Total', 'B', 'G', 'R', 'W']
    for i, j in enumerate(reference_spots):
        illum_ref[i] = np.fromiter(calibration['ambLightRef'][j].values(), dtype='float')
        
    return illum_ref
        

################################################################################################

def Calculate_Dataframe_Relative_Difference(df1, df2, by='Concentration', treated=False):
    avg_df1 = df1.groupby(by=by).mean().reset_index()
    avg_df2 = df2.groupby(by=by).mean().reset_index()
    Concentration = avg_df1['Concentration'][avg_df1['Concentration'] == avg_df2['Concentration']]
    Channels = ['Red', 'Green', 'Blue']
    diff = np.zeros((len(Concentration), 3))
    
    for i,j in enumerate(Concentration):
        for w,k in enumerate(Channels):
            if treated == False:
                diff[i, w] = relative_difference(avg_df1[avg_df1['Concentration'] == j]['{} Channel'.format(k)],
                                                       avg_df2[avg_df2['Concentration'] == j]['{} Channel'.format(k)])
            if treated == True:
                diff[i, w] = relative_difference(avg_df1[avg_df1['Concentration'] == j]['{} Channel'.format(k)],
                                                       avg_df2[avg_df2['Concentration'] == j]['Treated{}'.format(k)])
        
    return Concentration, diff
    
def relative_difference(reference,measurement):
    return(np.abs((measurement-reference)/reference))

def Return_Diff_Dataframe(df1, df2, by='Concentration'):
    Concentration = Calculate_Dataframe_Relative_Difference(df1, df2, by=by)[0]
    Length = len(Concentration)
    out = np.empty((Length, 7))
    out[:,0] = Concentration
    out[:,1:4] = Calculate_Dataframe_Relative_Difference(df1, df2, by=by)[1]
    out[:,4:] = Calculate_Dataframe_Relative_Difference(df1, df2, by=by, treated=True)[1]
    
    df_full = pd.DataFrame(data=out, columns=['Concentration', 'Red Error', 'Green Error', 'Blue Error', 'Treated Red Error', 
                                'Treated Green Error', 'Treated Blue Error'])
    df_avg = pd.DataFrame()
    df_avg['Concentration'] = out[:,0]
    df_avg['Before Treatment'] = out[:,1:4].mean(axis=1)
    df_avg['After Treatment'] = out[:,4:].mean(axis=1)
    
    return df_full, df_avg
    
    
def dataframe_calculate_bounds(dataframe, mult=1.5):
    Q1 = dataframe.quantile(0.25)
    Q3 = dataframe.quantile(0.75)
    IQR = Q3-Q1
    lower_bound = Q1 - mult*IQR
    upper_bound = Q3 + mult*IQR
    return lower_bound, upper_bound

def dataframe_remove_outliers(dataframe, Filter='Concentration', return_nan=False, mult=1.5):
    corrected = dataframe.copy()
    for i in test['{}'.format(Filter)].unique():
        
        crr_df = corrected[corrected['{}'.format(Filter)] == i]
        crr_lower_bound, crr_upper_bound = dataframe_calculate_bounds(crr_df, mult)
        corrected[crr_df < crr_lower_bound] = np.NaN
        corrected[crr_df > crr_upper_bound] = np.NaN
        if return_nan == False:
            corrected = corrected.fillna(value=crr_df.mean())
    return corrected


def color_polyfit(original_cmtx, corrected_cmtx, degree=3):
    coeff = np.empty((3, degree+1))
    for i in range(3):
        coeff[i,:] = np.polyfit(original_cmtx[:,i], corrected_cmtx[:,i], degree)
    return coeff

def img_polyfit(img, coeff):
    shape = img.shape
    new_shape = (img.shape[0]*img.shape[1], img.shape[2])
    img = img.reshape(new_shape)
    out = np.zeros(new_shape)
    
    poly_red = np.poly1d(coeff[0])
    poly_green = np.poly1d(coeff[1])
    poly_blue = np.poly1d(coeff[2])
    
    out = np.stack((poly_red(img[:,0]), poly_green(img[:,1]), poly_blue(img[:,2])), axis=1)
    
    return out.reshape(shape)

def array_polyfit(rgb, coeff):
    shape = rgb.shape

    out = np.zeros(shape)
    
    poly_red = np.poly1d(coeff[0])
    poly_green = np.poly1d(coeff[1])
    poly_blue = np.poly1d(coeff[2])
    
    out = np.stack((poly_red(rgb[:,0]), poly_green(rgb[:,1]), poly_blue(rgb[:,2])), axis=1)
    
    return out


def measure_error_df(cal, df):
    error = np.empty((len(cal['Concentration'].unique()),3))
    for i, j in enumerate(np.sort(cal['Concentration'].unique())):
        crr_red_error = np.abs(cal[cal['Concentration'] == j]['Red Channel'].values - 
                                df[df['Concentration'] == j]['TreatedRed'].values)/(cal[cal['Concentration'] == j]['Red Channel'].values)
        crr_green_error = np.abs(cal[cal['Concentration'] == j]['Green Channel'].values - 
                                  df[df['Concentration'] == j]['TreatedGreen'].values)/(cal[cal['Concentration'] == j]['Green Channel'].values)
        crr_blue_error = np.abs(cal[cal['Concentration'] == j]['Blue Channel'].values - 
                                  df[df['Concentration'] == j]['TreatedBlue'].values)/(cal[cal['Concentration'] == j]['Blue Channel'].values)
        error[i,:] = np.array([crr_red_error, crr_green_error, crr_blue_error]).reshape(1,3)
        
        results = pd.DataFrame()
        results['Concentration'] = cal['Concentration'].unique()
        results['Red Error'] = error[:,0]
        results['Green Error'] = error[:,1]
        results['Blue Error'] = error[:,2]
        
        
        
    return results, error, error.mean(axis=0), error.std(axis=0)

def calculate_df_dist_parameters(df, colorspace='RGB'):
    rgb = np.stack([df['Red Channel'], df['Green Channel'], df['Blue Channel']], axis=1)
    
    if colorspace == 'RGB':
        return np.array([rgb.mean(axis=0), rgb.std(axis=0)])
                   
    if colorspace =='LAB':
        lab = array_rgb2lab(rgb, CV=True)
        return np.array([lab.mean(axis=0), lab.std(axis=0)])

def index_samples(DataFrame, sample_index):
    df = DataFrame.copy()
    qrcode = np.empty(len(df.FileName), dtype='int')
    concentration = np.empty(len(df.FileName), dtype='float')
    for i, j in enumerate(df.FileName):
        qrcode[i] = j.split('uPad-')[1].split('.jpg')[0]
        concentration[i] = sample_index[sample_index['Device ID'] == qrcode[i]].Concentration
    df['QR Code'] = qrcode
    df['Concentration'] = concentration
    
    return df

def generate_colormap(df, analyte='pH'):
    df['Analyte'] = analyte
    df_avg = df_filter(df, by='Analyte').sort_values('Concentration')
    return np.stack([df_avg['Red Channel'], df_avg['Green Channel'], df_avg['Blue Channel']], axis=1)/255
    
def adjust_df(DataFrame, cal_mean, cal_std):
    df = DataFrame.copy()
    for i,j in enumerate(['Red', 'Green', 'Blue']):
        globals()['adjusted_{}'.format(j)] = cal_mean[i]+(DataFrame['Treated{}'.format(j)] - DataFrame['Treated{}'.format(j)].mean())*(cal_std[i]/DataFrame['Treated{}'.format(j)].std())
    df['TreatedRed'] = adjusted_Red
    df['TreatedGreen'] = adjusted_Green
    df['TreatedBlue'] = adjusted_Blue
    
    return df
    
def Directory(name):
    
    # Create directory ,it's a directory name which you are going to create.
    
    Directory_Name = name
    #try and catch block use to handle the exceptions.
    try:
        # Create  Directory  MyDirectory 
        os.mkdir(Directory_Name)
        #print if directory created successfully...
        print("Directory " , Directory_Name ,  " Created ") 
    except FileExistsError:
        ##print if directory already exists...
        print("Directory " , Directory_Name ,  " already exists...")

    return Directory_Name
    
def split_spot(spot, k):
    '''Randomly slips the spot array into smaller chuncks. Will drop pixels to make it divisible by k'''
    pixel_array = spot.reshape(spot.shape[0]*spot.shape[1],3) #Transforming 3D array into 2D
    np.random.shuffle(pixel_array) #Randomly shuffles the arra'
    div_lim = int(pixel_array.shape[0]/k)
    pixel_array = pixel_array[:(div_lim*k)]
    
    return np.array(np.split(pixel_array, k))

def non_zero_analysis_linear(im):
    '''Calculates the mean of non zero values in a given image, returning the mean [R,G,B] array'''
    red_mean = np.sum(im[:,0])/np.sum(im[:,0]!=0)
    green_mean = np.sum(im[:,1])/np.sum(im[:,1]!=0)
    blue_mean = np.sum(im[:,2])/np.sum(im[:,2]!=0)

    red_std = np.nanstd(np.where(np.isclose(im[:,0],0), np.nan, im[:,0]))
    green_std = np.nanstd(np.where(np.isclose(im[:,1],0), np.nan, im[:,1]))
    blue_std = np.nanstd(np.where(np.isclose(im[:,2],0), np.nan, im[:,2]))

    return np.array([red_mean,green_mean,blue_mean]), np.array([red_std, green_std, blue_std])

def non_zero_analysis_split(s):
    '''Calculates the mean of non zero values in a given image, returning the mean [R,G,B] array'''
    red_mean = s[:,:,0].sum(axis=1)/np.sum(s[:,:,0] != 0, axis=1)
    green_mean = s[:,:,1].sum(axis=1)/np.sum(s[:,:,1] != 0, axis=1)
    blue_mean = s[:,:,2].sum(axis=1)/np.sum(s[:,:,2] != 0, axis=1)

    red_std = np.nanstd(np.where(np.isclose(s[:,:,0],0), np.nan, s[:,:,0]), axis=1)
    green_std = np.nanstd(np.where(np.isclose(s[:,:,1],0), np.nan, s[:,:,1]), axis=1)
    blue_std = np.nanstd(np.where(np.isclose(s[:,:,2],0), np.nan, s[:,:,2]), axis=1)

    return np.stack([red_mean,green_mean,blue_mean], axis=1), np.stack([red_std, green_std, blue_std], axis=1)

def boolean_comparisson(array, boolean=[0,0,0]):
    return not np.array_equal(np.array(boolean), array)

def nonzero_stats(img, boolean =[0,0,0]):
    
    bin_map = np.logical_and(np.logical_and(img[:,:,0]==boolean[0],
                                                 img[:,:,1]==boolean[1]),img[:,:,2]==boolean[2])
    bin_map = np.invert(bin_map)
    filtered = img[bin_map]
    
    mean = filtered.mean(axis=(0))
    std = filtered.std(axis=(0))
    
    return mean, std, np.concatenate((mean, std), axis=0), bin_map

def colormatrix_total(ref_mask_colors, ref_mask_white, ref_total, full_output=False):
    
    '''Takes in a mask array, applies into the image array and takes the mean of the circles 
    according to the y coordinates. color is the orientation of the colored spots'''
    
    if full_output == True:
        colormatrix = np.zeros((4, 6))
        r = find_ref_ranges(ref_mask_colors)    
        colormatrix[0,:3] = nonzero_stats(ref_total[r[0,0]:r[0,1],r[0,2]:r[0,3],:])[0]
        colormatrix[0,3:6] = nonzero_stats(ref_total[r[0,0]:r[0,1],r[0,2]:r[0,3],:])[1]

        colormatrix[1,:3] = nonzero_stats(ref_total[r[1,0]:r[1,1],r[1,2]:r[1,3],:])[0]
        colormatrix[1,3:6] = nonzero_stats(ref_total[r[1,0]:r[1,1],r[1,2]:r[1,3],:])[1]

        colormatrix[2,:3] = nonzero_stats(ref_total[r[2,0]:r[2,1],r[2,2]:r[2,3],:])[0]
        colormatrix[2,3:6] = nonzero_stats(ref_total[r[2,0]:r[2,1],r[2,2]:r[2,3],:])[1]

        colormatrix[3,:3] = nonzero_stats(mask_img(ref_total,ref_mask_white))[0]
        colormatrix[3,3:6] = nonzero_stats(mask_img(ref_total,ref_mask_white))[1]

    else:    
        colormatrix = np.zeros((4, 3))
        r = find_ref_ranges(ref_mask_colors)    
        colormatrix[0,:] = nonzero_stats(ref_total[r[0,0]:r[0,1],r[0,2]:r[0,3],:])[0] 
        colormatrix[1,:] = nonzero_stats(ref_total[r[1,0]:r[1,1],r[1,2]:r[1,3],:])[0]
        colormatrix[2,:] = nonzero_stats(ref_total[r[2,0]:r[2,1],r[2,2]:r[2,3],:])[0]
        colormatrix[3,:] = nonzero_stats(mask_img(ref_total,ref_mask_white))[0]

    return colormatrix

def create_white_ref(mask, offset=[0,0]):
    from skimage.draw import circle
    ranges = find_ref_ranges(mask)
    radiusy = (ranges[:,1] - ranges[:,0])/2
    radiusx = (ranges[:,3] - ranges[:,2])/2
    
    centery = ranges[:,0]+radiusy
    centerx = ranges[:,2]+radiusx 
    
    distancex = np.mean([centerx[1]-centerx[0], centerx[2] - centerx[1]])
    distancey = np.mean([centery[1]-centery[0], centery[2] - centery[1]]).astype('int')
    
    positionx = (centerx[2] + distancex) + offset[0]
    positiony = (centery[2] + distancey) + offset [1]

    radii = np.mean([radiusx, radiusy]) ##circle radius
    
    white = np.zeros(mask.shape[:2]) 
    pixel = circle(positiony,positionx,radii)
    white[pixel[0], pixel[1]] = 1
    combined = mask + white
    
    return combined, white

def mask_img_inv(im, mask):
    '''Applies an inverse of a mask into an image'''
    outim = np.zeros(im.shape)
    for i in range(im.shape[2]):
        outim[:,:,i] = im[:,:,i]*(mask!=True)
    return outim

def color_transf(im_in, TotalGlobalRef, mask):
    '''Takes in reference (im_out) and input (im_in) images and converts them to the LAB colorspace.
    The function will take the distribution for each component and translate to the reference's mean and 
    shrink or expand for the reference's std. The function will them reconvert the images into the RGB colorspace'''
    
    img_lab = cv2.cvtColor(im_in, cv2.COLOR_RGB2LAB)
    
    adj = adjust_dist(img_lab, TotalGlobalRef.T)

    im = np.clip(adj, 0, 255).astype('uint8') 

    out = cv2.cvtColor(im, cv2.COLOR_LAB2RGB)
    
    return mask_img(out, mask)

def adjust_dist(in_img, GlobalRefCh, boolean=[0,128,128]):
    '''This function takes in a distribution and adjusts its std and mean to that of the out distribution
    The report mode outputs the std and mean of the adjusted distribution
    
    Total Ref should be in shape [[meanL, meana, meanb], [stdL, stda, stdb]]'''

    mean, std = nonzero_stats(in_img, boolean=boolean)[:2]
    
    adjusted_dist = np.float64(GlobalRefCh[0] + (in_img - mean)*(GlobalRefCh[1]/std))
    
    return adjusted_dist

def linear_transform(img, card_ref, global_ref, raw_out=False):
    '''MATRIX TRANSFORM METHOD - Calculates a transformation matrix from reference matrices (img_spot, ref_spot) by
    using the least squares method. This transformation matrix is them applied pixelwise in the image.'''
    
    transform = transform_matrix(card_ref, global_ref)
    return transform, apply_transform(img, transform, raw_out=raw_out)

def transform_matrix(mtx, ref):
    '''Calculates a transformation matrix between two matrices by using least square method'''
              
    transform = np.linalg.lstsq(mtx,ref, rcond=None)

    return transform[0]

def apply_transform(in_img, transform, raw_out=False):
    '''Applies a givem transformation matrix pixelwise in a image'''
    
    treated_img = np.tensordot(in_img,transform,axes=(-1,0))
    if raw_out == True:
        return treated_img
    elif raw_out == False:
        return np.clip(treated_img, 0, 255).astype('uint8')
    
def generate_global_ref(img_ref, colorspace='RGB', offset=[0,0]):
    if colorspace == 'RGB':
        ref_mask = threshold(PCA_im(img_ref)[:,:,0])
        total_ref_mask, white_mask = create_white_ref(ref_mask, offset=offset)
        total_ref = mask_img(img_ref, total_ref_mask)
        total_ref_lab = cv2.cvtColor(total_ref.astype('uint8'), cv2.COLOR_RGB2LAB)
        lab_ref = np.concatenate(nonzero_stats(total_ref_lab, boolean=[0,128,128])[:2], axis=0).reshape(1,6)
        cmtx = colormatrix_total(ref_mask, white_mask, total_ref, full_output=True)

    if colorspace == 'LAB':
        ref_mask = threshold(PCA_im(img_ref)[:,:,0])
        total_ref_mask, white_mask = create_white_ref(ref_mask, offset=offset)
        
        total_ref = mask_img(img_ref, total_ref_mask)
        total_ref_lab = cv2.cvtColor(total_ref.astype('uint8'), cv2.COLOR_RGB2LAB)
        
        lab_ref = np.concatenate(nonzero_stats(total_ref_lab, boolean=[0,128,128])[:2], axis=0).reshape(1,6)
        cmtx = colormatrix_total(ref_mask, white_mask, mask_img(total_ref_lab, total_ref_mask), 
                                                            full_output=True)
    
    return np.concatenate((lab_ref, cmtx), axis=0)

# def generate_global_ref(img_ref, colorspace='RGB', offset=[0,0]):
#     if colorspace == 'RGB':
#         ref_mask = threshold(PCA_mask(img_ref))
#         total_ref_mask, white_mask = create_white_ref(ref_mask, offset=offset)
#         total_ref = mask_img(img_ref, total_ref_mask)
#         total_ref_lab = cv2.cvtColor(total_ref.astype('uint8'), cv2.COLOR_RGB2LAB)
#         lab_ref = np.concatenate(nonzero_stats(total_ref_lab, boolean=[0,128,128])[:2], axis=0).reshape(1,6)
#         cmtx = colormatrix_total(ref_mask, white_mask, total_ref, full_output=True)

#     if colorspace == 'LAB':
#         ref_mask = threshold(PCA_mask(img_ref))
#         total_ref_mask, white_mask = create_white_ref(ref_mask, offset=offset)
        
#         total_ref = mask_img(img_ref, total_ref_mask)
#         total_ref_lab = cv2.cvtColor(total_ref.astype('uint8'), cv2.COLOR_RGB2LAB)
        
#         lab_ref = np.concatenate(nonzero_stats(total_ref_lab, boolean=[0,128,128])[:2], axis=0).reshape(1,6)
#         cmtx = colormatrix_total(ref_mask, white_mask, mask_img(total_ref_lab, total_ref_mask), 
#                                                             full_output=True)
    
#     return np.concatenate((lab_ref, cmtx), axis=0)


def find_ref_ranges(masked_img, sort=2):
    '''Given a spot mask, it will segment the spots by fiding the ranges from calculation of the spots edges.
    color is the orientation of the colored spots. Sorted by x distance (left to right) as default.
    the sort argument may be changed to alter the sorting. To sort by y change sort to 0'''
    from skimage.measure import find_contours
    idx = 0
    cont = np.array(find_contours(masked_img, .7, fully_connected='high'))
    spot1 = []
    spot2 = []
    spot3 = []
    ctr = []
    
    if cont.shape[0] >= 3:
        for i in range(len(cont)):
            if len(cont[i]) > 50:
                ctr.append(cont[i])
        ctr = np.array(ctr)
                
        if ctr.shape[0] == 3:
            for i in range(2):
                spot1.append(np.min(ctr[0], axis=0)[i])
                spot1.append(np.max(ctr[0], axis=0)[i])

                spot2.append(np.min(ctr[1], axis=0)[i])
                spot2.append(np.max(ctr[1], axis=0)[i])

                spot3.append(np.min(ctr[2], axis=0)[i])
                spot3.append(np.max(ctr[2], axis=0)[i])
            ranges = np.array([spot1, spot2, spot3])
                
            ranges = ranges[ranges[:,sort].argsort()]     

            return np.array(ranges, dtype='int')

        else:
            return(np.zeros((3,4), dtype='int'))
        
    else:
        return(np.zeros((3,4), dtype='int'))






################################################################################################

# ## Data Import
def find_sizes(filename):
    names = np.array(glob.glob(filename))
    image_array = []

    for i, file in enumerate(glob.glob(filename)):
        crr_im = plt.imread(file)
        crr_shape = crr_im.shape
        image_array.append(crr_shape)

    return np.array(image_array), np.array(names)


def img_import(filename, crop = (False, 0, 0, 0, 0), padding=(False, 0, 0,0)):
    '''Imports all images with the given filename from the file. For multiple images use: *.jpg, *.png, etc.
    Returns a 2 dimensional array with array[0] being the images and array[1] the filenames. Can crop the imported
    images in xy by suppling crop (x0, xf, y0, yf). If cropping, crop[0] should be true. All images should be the '''

    if crop[0] == False:

        if padding[0] == False:
            names = np.array(glob.glob(filename))
            image_array = []
            size_array = []

            for i, file in enumerate(glob.glob(filename)):
                crr_im = plt.imread(file)
                crr_shape = crr_im.shape
                image_array.append(crr_im)
                size_array.append(crr_shape)

            return np.array(image_array), np.array(names), np.array(size_array)

        if padding[0] == True:
            names = np.array(glob.glob(filename))
            image_array = []
            size_array = []

            for i, file in enumerate(glob.glob(filename)):
                crr_im = plt.imread(file)
                crr_shape = crr_im.shape
                img_pad = np.zeros((padding[1], padding[2], padding[3])).astype('uint8')
                img_pad[0:crr_shape[0], 0:crr_shape[1]] = crr_im
                image_array.append(img_pad)
                size_array.append(crr_shape)
                
            return np.array(image_array), np.array(names), np.array(size_array)

    if crop[0] == True:
        names = np.array(glob.glob(filename))
        image_array = []
        size_array = []

        for i, file in enumerate(glob.glob(filename)):
            crr_im = plt.imread(file)
            crr_shape = crr_im.shape
            image_array.append(crr_im[crop[1]:crop[2],crop[3]:crop[4]])
            size_array.append(crr_shape)

        return np.array(image_array), np.array(names), np.array(size_array)
        


# ## Spot Detection & Segmentation

def PCA_mask(im_in, dimensions=2):
    pca = PCA_im(im_in, dimensions=dimensions)
    return np.sqrt(np.power(pca[:,:,0],2) + np.power(pca[:,:,1],2))

def PCA_im(im_in, dimensions=2):
    '''Applies principal component analysis in a image array and return a [x,y,components] array that can
    be read as an image'''

    new_shape = (im_in.shape[0]*im_in.shape[1],im_in.shape[2])
    pca = sd.PCA(n_components=dimensions)
    pcomponents = pca.fit_transform(im_in.reshape(new_shape))
    out = pcomponents.reshape(im_in.shape[0],im_in.shape[1],dimensions)

    return out


def threshold(im, threshold=threshold_li, reverse=False):
    '''Creates an image mask from a given threshold'''
    thr = threshold(im)

    if reverse == False:
        mask = im>thr
    else:
        mask = im<thr
    return mask

def mask_img(im, mask):
    '''Applies an mask into an image'''
    outim = np.zeros(im.shape)
    for i in range(im.shape[2]):
        outim[:,:,i] = im[:,:,i]*mask
    return outim

def mask_reconstruction(mask, filters=(1,20,30)):
    for i in range(filters[0]):
        mask = morpho.erosion(mask)
    for i in range(filters[1]):
        mask = morpho.dilation(mask)
    for i in range(filters[2]):
        mask = morpho.erosion(mask)
    return mask

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def sort_circles(ranges, n=5):
    radiusy = (ranges[:,1] - ranges[:,0])/2
    radiusx = (ranges[:,3] - ranges[:,2])/2
    
    centery = ranges[:,0]+radiusy
    centerx = ranges[:,2]+radiusx

    centroidx = np.mean(centerx)
    centroidy = np.mean(centery)
    
    centery = centery - centroidx
    centerx = centerx - centroidy

    sort = cart2pol(centerx, centery)[1]
    
    ranges_sorted = ranges[sort.argsort()] #L3, L4, L0, L1, L2
    
    if n == 5:
        spots = np.array([ranges_sorted[2], ranges_sorted[3], ranges_sorted[4], ranges_sorted[0], ranges_sorted[1]])
    else:
        spots = np.array(ranges_sorted)
    
#     return ranges, sort, ranges_sorted 
    return spots #L0, L1, L2, L3, L4


def find_ranges(mask, pixel_threshold=50, n=5):
    from skimage.measure import find_contours
    cont = np.array(find_contours(mask, .7, fully_connected="high"), dtype=object)
    for i in range(n):
        locals()['spot{}'.format(i)] = []
    ctr = []
    ranges = []

    if cont.shape[0] >= n:
        for i in range(cont.shape[0]):
            if len(cont[i]) > pixel_threshold:
                ctr.append(cont[i])
        ctr = np.array(ctr, dtype=object)
        ctr = ctr[np.argsort([(i.shape[0]*i.shape[1]) for i in ctr])[::-1]]
        ctr = ctr[0:n]
        if ctr.shape[0] == n:
            for k in range(n):
                for i in range(2):
                    locals()['spot{}'.format(k)].append(np.min(ctr[k], axis=0)[i])
                    locals()['spot{}'.format(k)].append(np.max(ctr[k], axis=0)[i])
                ranges.append(locals()['spot{}'.format(k)])

            ranges = np.array(ranges)
            
        else:
            return(np.zeros((3,5), dtype='int'))
        
        return sort_circles(ranges, n).astype('int')

    else:
        return(np.zeros((3,5), dtype='int')) 

def crop_spots(img, ranges, n=5):
    out = []
    for i in range(n):
        out.append(img[ranges[i,0]:ranges[i,1], ranges[i,2]:ranges[i,3]])
    # return np.array(out, dtype=object)
    return out

def extract_spots(img, crop, filters=(1,20,30), n=5):
    img = img[crop[0]:crop[1], crop[2]:crop[3]]
    mask = mask_reconstruction(threshold(PCA_mask(img)), filters)
    masked_img = mask_img(img,mask)
    ranges = find_ranges(mask, n=n) #L0, L1 .... L4
    spots = crop_spots(masked_img, ranges,n=n)
    return  mask, masked_img, spots

def extract_ref_spots(img, crop):
    img = img[crop[0]:crop[1], crop[2]:crop[3]]
    PCA = PCA_mask(img)
    mask = mask_reconstruction(threshold(PCA))
    masked_img = mask_img(img,mask)
    ranges = find_ref_ranges(mask) #L0, L1 .... L4
    spots = crop_spots(masked_img, ranges, n=3)
    return  mask, masked_img, spots

# ## Data extraction

def non_zero_analysis(im):
    '''Calculates the mean of non zero values in a given image, returning the mean [R,G,B] array'''
    red_mean = np.sum(im[:,:,0])/np.sum(im[:,:,0]!=0)
    green_mean = np.sum(im[:,:,1])/np.sum(im[:,:,1]!=0)
    blue_mean = np.sum(im[:,:,2])/np.sum(im[:,:,2]!=0)

    red_std = np.nanstd(np.where(np.isclose(im[:,:,0],0), np.nan, im[:,:,0]))
    green_std = np.nanstd(np.where(np.isclose(im[:,:,1],0), np.nan, im[:,:,1]))
    blue_std = np.nanstd(np.where(np.isclose(im[:,:,2],0), np.nan, im[:,:,2]))

    return np.array([red_mean,green_mean,blue_mean]), np.array([red_std, green_std, blue_std])

def average_spots(imgs):
    data = []
    for i in range(imgs.shape[0]):
        data.append(non_zero_analysis(imgs[i]))
    data = np.array(data)
    mean = np.mean(data[:,0], axis=0)
    n = data.shape[0]
    std = np.sqrt(np.sum(data[:,1]**2, axis=0))/n
    return mean, std

def calculate_concentration(sample, calibration, model, output = 'pca'):
    pca = sd.PCA(n_components=2)
    pca = pca.fit(calibration)
    pca_value = pca.transform(sample.reshape((1,3)))

    if output == 'pca':
        return(pca_value[0,0])
    if output == 'concentration':
        return model(pca_value[0,0])

def generate_DataFrame(crop, filetype='*.bmp', n_spots=5, precrop=(False, 0, 0, 0, 0), padding=(False, 0, 0,0), 
                       export_spots=False, out_format='.jpg', k=1, filters=(1,20,30)):
    images, filenames, sizes = img_import(filetype, crop=precrop, padding = padding)
    files = filenames

    filenames = np.repeat(filenames, n_spots*k, axis=0) #change for different n_spots

    rgb = np.zeros(((images.shape[0]*n_spots*k), images.shape[3]))
    std = np.zeros(((images.shape[0]*n_spots*k), images.shape[3]))
    analyte = np.zeros(filenames.shape).astype('str')
    concentration = np.zeros(filenames.shape).astype('str')
    data = pd.DataFrame()
    label = np.zeros(filenames.shape).astype('str')

    if export_spots == True:
        cwd = os.getcwd()
        folder = Directory('Exported Spots')


    for i in range(images.shape[0]):
        print('Analysing image {}'.format(files[i]))
        crr_img = images[i][:sizes[i,0], :sizes[i,1]]
        spots = extract_spots(crr_img, crop, filters, n=n_spots)[2]
        label[(n_spots*i)*k:(n_spots*i+n_spots)*k] = np.array(['L{}'.format(x) for x in range(n_spots)]).repeat(k)
        
        for j in range(n_spots):
            split = split_spot(spots[j],k)
            
            if k==1:
                rgb[n_spots*i+j], std[n_spots*i+j] = non_zero_analysis_split(split)
                
            else:
                rgb[((n_spots*i+j)*k):((n_spots*i+j)*k + k)] = non_zero_analysis_split(split)[0]
                std[((n_spots*i+j)*k):((n_spots*i+j)*k + k)] = non_zero_analysis_split(split)[1]
            
            if export_spots == True:
                crr_spot = Image.fromarray(spots[j].astype('uint8'))
                crr_spot.save(cwd+'/'+folder+'/'+'[S]{}_[F]{}_{}'.format(label[j], files[i], out_format))
                
                

    data['FileName'] = filenames
    data['Spot'] = label
    data['Red Channel'] = rgb[:,0]
    data['Green Channel'] = rgb[:,1]
    data['Blue Channel'] = rgb[:,2]
    data['R std'] = std[:,0]
    data['G std'] = std[:,1]
    data['B std'] = std[:,2]

    return data
# ## Data Processing

def df_filter(df,by='FileName', columns=['FileName', 'Analyte', 'Concentration', 'Red Channel', 'Green Channel',
       'Blue Channel', 'R std', 'G std', 'B std'], treated=False):

    if by == 'FileName':
        out = pd.DataFrame(columns=columns)
        filenames = df.FileName.unique()
        for i in filenames:
            filtered_DF = df[df.FileName == i]
            file_mean = filtered_DF.mean()
            file_count = df.FileName.value_counts()[i]
            new_data = pd.DataFrame(data = np.zeros((1,len(columns))), columns=columns)
            new_data['FileName'] = i
            new_data['Analyte'] = filtered_DF.Analyte.unique()[0]
            new_data['Concentration'] = filtered_DF.Concentration.unique()[0]
            new_data['Red Channel'] = file_mean['Red Channel']
            new_data['Green Channel'] = file_mean['Green Channel']
            new_data['Blue Channel'] = file_mean['Blue Channel']
            new_data['R std'] = np.sqrt(np.sum(filtered_DF['R std']**2))/file_count
            new_data['G std'] = np.sqrt(np.sum(filtered_DF['G std']**2))/file_count
            new_data['B std'] = np.sqrt(np.sum(filtered_DF['B std']**2))/file_count
            out = out.append(new_data)

        return out.reset_index(drop=True)

    if by == 'Analyte':
        if treated == True:
            columns=['FileName', 'Analyte', 'Concentration', 'Red Channel', 'Green Channel',
       'Blue Channel', 'R std', 'G std', 'B std', 'TreatedRed', 'TreatedGreen', 'TreatedBlue', 'TR std', 'TG std', 'TB std']
        new_columns = columns[1:]
        out = pd.DataFrame(columns=new_columns)
        analytes = df.Analyte.unique()
        concentrations = df.Concentration.unique()
        for i in analytes:
            for j in concentrations:
                if df[(df.Analyte == i) & (df.Concentration == j)].empty:
                    pass
                else:
                    filtered_DF = df[(df.Analyte == i) & (df.Concentration == j)]
                    file_mean = filtered_DF.mean()
                    file_count = len(df[(df.Analyte == i) & (df.Concentration == j)])
                    new_data = pd.DataFrame(data = np.zeros((1,len(new_columns))), columns=new_columns)
                    new_data['Analyte'] = i
                    new_data['Concentration'] = j
                    new_data['Red Channel'] = file_mean['Red Channel']
                    new_data['Green Channel'] = file_mean['Green Channel']
                    new_data['Blue Channel'] = file_mean['Blue Channel']
                    if treated==True:
                        new_data['TreatedRed'] = file_mean['TreatedRed']
                        new_data['TreatedGreen'] = file_mean['TreatedGreen']
                        new_data['TreatedBlue'] = file_mean['TreatedBlue']
                        new_data['TR std'] = filtered_DF['TreatedRed'].describe()['std']
                        new_data['TG std'] = filtered_DF['TreatedGreen'].describe()['std']
                        new_data['TB std'] = filtered_DF['TreatedBlue'].describe()['std']
                    new_data['R std'] = np.sqrt(np.sum(filtered_DF['R std']**2))/file_count
                    new_data['G std'] = np.sqrt(np.sum(filtered_DF['G std']**2))/file_count
                    new_data['B std'] = np.sqrt(np.sum(filtered_DF['B std']**2))/file_count
                    out = out.append(new_data)


        return out.reset_index(drop=True)


def calculate_PCA(DataFrame, Save = False, Treated=False):
    
    rgb = np.stack((DataFrame['Red Channel'], DataFrame['Green Channel'], DataFrame['Blue Channel']), axis=1)
    std = np.stack((DataFrame['R std'], DataFrame['G std'], DataFrame['B std']), axis=1)
    pca = sd.PCA(n_components=2)
    pcomponents = pca.fit_transform(rgb)
    plusstd = pca.fit_transform(rgb+std)
    minusstd = pca.fit_transform(rgb-std)
    error = np.abs(plusstd-minusstd)
    out = DataFrame.copy()

    out['PC1'] = pcomponents[:,0]
    out['PC2'] = pcomponents[:,1]
    out['PC1 Error'] = error[:,0]
    out['PC2 Error'] = error[:,1]
    
    if Treated == True:
        rgb = np.stack((DataFrame['TreatedRed'], DataFrame['TreatedGreen'], DataFrame['TreatedBlue']), axis=1)
        pca = sd.PCA(n_components=2)
        pcomponents = pca.fit_transform(rgb)

        out['TreatedPC1'] = pcomponents[:,0]
        out['TreatedPC2'] = pcomponents[:,1]
        out = out.reset_index(drop=True)
        
    if Save == True:
        file = input('Enter File Name: ')
        out.to_csv('{} PCA.csv'.format(file))

    return out

def array_rgb2hsv(array):
    hsv = np.ones([array.shape[0],1,3])
    hsv[:,0] = array
    hsv = rgb2hsv(hsv)
    hsv = hsv[:,0,:]

    return hsv

def array_hsv2rgb(array):
    rgb = np.ones([array.shape[0],1,3])
    rgb[:,0] = array
    rgb = hsv2rgb(rgb)
    rgb = rgb[:,0,:]

    return rgb

def array_rgb2lab(array, CV=False):
    rgb = np.ones([array.shape[0],1,3])
    rgb[:,0] = array
    if CV == False:
        lab = rgb2lab(rgb)
    if CV == True:
        lab = cv2.cvtColor(rgb.astype('uint8'), cv2.COLOR_RGB2LAB)
    lab = lab[:,0,:]
    
    return lab

def array_lab2rgb(array, CV=False):
    lab = np.ones([array.shape[0],1,3])
    lab[:,0] = array
    if CV == False:
        rgb = lab2rgb(lab)
    if CV == True:
        rgb = cv2.cvtColor(lab.astype('uint8'), cv2.COLOR_LAB2RGB)
    
    rgb = rgb[:,0,:]
    
    return rgb

def calculate_HSV(DataFrame, Save = False, Treated = False):
    
    rgb = np.stack((DataFrame['Red Channel'], DataFrame['Green Channel'], DataFrame['Blue Channel']), axis=1)/255
    std = np.stack((DataFrame['R std'], DataFrame['G std'], DataFrame['B std']), axis=1)/255

    hsv =  array_rgb2hsv(rgb)
    plusstd = array_rgb2hsv(rgb+std)
    minusstd = array_rgb2hsv(rgb-std)
    error = np.abs(plusstd-minusstd)
    out = DataFrame.copy()

    out['H Channel'] = hsv[:,0]
    out['S Channel'] = hsv[:,1]
    out['V Channel'] = hsv[:,2]

    out['H std'] = error[:,0]
    out['S std'] = error[:,1]
    out['V std'] = error[:,2]
    
    if Treated == True:
        rgb = np.stack((DataFrame['TreatedRed'], DataFrame['TreatedGreen'], DataFrame['TreatedBlue']), axis=1)/255
        hsv =  array_rgb2hsv(rgb)

        out['TreatedH'] = hsv[:,0]
        out['TreatedS'] = hsv[:,1]
        out['TreatedV'] = hsv[:,2]
        out = out.reset_index(drop=True)
        
        

    if Save == True:
        file = input('Enter File Name: ')
        out.to_csv('{} HSV.csv'.format(file))

    return out
    
def calculate_colorspaces(df, Treated=False):
    if Treated == False:
        return calculate_LAB(calculate_HSV(calculate_PCA(df)))
    if Treated == True:
        return calculate_LAB(calculate_HSV(calculate_PCA(df, Treated=True), Treated=True), Treated=True)

def calculate_LAB(DataFrame, Save = False, Treated=False):
    rgb = np.stack((DataFrame['Red Channel'], DataFrame['Green Channel'], DataFrame['Blue Channel']), axis=1)/255
    std = np.stack((DataFrame['R std'], DataFrame['G std'], DataFrame['B std']), axis=1)/255

    lab =  array_rgb2lab(rgb)
    plusstd = array_rgb2lab(rgb+std)
    minusstd = array_rgb2lab(rgb-std)
    error = np.abs(plusstd-minusstd)
    out = DataFrame.copy()

    out['L Channel'] = lab[:,0]
    out['a Channel'] = lab[:,1]
    out['b Channel'] = lab[:,2]

    out['L std'] = error[:,0]
    out['a std'] = error[:,1]
    out['b std'] = error[:,2]
    
    if Treated == True:
        rgb = np.stack((DataFrame['TreatedRed'], DataFrame['TreatedGreen'], DataFrame['TreatedBlue']), axis=1)/255
        lab =  array_rgb2lab(rgb)

        out['TreatedL'] = lab[:,0]
        out['Treateda'] = lab[:,1]
        out['Treatedb'] = lab[:,2]
        out = out.reset_index(drop=True)
        
    if Save == True:
        file = input('Enter File Name: ')
        out.to_csv('{} lab.csv'.format(file))

    return out

def Treat_DataFrame(DataFrame, color_ref_in, color_ref_out, Save=False):
    df = DataFrame.copy()
    transform_mtx = np.linalg.lstsq(color_ref_in, color_ref_out, rcond=None)[0]
    new_data = np.zeros((len(df),3))
    for i in range(len(df)):
            red = df.loc[i, 'Red Channel']
            green = df.loc[i, 'Green Channel']
            blue = df.loc[i, 'Blue Channel']
            crr_data = np.array([red, green, blue]).reshape(1,3)
            new_data[i] = np.matmul(crr_data, transform_mtx)
            df['TreatedRed'] = new_data[:,0]
            df['TreatedGreen'] = new_data[:,1]
            df['TreatedBlue'] = new_data[:,2]

    if Save == True:
        file = input('Enter File Name: ')
        df.to_excel('{}_Treated.xlsx'.format(file))

    return df

def Apply_Poly_DataFrame(DataFrame, coeff, files,  Save=False, clip=True):
    df = DataFrame.copy()
    new_data = np.zeros((len(df),3))
    for j in range(files.shape[0]):
        crr_df = df[df['FileName']==files[j]].reset_index()
        crr_coeff = coeff[j]
        for i in range(len(crr_df)):
            red = crr_df.loc[i, 'Red Channel']
            green = crr_df.loc[i, 'Green Channel']
            blue = crr_df.loc[i, 'Blue Channel']
            crr_data = np.array([red, green, blue]).reshape(1,3)
            if clip == True:
                new_data[5*j+i] = np.clip(array_polyfit(crr_data, crr_coeff), 0, 255)
            if clip == False:
                new_data[5*j+i] = array_polyfit(crr_data, crr_coeff)
        

    df['TreatedRed'] = new_data[:,0]
    df['TreatedGreen'] = new_data[:,1]
    df['TreatedBlue'] = new_data[:,2]

        
    if Save == True:
        file = input('Enter File Name: ')
        df.to_excel('{}_Treated.xlsx'.format(file))
                
                
    return df

def Apply_Trs_DataFrame(DataFrame, trs, files,  Save=False, colorspace='RGB', adj_df=(False, [0,0,0], [0,0,0])):
    df = DataFrame.copy()
    new_data = np.zeros((len(df),3))
    for j in range(files.shape[0]):
        crr_df = df[df['FileName']==files[j]].reset_index()
        crr_trs = trs[j]
        for i in range(len(crr_df)):
            red = crr_df.loc[i, 'Red Channel']
            green = crr_df.loc[i, 'Green Channel']
            blue = crr_df.loc[i, 'Blue Channel']
            crr_data = np.array([red, green, blue]).reshape(1,3)
            
            if colorspace == 'RGB':
                new_data[5*j+i] = np.matmul(crr_data, crr_trs)
                    
            if colorspace == 'LAB':
                crr_data_lab = array_rgb2lab(crr_data, CV=True)
                new_data[5*j+i] = np.matmul(crr_data_lab, crr_trs)
                
                        
    if adj_df[0] == True:
        new_data = (adj_df[1] + (new_data - new_data.mean(axis=0))*(adj_df[2]/new_data.std(axis=0)))
        
    if colorspace == 'LAB':
        new_data = array_lab2rgb(new_data, CV=True)

    df['TreatedRed'] = new_data[:,0]
    df['TreatedGreen'] = new_data[:,1]
    df['TreatedBlue'] = new_data[:,2]

        
    if Save == True:
        file = input('Enter File Name: ')
        df.to_excel('{}_Treated.xlsx'.format(file))
                
                
    return df



# ## Data Visualization

def create_color_block(rgb, shape=(100, 100, 3)):
    block = np.zeros(shape)
    block[:,:] = rgb
    return block.astype('uint8')

# Horizontal

def plot_rgb_results(rgb,sd,name, figsize, title):
    rows = rgb.shape[0]
    fig, axes = plt.subplots(3, rows, figsize=figsize)
    ax = axes.ravel()
    for i in range(rows):
        center = rgb[i]
        left = np.clip((rgb[i] - sd[i]), 0, 255)
        right = np.clip((rgb[i] + sd[i]), 0, 255)
        ax[i].imshow(create_color_block(left))
        ax[i].axis('off')
        ax[i].set_title(name[i]+' - SD')
        ax[i+1*rows].imshow(create_color_block(center))
        ax[i+1*rows].axis('off')
        ax[i+1*rows].set_title(name[i])
        ax[i+2*rows].imshow(create_color_block(right))
        ax[i+2*rows].axis('off')
        ax[i+2*rows].set_title(name[i]+' + SD')
    plt.suptitle('{} Color Output'.format(title))
    plt.savefig('{}COLORBLOCK.png'.format(title))
    plt.show()

def plot_rgb_results_nostd(rgb, name, figsize, title, fz):
    rows = rgb.shape[0]
    fig, axes = plt.subplots(1, rows, figsize=figsize)
    ax = axes.ravel()
    for i in range(rows):
        center = rgb[i]
        ax[i].imshow(create_color_block(center))
        ax[i].axis('off')
        ax[i].set_title(name[i], fontsize=fz)
    plt.suptitle('{} Color Output'.format(title))
    plt.savefig('{}COLORBLOCK.png'.format(title))
    plt.show()

def plot_COLORBLOCK_avg_nostd(df_avg, title, treated = False, size=(10,6), fz=10):
    if treated == False:
        df_avg = df_avg.sort_values(['Concentration'])
        rgb_df_avg = np.stack([df_avg['Red Channel'], df_avg['Green Channel'], df_avg['Blue Channel']], axis=1)
        concentration = np.array(df_avg['Concentration']).astype('str')
        plot_rgb_results_nostd(rgb_df_avg, concentration, size, title, fz)
    if treated == True:
        df_avg = df_avg.sort_values(['Concentration'])
        rgb_df_avg = np.stack([df_avg['TreatedRed'], df_avg['TreatedGreen'], df_avg['TreatedBlue']], axis=1)
        concentration = np.array(df_avg['Concentration']).astype('str')
        plot_rgb_results_nostd(rgb_df_avg, concentration, size, title, fz)


#Vertical

# def plot_rgb_results(rgb,sd,name):
#     rows = rgb.shape[0]
#     fig, axes = plt.subplots(rows, 3)
#     ax = axes.ravel()
#     for i in range(rows):
#         center = rgb[i]
#         left = rgb[i] - sd[i]
#         right = rgb[i] + sd[i]
#         ax[3*i].imshow(create_color_block(left))
#         ax[3*i].axis('off')
#         ax[3*i].set_title(name[i]+' - SD')
#         ax[3*i+1].imshow(create_color_block(center))
#         ax[3*i+1].axis('off')
#         ax[3*i+1].set_title(name[i])
#         ax[3*i+2].imshow(create_color_block(right))
#         ax[3*i+2].axis('off')
#         ax[3*i+2].set_title(name[i]+' + SD')
#     plt.show()

def plot_RGB_avg(df_avg, xlabel, title, ylabel='Channel'):
    plt.figure(figsize=(8,6))
    plt.errorbar(df_avg['Concentration'], df_avg['Red Channel'], df_avg['R std'], fmt='none', color='r')
    plt.plot(df_avg['Concentration'], df_avg['Red Channel'],'--o', color='r', label='R std')

    plt.errorbar(df_avg['Concentration'], df_avg['Green Channel'], df_avg['G std'], fmt='none', color='g')
    plt.plot(df_avg['Concentration'], df_avg['Green Channel'],'--o', color='g', label='G std')

    plt.errorbar(df_avg['Concentration'], df_avg['Blue Channel'], df_avg['B std'], fmt='none', color='b')
    plt.plot(df_avg['Concentration'], df_avg['Blue Channel'],'--o', color='b', label='B std')
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig('{}AVG_RGB.png'.format(title), dpi = 300)
    plt.show()

def plot_PCA(df, xlabel, title, ylabel='Principal Component 1'):
    PCA_Full = calculate_PCA(df, Save=False)
    plt.figure(figsize=(8,4))
    plt.plot(PCA_Full['Concentration'], PCA_Full['PC1'],'o', color='b', label='PC1')
    plt.plot(PCA_Full['Concentration'], PCA_Full['PC2'],'o', color='gray', alpha=.2, label='PC2')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig('{}_PCA.png'.format(title), dpi = 300)
    plt.show()

def plot_PCA_avg(df_avg, xlabel, title, ylabel='Principal Component 1'):
    PCA = calculate_PCA(df_avg, Save=False)
    plt.figure(figsize=(8,4))
    plt.errorbar(PCA['Concentration'], PCA['PC1'], PCA['PC1 Error'], fmt='none', color='r', label='PC1 Error')
    plt.plot(PCA['Concentration'], PCA['PC1'],'o', color='b', label='PC1')
    plt.plot(PCA['Concentration'], PCA['PC2'],'o', color='gray', alpha=.2, label='PC2')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig('{}AVG_PCA.png'.format(title), dpi = 300)
    plt.show()

def plot_COLORBLOCK_avg(df_avg, title, treated = False):
    if treated == False:
        df_avg = df_avg.sort_values(['Concentration'])
        rgb_df_avg = np.stack([df_avg['Red Channel'], df_avg['Green Channel'], df_avg['Blue Channel']], axis=1)
        std_df_avg = np.stack([df_avg['R std'], df_avg['G std'], df_avg['B std']], axis=1)
        concentration = np.array(df_avg['Concentration']).astype('str')
        plot_rgb_results(rgb_df_avg, std_df_avg, concentration, (10,6), title)
    if treated == True:
        df_avg = df_avg.sort_values(['Concentration'])
        rgb_df_avg = np.stack([df_avg['TreatedRed'], df_avg['TreatedGreen'], df_avg['TreatedBlue']], axis=1)
        std_df_avg = np.stack([df_avg['TR std'], df_avg['TG std'], df_avg['TB std']], axis=1)
        concentration = np.array(df_avg['Concentration']).astype('str')
        plot_rgb_results(rgb_df_avg, std_df_avg, concentration, (10,6), title)

def plot_boxplot(df, xlabel, title, xticks, ylabel='Principal Component 1'):
    PCA_Full = calculate_PCA(df, Save=False)
    plt.figure(figsize=(8,4))
    plot_data = PCA_Full
    positions = []
    data = []
    for i in plot_data.Concentration.unique():
        positions.append(i)
        data.append(PCA_Full[PCA_Full.Concentration == i]['PC1'])

    plt.boxplot(data, positions=positions, widths=.2, showmeans=True, notch=True)
    plt.xticks(xticks, xticks)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig('{}_BoxPlot.png'.format(title), dpi = 300)
    plt.show()
