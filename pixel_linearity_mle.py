#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import esutil
from scipy import ndimage
import sys
import galsim
import matplotlib.cm as cm
import cdmodel_functions


def apply_cdmodel (im, factor=1):
    """
    Uses galsim to apply "cdmodel" to an input image. 
    "cdmodel" ('charge deflection') is an implementation of the 'brightter-fatter'
    model by Antilogus et al 2014 as performed by Gruen et al 2015.
    """
    (aL,aR,aB,aT) = cdmodel_functions.readmeanmatrices()
    cd = galsim.cdmodel.BaseCDModel (factor*aL,factor*aR,factor*aB,factor*aT)
    im_out=cd.applyForward(im)

    return im_out


def get_simulated_array (delta_time=10, n_ext=10, doCD = False, factor=1, base_size=64,
 offsets = None):
    import galsim
    
    
    n_stars = len(offsets)

    ext_all = np.arange(1,n_ext+1)
    sci = []
    err = []
    mask = []
    time=[]
    sigma_noise = .10
    gal_flux_rate = 2.e3
    gal_fwhm = 0.5       # arcsec
    pixel_scale=0.13
    
    #time_vec = np.linspace (0., n_ext*delta_time, n_ext)
    #random_seed = 1234568
    rng = galsim.BaseDeviate()

    for ext in ext_all:
        
        profile=galsim.Gaussian(flux=gal_flux_rate, fwhm=gal_fwhm)
        if n_stars > 1:
            sci_im=profile.drawImage(image=galsim.Image(base_size, base_size, dtype=np.float64), scale=pixel_scale, offset=offsets[0])

            for i in xrange(n_stars-1):
                sci_im = sci_im + profile.drawImage(image=
                                                    galsim.Image(base_size, base_size, dtype=np.float64),
                                                    scale=pixel_scale,offset=offsets[i])
    
        else:
            sci_im=profile.drawImage(image=galsim.Image(base_size, base_size, dtype=np.float64), scale=pixel_scale)
        
    
        sci_im.addNoise(galsim.GaussianNoise(rng = rng, sigma=sigma_noise))
        if doCD is True:
            sci_im=apply_cdmodel(sci_im,factor=factor)

        sci.append( sci_im.array )
        err.append( sci_im.array*0.+ sigma_noise/np.sqrt(ext) )
        mask.append( sci_im.array * 0 )
    
        time.append( sci_im.array*0 + ext*delta_time )
    
    time=np.array(time)

    sci_arr = np.cumsum(np.array(sci), axis=0)/ time
    err_arr = np.array(err)
    mask_arr = np.array(mask)


    return sci_arr, err_arr, mask_arr

def get_cd_deriv(image_obj, delta_factor = 1.0):
    image_obj_p = apply_cdmodel(image_obj,factor = delta_factor)
    delta_image_obj = (image_obj_p - image_obj) / delta_factor
    return delta_image_obj

def get_predicted_final_image(sci_arr, err_arr, mask_arr):
    return sci_arr[1,:,:]

def get_predicted_final_cinv(err_arr, mask_arr):
    return (1./err_arr[1,:,:]**2)*0.5

def get_final_image(sci_arr, err_arr, mask_arr):
    return sci_arr[0,:,:]

def estimator(sci_arr, err_arr, mask_arr ):

    n_ext=sci_arr.shape[0]
    est_vec, est_err_vec =[],[]

    for i in range(n_ext-1):
        #I_pred = get_predicted_final_image(sci_arr, err_arr, mask_arr)
        #I_act = get_final_image(sci_arr, err_arr, mask_arr)
        I_pred = sci_arr[i+1,:,:]
        I_act  = sci_arr[i,:,:]
        
        image_obj_pred = galsim.Image(I_pred)
        cd_deriv = get_cd_deriv(image_obj_pred)
        # Cinv will just be a 2D array the same size as a the predicted data vector.
        # because we're assuming the pixel noise is uncorrelated. (Ha!)
        Cinv = get_predicted_final_cinv(err_arr, mask_arr)

        est_num = np.dot( (I_act - I_pred).flatten(), cd_deriv.array.flatten() * Cinv.flatten())
        est_denom = np.dot( cd_deriv.array.flatten() * Cinv.flatten(), cd_deriv.array.flatten())

        est = est_num/est_denom
        est_err = 1./np.sqrt(est_denom)

        est_vec.append(est)
        est_err_vec.append(est_err)

    est_vec, est_err_vec = np.array(est_vec), np.array(est_err_vec)

    return np.mean(est_vec), np.sqrt ( np.mean(est_err**2*1.0/(n_ext-1)) )


def main (argv):
    n=1000
    
    sum_est_cd, sum_est_var_cd = 0.0, 0.0
    sum_est_nocd, sum_est_var_nocd = 0.0, 0.0
    all_est_cd = np.zeros(n)
    all_est_nocd = np.zeros(n)
    base_size=64
    n_stars = 10

    offsets_vec=[ base_size/2. * np.random.rand(2) for i in xrange(n_stars) ]
    for i in xrange(n):


        sci_arr_cd, err_arr_cd, mask_arr_cd = get_simulated_array (delta_time=10, n_ext=10, doCD = True, factor=1.5, offsets=offsets_vec, base_size= base_size)
        sci_arr_nocd, err_arr_nocd, mask_arr_nocd = get_simulated_array (delta_time=10, n_ext=10, doCD = False, offsets=offsets_vec, base_size= base_size)

    
        est_cd, est_err_cd = estimator (sci_arr_cd, err_arr_cd, mask_arr_cd )
        est_nocd, est_err_nocd = estimator (sci_arr_nocd, err_arr_nocd, mask_arr_nocd )
        all_est_cd[i] = est_cd
        all_est_nocd[i] = est_nocd
        
        sum_est_cd+=est_cd
        sum_est_var_cd+=est_err_cd**2
    
        sum_est_nocd+=est_nocd
        sum_est_var_nocd+=est_err_nocd**2
        print "cumulative yes CD: ", i, sum_est_cd*1./i, np.sqrt(sum_est_var_cd)*1./i
        print "cumulative no CD: ", i,sum_est_nocd*1./i, np.sqrt(sum_est_var_nocd)*1./i
        
    
    print " "
    print "Final yes CD: ", sum_est_cd*1./n, np.sqrt (sum_est_var_cd*1./n**2)
    print "final no CD: ", sum_est_nocd*1./n, np.sqrt (sum_est_var_nocd*1./n**2)
    print "empirical stdev. (CD): ", np.std(all_est_cd), np.std(all_est_cd)/ np.sqrt(n)
    print "empirical stdev. (noCD): ",np.std(all_est_nocd) ,np.std(all_est_nocd) / np.sqrt(n)
        
    stop




if __name__ == "__main__":
    import pdb, traceback
    try:
        main(sys.argv)
    except:
        thingtype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)


