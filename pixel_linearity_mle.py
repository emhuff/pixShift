#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import esutil
from scipy import ndimage
import sys
import galsim
import matplotlib.cm as cm
import cdmodel_functions


def apply_nonlinearity (im, factor=0.01, doCD = True):
    """
    Uses galsim to apply "cdmodel" to an input image. 
    "cdmodel" ('charge deflection') is an implementation of the 'brightter-fatter'
    model by Antilogus et al 2014 as performed by Gruen et al 2015.
    """
    if doCD is True:
        (aL,aR,aB,aT) = cdmodel_functions.readmeanmatrices()
        cd = galsim.cdmodel.BaseCDModel (factor*aL,factor*aR,factor*aB,factor*aT)
        im_out=cd.applyForward(im)
    
    else:
        f=lambda x,beta : x + beta*x*x
        im_out=im.copy()
        im_out.applyNonlinearity(f,factor)
    
    return im_out


def get_simulated_array (delta_time=10, n_ext=10, doCD = False, factor=1, base_size=64, offsets = None):
    import galsim
    n_stars = len(offsets)
    ext_all = np.arange(1,n_ext+1)
    sci = []
    err = []
    mask = []
    time=[]
    sigma_noise = .10
    gal_flux_rate = 2.e4
    gal_fwhm = 0.5       # arcsec
    pixel_scale=0.13
    #time_vec = np.linspace (0., n_ext*delta_time, n_ext)
    #random_seed = 1234568
    rng = galsim.BaseDeviate()
    for ext in ext_all:
        profile=galsim.Gaussian(flux=gal_flux_rate, fwhm=gal_fwhm)
        if n_stars > 1:
            sci_im=profile.drawImage(image=galsim.Image(base_size, base_size, dtype=np.float64), scale=pixel_scale, offset=offsets[0])
            for i in xrange(1,n_stars):
                sci_im = sci_im + profile.drawImage(image=
                                                    galsim.Image(base_size, base_size, dtype=np.float64),
                                                    scale=pixel_scale,offset=offsets[i])
        else:
            sci_im=profile.drawImage(image=galsim.Image(base_size, base_size, dtype=np.float64), scale=pixel_scale)
        
        sci_im.addNoise(galsim.GaussianNoise(rng = rng, sigma=sigma_noise))
        if doCD == True:
            sci_im=apply_nonlinearity(sci_im,factor=factor)

        sci.append( sci_im.array * delta_time )
        err.append( sci_im.array * 0. + sigma_noise *  delta_time)
        mask.append( sci_im.array * 0 )

        time.append( sci_im.array*0 + ext*delta_time )
    
    time=np.array(time)

    sci_arr = np.cumsum(np.array(sci),axis=0) / time
    
    err_arr = np.sqrt(np.cumsum(np.array(err)**2,axis=0)) / time
    mask_arr = np.array(mask)
    return sci_arr, err_arr, mask_arr

def get_predicted_sci_array(sci_arr,delta_time=10, n_ext=10, doCD = False, factor=1, pixel_scale = 0.13):
    this_sci_im =  sci_arr[0,:,:] * delta_time
    sci_im = [this_sci_im]
    err = []
    mask = []
    time = [this_sci_im * 0. + delta_time]
    for i in (range(1,n_ext)):
        this_sci_arr = sci_arr[0,:,:]*delta_time + sci_im[i-1]
        this_sci_im = galsim.Image( this_sci_arr, scale=  pixel_scale)
        if doCD == True:
            this_sci_im = apply_nonlinearity(this_sci_im, factor = factor)
        sci_im.append( this_sci_im.array )
        time.append( this_sci_arr * 0. + (i+1) * delta_time)

    time = np.array(time)
    sci_arr_pred = np.array(sci_im)/time
    return sci_arr_pred
    

def get_cd_deriv(image_obj, delta_factor = 0.01):
    image_obj_p = apply_nonlinearity(image_obj,factor = delta_factor)
    delta_image_obj = (image_obj_p - image_obj) / delta_factor
    #stop
    return delta_image_obj

def get_predicted_final_image(sci_arr, err_arr, mask_arr):
    return sci_arr[1,:,:]

def get_predicted_final_cinv(err_arr, mask_arr):
    return (1./err_arr[1,:,:]**2)*0.5

def get_final_image(sci_arr, err_arr, mask_arr):
    return sci_arr[0,:,:]

def estimator(sci_arr, err_arr, mask_arr, delta_time = 10., delta_factor = .10, pixel_scale = 0.13):

    n_ext=sci_arr.shape[0]
    est_vec, est_err_vec =[],[]
    sci_pred_p = get_predicted_sci_array(sci_arr, delta_time = delta_time,
                                         n_ext = n_ext, factor = delta_factor,
                                         pixel_scale = pixel_scale, doCD = True)
    sci_pred_0 = get_predicted_sci_array(sci_arr, delta_time = delta_time,
                                         n_ext = n_ext, doCD = False, factor = 0.,
                                         pixel_scale = pixel_scale)
    cd_deriv = (sci_pred_p - sci_pred_0) * (1. / delta_factor)
    Cinv = 1./err_arr**2
    diff = sci_arr - sci_pred_0
    est_num = np.dot ( cd_deriv.flatten(), diff.flatten() * Cinv.flatten())
    est_denom = np.dot( cd_deriv.flatten() * Cinv.flatten(), cd_deriv.flatten() )
    
    est, est_err = est_num / est_denom, 1./np.sqrt(est_denom)
    stop
    return est, est_err


def main (argv):
    n=100
    sum_est_cd, sum_est_var_cd = 0.0, 0.0
    sum_est_nocd, sum_est_var_nocd = 0.0, 0.0
    all_est_cd = np.zeros(n)
    all_est_nocd = np.zeros(n)
    base_size=32
    n_stars = 2
    factor = 1.

    offsets_vec=[ base_size/2. * np.random.rand(2) for i in xrange(n_stars) ]
    for i in xrange(n):
        sci_arr_cd, err_arr_cd, mask_arr_cd = get_simulated_array (delta_time=10, n_ext=10, doCD = True, factor=factor, offsets=offsets_vec, base_size= base_size)
        sci_arr_nocd, err_arr_nocd, mask_arr_nocd = get_simulated_array (delta_time=10, n_ext=10, doCD = False, offsets=offsets_vec, base_size= base_size)

        est_cd, est_err_cd = estimator (sci_arr_cd, err_arr_cd, mask_arr_cd)
        est_nocd, est_err_nocd = estimator (sci_arr_nocd, err_arr_nocd, mask_arr_nocd)
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


