#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import esutil
from scipy import ndimage
import sys
import galsim
import matplotlib.cm as cm
import cdmodel_functions
import fitsio
import subprocess as S

file_path1="/Users/amalagon/WFIRST/WFC3_data/data/omega-cen-all_data/omega-cen-ima-files/ibcj09ksq_ima.fits"
file_path2="/Users/amalagon/WFIRST/WFC3_data/data/omega-cen-all_data/omega-cen-ima-files/ibcj09kkq_ima.fits"

file_path3="/Users/amalagon/WFIRST/WFC3_data/multiaccum_ima_files_omega_cen/ibcf81qkq_ima.fits"    # star
file_path4="/Users/amalagon/WFIRST/WFC3_data/data/standard-stars-hst/GD153/ibcf0cvmq_ima.fits"
file_path5="/Users/amalagon/WFIRST/WFC3_data/data/standard-stars-hst/GD71_G191B2B/ibcf90i1q_ima.fits"
#file_path = "../../Data/ibcj09ksq_ima.fits" # path to file on Huff's machine.
file_path = "../../Data/ibcf0cvmq_ima.fits" # path to file on Huff's machine.


all_files_path="/Users/amalagon/WFIRST/WFC3_data/data/standard-stars-hst/all_files/"
GD71_G191B2B_files_path="/Users/amalagon/WFIRST/WFC3_data/data/standard-stars-hst/GD71_G191B2B/"

def get_image_data(file=file_path, ext_per_image = 5, exclude_last = 3):
    hdr = esutil.io.read_header(file)
    n_ext  = hdr.get("NEXTEND")
    ext_all = np.arange(1,n_ext,ext_per_image)
    sci = []
    err = []
    mask = []
    for ext in ext_all:
        sci.append(esutil.io.read(file,ext=ext))
        err.append(esutil.io.read(file,ext=ext+1))
        mask.append(esutil.io.read(file,ext=ext+2))

    sci_use = sci[0:-exclude_last]
    err_use = err[0:-exclude_last]
    mask_use = mask[0:-exclude_last]
    sci_arr = np.array(sci_use)
    err_arr = np.array(err_use)
    mask_arr = np.array(mask_use)
    return sci_arr[::-1,:,:], err_arr[::-1,:,:], 1.0*mask_arr[::-1,:,:]

def apply_nonlinearity (im, factor=0.01, doCD = True):
    """
    Uses galsim to apply "cdmodel" to an input image. 
    "cdmodel" ('charge deflection') is an implementation of the 'brightter-fatter'
    model by Antilogus et al 2014 as performed by Gruen et al 2015.
    """
    if doCD is True:
        (aL,aR,aB,aT) = cdmodel_functions.readmeanmatrices()
        cd = galsim.cdmodel.BaseCDModel (factor*aL,factor*aR,factor*aB,factor*aT)
        im_out=cd.applyForward(im.copy())
    else:
        f=lambda x,beta : x + beta*x*x
        im_out=im.copy()
        im_out.applyNonlinearity(f,factor)
    
    return im_out

def get_true_flux(offsets = None, base_size = 64):
    sigma_noise = .10
    gal_flux_rate = 2.e3
    gal_fwhm = 0.5       # arcsec
    pixel_scale=0.13
    profile=galsim.Gaussian(flux=gal_flux_rate, fwhm=gal_fwhm)
    n_stars = len(offsets)
    if n_stars > 1:
        sci_im=profile.drawImage(image=galsim.Image(base_size, base_size, dtype=np.float64), scale=pixel_scale, offset=offsets[0])
        for i in xrange(1,n_stars):
            sci_im = sci_im + profile.drawImage(image= galsim.Image(base_size, base_size, dtype=np.float64),
                                                scale=pixel_scale,offset=offsets[i])
    else:
        sci_im =profile.drawImage(image=galsim.Image(base_size, base_size, dtype=np.float64), scale=pixel_scale)

    return sci_im

            

def get_simulated_array (delta_time=10, n_ext=10, doCD = False, factor=1., base_size=64, offsets = None, addNoise = True, true_flux = None):
    # True flux needs to be a GalSim image object
    import galsim
    ext_all = np.arange(1,n_ext+1)
    sci = []
    err = []
    mask = []
    time=[]
    sigma_noise = .10
    gal_flux_rate = 2.e5
    gal_fwhm = 0.5       # arcsec
    pixel_scale=0.13
    #time_vec = np.linspace (0., n_ext*delta_time, n_ext)
    #random_seed = 1234568
    rng = galsim.BaseDeviate()
    if true_flux == None:
        true_flux = get_true_flux( offsets = offsets, base_size =  base_size )
    
    for ext in ext_all:
        if len(sci) == 0:
            sci_im = true_flux * delta_time
        else:
            sci_im = true_flux * delta_time + sci[-1]
        if doCD == True:
            sci_im = apply_nonlinearity( sci_im , factor = factor )
        if addNoise == True:
            sci_im.addNoise(galsim.GaussianNoise(rng = rng, sigma=sigma_noise))
        if len(sci) == 0:
            sci.append( sci_im.array )
            err.append( sci_im.array * 0. + (sigma_noise *  delta_time) )
        else:
            sci.append( sci_im.array )
            err.append( np.sqrt( ((sigma_noise*delta_time)**2 + err[-1]**2)) )
        this_mask = np.zeros(sci_im.array.shape)    #* 0.
        #np.seterr(all='raise')
        #stop
        #this_mask[12,:] = 1
        mask.append( this_mask )

        #time.append( sci_im.array*0. + ext*delta_time )
        time.append(np.zeros(sci_im.array.shape) + ext*delta_time)

    time=np.array(time)

    sci_arr = np.array(sci) / time
    err_arr = np.array(err) / time
    mask_arr = np.array(mask)
    return sci_arr, err_arr, mask_arr



def estimator(sci_arr, err_arr, mask_arr, delta_time = 10., delta_factor = .10, pixel_scale = 0.13):
    
    n_ext=sci_arr.shape[0]
    est_vec, est_err_vec =[],[]
    true_flux = galsim.Image(sci_arr[0,:,:],scale=pixel_scale)
    sci_pred_0,_,_ = get_simulated_array(delta_time = delta_time, n_ext = n_ext, doCD = False, base_size = sci_arr.shape[1],
                                     addNoise = False, true_flux = true_flux)
    sci_pred_p,_,_ = get_simulated_array(delta_time = delta_time, n_ext = n_ext, doCD = True, factor = delta_factor,
                                     base_size = sci_arr.shape[1], addNoise = False, true_flux = true_flux)
    cd_deriv = (sci_pred_p - sci_pred_0) * (1. / delta_factor)
    Cinv = 1./err_arr**2
    diff = sci_arr - sci_pred_0
    use = (err_arr > 0.) & (mask_arr == 0)

    est_num = np.dot ( cd_deriv[use], diff[use] * Cinv[use])    # become 1D vectors
    est_denom = np.dot( cd_deriv[use] * Cinv[use], cd_deriv[use] )
    est, est_err = est_num / est_denom, 1./np.sqrt(est_denom)
    #stop
    return est, est_err


def main (argv):
    simulate = False

    #file_path = "../../Data/ibcj09ksq_ima.fits" # path to file on Huff's machine.
    #file_path = "../../Data/ibcf0cvmq_ima.fits" # path to file on Huff's machine.
    #file_path = ["../../Data/ibcf0cvmq_ima.fits", "../../Data/ibcj09ksq_ima.fits"] # path to file on Huff's machine.
    #file_path=[file_path2] #, file_path4, file_path5]
    cmd="ls %s*ima*.fits" %GD71_G191B2B_files_path
    file_path=S.Popen([cmd], shell=True, stdout=S.PIPE).communicate()[0].split()


    n=len(file_path)
    sum_est_cd, sum_est_var_cd = 0.0, 0.0
    sum_est_nocd, sum_est_var_nocd = 0.0, 0.0
    all_est_cd = np.zeros(n)
    all_est_nocd = np.zeros(n)
    base_size=64
    n_stars = 3
    factor = 2.0
    
    # Real data
    exclude_last=2
    

    offsets_vec= [ base_size *( np.random.rand(2) - 1) for i in xrange(n_stars) ]
    for i in xrange(n):
        this_file_path = file_path[i]
        if simulate is True:
            true_flux = get_true_flux(offsets=offsets_vec, base_size = base_size)
            sci_arr_cd, err_arr, mask_arr = get_simulated_array (delta_time=10, n_ext=10, doCD = True, factor=factor, true_flux = true_flux)
            sci_arr_nocd, _, _ = get_simulated_array (delta_time=10, n_ext=10, doCD = False, true_flux = true_flux)
        else:
            hdr = esutil.io.read_header(this_file_path)
            n_samp  = hdr.get("NSAMP")
            ### Calculate delta time for real data
            time1 = fitsio.read_header(this_file_path, 1).get ("SAMPTIME")
            time2 = fitsio.read_header(this_file_path, 6).get ("SAMPTIME")
            delta_time = np.abs(time2 - time1)
            print "delta_time: ", delta_time
            
            
            sci_arr, err_arr, mask_arr = get_image_data(file=this_file_path, ext_per_image = 5, exclude_last = exclude_last)
            true_flux = galsim.Image(sci_arr[0,:,:],scale = 0.13)
            #stop

            sci_arr_cd, _, _ = get_simulated_array (delta_time=delta_time, n_ext=n_samp-exclude_last, doCD = True, factor=factor, true_flux = true_flux)
            sci_arr_nocd, _, _ = get_simulated_array (delta_time=delta_time, n_ext=n_samp-exclude_last, doCD = False, true_flux = true_flux)

        est_cd, est_err_cd = estimator (sci_arr_cd, err_arr, mask_arr)
        est_nocd, est_err_nocd = estimator (sci_arr_nocd, err_arr, mask_arr)
        #print est_cd, est_nocd
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


