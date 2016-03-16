#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import esutil
from scipy import ndimage
import sys
import galsim

import cdmodel_functions


#file_path="/Users/amalagon/WFIRST/WFC3_data/data/omega-cen-all_data/omega-cen-ima-files/ibcj09ksq_ima.fits"
#file_path="/Users/amalagon/WFIRST/WFC3_data/data/omega-cen-all_data/omega-cen-ima-files/ibcj09kkq_ima.fits"
#file_path="/Users/amalagon/WFIRST/WFC3_data/multiaccum_ima_files_omega_cen/ibcf81qkq_ima.fits"
#file_path="/Users/amalagon/WFIRST/WFC3_data/data/standard-stars-hst/GD153/ibcf0cvmq_ima.fits"
file_path="/Users/amalagon/WFIRST/WFC3_data/data/standard-stars-hst/GD71_G191B2B/ibcf90i1q_ima.fits"

def apply_cdmodel (im, factor=1):
    """
    Uses galsim to apply "cdmodel" to an input image. 
    "cdmodel" ('charge deflection') is an implementation of the 'brightter-fatter'
    model by Antilogus et al 2014 as performed by Gruen et al 2015.
    """
    (aL,aR,aB,aT) = cdmodel_functions.readmeanmatrices()
    cd = galsim.cdmodel.BaseCDModel (factor*aL,factor*aR,factor*aB,factor*aT)
    im=cd.applyForward(im)
    return im



def get_image_array(file=file_path, ext_per_image = 5,
                    exclude_last = 3):
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
    return sci_arr, err_arr, 1.0*mask_arr


def get_simulated_array (delta_time=10, n_ext=10):
    import galsim

    ext_all = np.arange(1,n_ext+1)
    sci = []
    err = []
    mask = []
    
    gal_flux_rate = 1.e6
    gal_fwhm = 0.4       # arcsec
    pixel_scale=0.11
    
    #time_vec = np.linspace (0., n_ext*delta_time, n_ext)
    random_seed = 1234567
    rng = galsim.BaseDeviate(random_seed)
    base_size=32
    
    
    sci_im=np.zeros((base_size, base_size))
    err_im_temp=np.zeros(sci_im.shape)

    for ext in ext_all:
        
        profile=galsim.Gaussian(flux=gal_flux_rate, fwhm=gal_fwhm)
        sci_im=profile.drawImage(image=galsim.Image(base_size, base_size, dtype=np.float64), scale=pixel_scale)
        #noise = galsim.PoissonNoise(rng)
        #sci_im.addNoise(noise)
        #sci_im+=sci_im # To correlate the noise
        sci_im=sci_im.array
        err_im=np.sqrt(sci_im)
        mask_im=np.zeros(err_im.shape)
        
        sci_im+=sci_im

        err_im = np.sqrt ( err_im_temp**2 +  err_im**2 )
        err_temp = err_im
        
        sci.append(sci_im)
        err.append(err_im)
        mask.append(mask_im)

    sci_arr = np.array(sci)
    err_arr = np.array(err)
    mask_arr = np.array(mask)
    
    return sci_arr, err_arr, mask_arr






def make_chi2_map(sci_arr, err_arr, mask_arr):
    mean_image = np.mean(sci_arr,axis=0)
    deviant_arr = sci_arr - np.expand_dims(mean_image,axis=0)
    chisq = np.sum((deviant_arr/err_arr)**1,axis=0)
    chisq_dof = chisq*1./sci_arr.shape[0]
    mask_summed = (np.sum(mask_arr,axis=0) > 0) | ~np.isfinite(chisq)  # bad guys
    chisq_dof[mask_summed] = 0.
    return chisq_dof, mask_summed


#def make_quadratict_map (sci_arr, err_arr, mask_arr):
#    mean_image=np.mean(sci_arr, axis=0)
#    deviant_arr = sci_arr - np.expand_dims(mean_image,axis=0)


def main(argv):
    sci_arr, err_arr, mask_arr = get_image_array()
    #sci_arr, err_arr, mask_arr = get_simulated_array(delta_time=10, n_ext=10)

    theMap, ubermask = make_chi2_map(sci_arr, err_arr, mask_arr)
    fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize=(14,7))
    im1 = ax1.imshow(theMap,vmin=0,vmax=2)
    image = np.mean(sci_arr,axis=0)
    image_sharp = image - ndimage.gaussian_filter(image,5.)
    use = ~ubermask  #Negating the bad guys
    ## IPC:
    #u = 0.0011
    #v = 0.0127
    #w = 0.936
    #x = 0.0164

    ## Laplacain kernel
    a=0.02

    image_filtered = image * 0.
    theFilter = np.array([[0.,a, 0.], [a, -4*a, a], [0., a, 0.]])  #Laplacian
    for i in xrange(3):
        for j in xrange(3):
            image_filtered = image_filtered + theFilter[i,j] * image

    image_filtered-=image

    ax1.set_title("chisq dof map")
    im2 = ax2.imshow(np.arcsinh(np.mean(sci_arr,axis=0)/0.1),cmap=plt.cm.Greys)
    ax2.set_title("mean science image")
    fig.savefig("chisq_nonlin_map.png")
    fig.show()

    from matplotlib.colors import LogNorm

    root= file_path.split('/')[-1].split('.')[0]
    header=esutil.io.read_header (file_path)
    name, time = header.get('TARGNAME').strip(), int(header.get ('EXPTIME'))

    #root='simulated_gaussian'
    #name='andres'
    #time=100


    fig2, ax4 = plt.subplots(nrows=1,ncols=1,figsize=(7,7))
    #ax3.plot(image.flatten(),theMap.flatten(),',')
    im_filtered_min, im_filtered_max = np.percentile (np.abs(image_filtered[use].flatten()), [5,95]  )
    ax4.hist2d(np.abs(image_filtered[use].flatten()),np.abs(theMap[use].flatten()),norm=LogNorm(),
               bins = [np.linspace(im_filtered_min - 0.5*np.abs(im_filtered_min), im_filtered_max +  0.5*np.abs(im_filtered_max),100),np.linspace(0,1.0,100)])
               #norm=LogNorm())
    ax4.axhline (0., linestyle="--")
    ax4.set_xlabel("(Laplacian filtered image value)**2")
    ax4.set_ylabel("chi2")
    fig2.savefig("flux_chi2_corr_%s_%s_%g.png" %(root, name, time)  )
    fig2.show()
    stop

 
if __name__ == "__main__":
    import pdb, traceback
    try:
        main(sys.argv)
    except:
        thingtype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
    
