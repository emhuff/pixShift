#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import esutil
from scipy import ndimage
import sys
import galsim
import matplotlib.cm as cm
import cdmodel_functions


#file_path="/Users/amalagon/WFIRST/WFC3_data/data/omega-cen-all_data/omega-cen-ima-files/ibcj09ksq_ima.fits"
#file_path="/Users/amalagon/WFIRST/WFC3_data/data/omega-cen-all_data/omega-cen-ima-files/ibcj09kkq_ima.fits"
#file_path="/Users/amalagon/WFIRST/WFC3_data/multiaccum_ima_files_omega_cen/ibcf81qkq_ima.fits"
#file_path="/Users/amalagon/WFIRST/WFC3_data/data/standard-stars-hst/GD153/ibcf0cvmq_ima.fits"
file_path="/Users/amalagon/WFIRST/WFC3_data/data/standard-stars-hst/GD71_G191B2B/ibcf90i1q_ima.fits"
#file_path = "../../Data/ibcj09ksq_ima.fits" # path to file on Huff's machine.
#file_path = "../../Data/ibcf0cvmq_ima.fits" # path to file on Huff's machine.
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


def get_simulated_array (delta_time=10, n_ext=5, doCD = False, factor=1):
    import galsim

    ext_all = np.arange(1,n_ext+1)
    sci = []
    err = []
    mask = []
    time=[]
    sigma_noise = .02
    gal_flux_rate = 2.e3
    gal_fwhm = 0.5       # arcsec
    pixel_scale=0.13
    
    #time_vec = np.linspace (0., n_ext*delta_time, n_ext)
    random_seed = 1234567
    rng = galsim.BaseDeviate(random_seed)
    base_size=64

    for ext in ext_all:
        
        profile=galsim.Gaussian(flux=gal_flux_rate, fwhm=gal_fwhm)
        sci_im=profile.drawImage(image=galsim.Image(base_size, base_size, dtype=np.float64), scale=pixel_scale)
        sci_im.addNoise(galsim.GaussianNoise(rng = rng, sigma=sigma_noise))
        if doCD is True:
            sci_im=apply_cdmodel(sci_im,factor=factor)

        sci.append( sci_im.array )
        err.append( sci_im.array*0.+ sigma_noise )
        mask.append( sci_im.array * 0 )
    
        time.append( sci_im.array*0 + ext*delta_time )
    
    time=np.array(time)

    sci_arr = np.cumsum(np.array(sci), axis=0)/ time
    err_arr = np.array(err)
    mask_arr = np.array(mask)


    return sci_arr, err_arr, mask_arr


def make_chi2_map(sci_arr, err_arr, mask_arr):
    mean_image = np.average(sci_arr,weights=1.0/err_arr**2, axis=0)
    deviant_arr = sci_arr - np.expand_dims(mean_image,axis=0)
    chisq = np.sum((deviant_arr/err_arr)**1,axis=0)
    chisq_dof = chisq*1./sci_arr.shape[0]
    mask_summed = (np.sum(mask_arr,axis=0) > 0) | ~np.isfinite(chisq)  # bad guys
    chisq_dof[mask_summed] = 0.
    return chisq_dof, mask_summed


#def make_quadratict_map (sci_arr, err_arr, mask_arr):
#    mean_image=np.mean(sci_arr, axis=0)
#    deviant_arr = sci_arr - np.expand_dims(mean_image,axis=0)

def plot_average_pixel_trend(sci_arr, err_arr, mask_arr, scale = 0.13,
                             doPlot = True, doCD = False,factor=1):
    chi2Map, ubermask = make_chi2_map(sci_arr, err_arr, mask_arr)
    use = ~ubermask  #Negating the bad guys
    image = np.average(sci_arr,axis=0, weights = 1./err_arr**2)
    image[ubermask] = -np.inf
    '''
    image_filtered = image * 0.
    a=0.1
    theFilter = np.array([[0.,a, 0.], [a, -4*a, a], [0., a, 0.]])  #Laplacian
    for i in xrange(3):
        for j in xrange(3):
            image_filtered = image_filtered + theFilter[i,j] * image
    image_filtered-=image
    '''
    image_filtered = apply_cdmodel (galsim.Image(image,scale=scale), factor= factor).array - image
    
    image_filtered[ubermask | ~np.isfinite(image_filtered)] = 0.

    # Bin the filtered image values into quantiles.
    nq = 10
    quant = np.percentile(image_filtered[use].flatten(), np.linspace(0,100,nq))
    deviant_arr = sci_arr - np.expand_dims(image,axis=0)
    max_interval = 0.
    timeseries = []
    
    for i in xrange(nq-1):
        these_pixels = ( (image_filtered > quant[i]) &
                         (image_filtered <= quant[i+1]) &
                         (image_filtered != 0) )
        these_pixels_3d = np.repeat(np.expand_dims(these_pixels,axis=0),sci_arr.shape[0],axis=0)
        this_dev_array = deviant_arr.copy()
        this_dev_array[~these_pixels_3d] = 0.
        this_npix = np.sum(these_pixels)
        this_timeseries = np.sum(np.sum(this_dev_array,axis=1),axis=1) * 1./this_npix
        timeseries.append(this_timeseries)
        this_interval = np.abs(np.max(this_timeseries) - np.min(this_timeseries))
        if this_interval > max_interval:
            max_interval = this_interval
    offset_array = (np.arange(nq) - np.mean(np.arange(nq))) * max_interval
    if doPlot is True:
        fig,(ax1,ax2,ax3) = plt.subplots(nrows=1,ncols=3,figsize=(28,6))
        colors = cm.seismic(np.linspace(0, 1, nq-1))
        ax1.imshow(np.arcsinh(image))
        ax2.imshow(image_filtered,cmap=cm.seismic,vmin = quant[1],vmax=-quant[1])
        ax2.set_title("Laplacian-filtered image")
        for i in xrange(nq-1):
            ax3.plot((timeseries[i] + offset_array[i])[::-1],color=colors[i],marker='.')
        fig.savefig("linearity_timeseries_trend.png")
        ax3.set_xlabel ("Time (arbitrary units)")
        ax3.set_ylabel ("Corrected pixel flux (e/sec)")
    
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.3)
        fig.show()
    timeseries_offset = [ts + os for ts,os in zip(timeseries,offset_array)]
    return timeseries_offset
            
def main(argv):
    
    #sci_arr, err_arr, mask_arr = get_image_array()
    sci_arr_nocd, err_arr_nocd, mask_arr_nocd = get_simulated_array(delta_time=5, n_ext=10, doCD = False)
    sci_arr_cd, err_arr_cd, mask_arr_cd = get_simulated_array(delta_time=5, n_ext=10, doCD = True, factor=1)
    timeseries_nocd = plot_average_pixel_trend(sci_arr_nocd, err_arr_nocd, mask_arr_nocd,doPlot= True)
    timeseries_cd = plot_average_pixel_trend(sci_arr_cd, err_arr_cd, mask_arr_cd,doPlot= False)

    fig,ax = plt.subplots()
    for tnocd,tcd in zip(timeseries_nocd,timeseries_cd):
        ax.plot((tcd - tnocd)[::-1])
    fig.show()
    stop

    
    '''
    chi2Map, ubermask = make_chi2_map(sci_arr, err_arr, mask_arr)
    fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize=(14,7))
    im1 = ax1.imshow(chi2Map,vmin=0,vmax=2)
    image = np.mean(sci_arr,axis=0)
    image_sharp = image - ndimage.gaussian_filter(image,5.)
    use = ~ubermask  #Negating the bad guys
    ## IPC:
    #u = 0.0011
    #v = 0.0127
    #w = 0.936
    #x = 0.0164

    ## Laplacian kernel
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

    root='simulated_gaussian'
    name='andres'
    time=100


    fig2, ax4 = plt.subplots(nrows=1,ncols=1,figsize=(7,7))
    #ax3.plot(image.flatten(),chi2Map.flatten(),',')
    im_filtered_min, im_filtered_max = np.percentile ((image_filtered[use].flatten()), [5,95]  )
    ax4.hist2d((image_filtered[use].flatten()),(chi2Map[use].flatten()),norm=LogNorm(),
               bins = [np.linspace(im_filtered_min - 0.5*np.abs(im_filtered_min), im_filtered_max +  0.5*np.abs(im_filtered_max),100),np.linspace(-2.0,2.0,100)])
               #norm=LogNorm())
    ax4.axhline (0., linestyle="--")
    ax4.set_xlabel("(Laplacian filtered image value)**2")
    ax4.set_ylabel("chi2")
    fig2.savefig("flux_chi2_corr_%s_%s_%g.png" %(root, name, time)  )
    fig2.show()
    stop
    '''
 
if __name__ == "__main__":
    import pdb, traceback
    try:
        main(sys.argv)
    except:
        thingtype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
    
