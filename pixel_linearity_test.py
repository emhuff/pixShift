#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import esutil
from scipy import ndimage
import sys

import cdmodel_functions


file_path="/Users/amalagon/WFIRST/WFC3_data/data/omega-cen-all_data/ibcj09ksq_ima.fits"

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
    return sci_arr, err_arr, mask_arr


def make_chi2_map(sci_arr, err_arr, mask_arr):
    mean_image = np.mean(sci_arr,axis=0)
    deviant_arr = sci_arr - np.expand_dims(mean_image,axis=0)
    chisq = np.sum((deviant_arr/err_arr)**2,axis=0)
    chisq_dof = chisq*1./sci_arr.shape[0]
    mask_summed = np.sum(mask_arr,axis=0) | ~np.isfinite(chisq)
    chisq_dof[mask_summed != 0] = 0.
    return chisq_dof, mask_summed


#def make_quadratict_map (sci_arr, err_arr, mask_arr):
#    mean_image=np.mean(sci_arr, axis=0)
#    deviant_arr = sci_arr - np.expand_dims(mean_image,axis=0)


def main(argv):
    sci_arr, err_arr, mask_arr = get_image_array()
    theMap, ubermask = make_chi2_map(sci_arr, err_arr, mask_arr)
    fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize=(14,7))
    im1 = ax1.imshow(theMap,vmin=0,vmax=2)
    image = np.mean(sci_arr,axis=0)
    image_sharp = image - ndimage.gaussian_filter(image,5.)
    use = ubermask == 0
    u = 0.0011
    v = 0.0127
    w = 0.936
    x = 0.0164
    image_filtered = image * 0.
    theFilter = np.array([[u,v, u], [x, w, x], [u, v, u]])
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
    fig2,(ax3,ax4) = plt.subplots(nrows=1,ncols=2,figsize=(14,7))
    #ax3.plot(image.flatten(),theMap.flatten(),',')
    ax3.hist2d(np.log10(image_sharp[use].flatten()),np.log10(theMap[use].flatten()),
               bins = [np.linspace(-1,2,100),np.linspace(-2,1,100)],
               norm=LogNorm())
    ax3.set_xlabel('(image - smoothed image) value')
    ax3.set_ylabel("chi2")
    ax4.hist2d(np.log10(image_filtered[use].flatten()),np.log10(theMap[use].flatten()),norm=LogNorm())
               #bins = [np.linspace(0.5,2.5,100),np.linspace(-2,1,100)],
               #norm=LogNorm())
    ax4.set_xlabel("ipc filtered image value")
    ax4.set_ylabel("chi2")
    #ax3.set_xscale('log')
    #ax3.set_yscale('log')
    fig2.savefig("flux_chisq_correlation.png")
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
    
