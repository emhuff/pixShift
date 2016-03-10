#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import esutil
import sys

def get_image_array(file ="../Data/ibcj09ksq_ima.fits", ext_per_image = 5,
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


def make_nonlin_map(sci_arr, err_arr, mask_arr):
    mean_image = np.mean(sci_arr,axis=0)
    deviant_arr = sci_arr - np.expand_dims(mean_image,axis=0)
    chisq = np.sum((deviant_arr/err_arr)**2,axis=0)
    chisq_dof = chisq*1./sci_arr.shape[0]
    mask_summed = np.sum(mask_arr,axis=0) | ~np.isfinite(chisq)
    chisq_dof[mask_summed != 0] = 0.
    return chisq_dof

def main(argv):
    sci_arr, err_arr, mask_arr = get_image_array()
    theMap = make_nonlin_map(sci_arr, err_arr, mask_arr)
    fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize=(14,7))
    im1 = ax1.imshow(theMap,vmin=0,vmax=2)
    ax1.set_title("chisq dof map")
    im2 = ax2.imshow(np.arcsinh(np.mean(sci_arr,axis=0)/0.1),cmap=plt.cm.Greys)
    ax2.set_title("mean science image")
    fig.savefig("chisq_nonlin_map.png")
    fig.show()
    stop

 
if __name__ == "__main__":
    import pdb, traceback
    try:
        main(sys.argv)
    except:
        thingtype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
    
