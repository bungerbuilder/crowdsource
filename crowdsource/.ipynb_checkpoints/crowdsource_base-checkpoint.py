"""Crowded field photometry pipeline.

This module fits positions, fluxes, PSFs, and sky backgrounds of images.
Intended usage is:
>>> x, y, flux, model, psf = fit_im(im, psf_initial, weight=wim,
                                    psfderiv=numpy.gradient(-psf),
                                    nskyx=3, nskyy=3, refit_psf=True)
which returns the best fit positions (x, y), fluxes (flux), model image
(model), and improved psf (psf) to the image im, with an initial psf guess
(psf_initial), an inverse-variance image wim, and a variable sky background.

See mosaic.py for how to use this on a large image that is too big to be fit
entirely simultaneously.
"""

import time
import sys
import os
import numpy
import pdb
import crowdsource.psf as psfmod
import scipy.ndimage.filters as filters
from collections import OrderedDict
from scipy import sparse
from typing import List, Tuple, Optional

nodeblend_maskbit = 2**30
sharp_maskbit = 2**31


def shift(im, offset, **kw):
    """Wrapper for scipy.ndimage.interpolation.shift"""
    from scipy.ndimage.interpolation import shift
    if 'order' not in kw:
        kw['order'] = 4
        # 1" Gaussian: 60 umag; 0.75": 0.4 mmag; 0.5": 4 mmag
        # order=3 roughly 5x worse.
    if 'mode' not in kw:
        kw['mode'] = 'nearest'
    if 'output' not in kw:
        kw['output'] = im.dtype
    return shift(im, offset, **kw)


def sim_image(nx, ny, nstar, psf, noise, nskyx=3, nskyy=3, stampsz=19):
    im = numpy.random.randn(nx, ny).astype('f4')*noise
    stampszo2 = stampsz // 2
    im = numpy.pad(im, [stampszo2, stampszo2], constant_values=-1e6,
                   mode='constant')
    x = numpy.random.rand(nstar).astype('f4')*(nx-1)
    y = numpy.random.rand(nstar).astype('f4')*(ny-1)
    flux = 1./numpy.random.power(1.0, nstar)
    for i in range(nstar):
        stamp = psf(x[i], y[i], stampsz=stampsz)
        xl = numpy.round(x[i]).astype('i4')
        yl = numpy.round(y[i]).astype('i4')
        im[xl:xl+stampsz, yl:yl+stampsz] += stamp*flux[i]
    if (nskyx != 0) or (nskyy != 0):
        im += sky_model(100*numpy.random.rand(nskyx, nskyy).astype('f4'),
                        im.shape[0], im.shape[1])
    ret = im[stampszo2:-stampszo2, stampszo2:-stampszo2], x, y, flux
    return ret


def significance_image(im, model, isig, psf, sz=19):
    """Significance of a PSF at each point, without local background fit."""
    # assume, for the moment, the image has already been sky-subtracted
    def convolve(im, kernel):
        from scipy.signal import fftconvolve
        return fftconvolve(im, kernel[::-1, ::-1], mode='same')
        # identical to 1e-8 or so
        # from scipy.ndimage.filters import convolve
        # return convolve(im, kernel[::-1, ::-1], mode='nearest')
    psfstamp = psfmod.central_stamp(psf, sz).copy()
    sigim = convolve(im*isig**2., psfstamp)
    varim = convolve(isig**2., psfstamp**2.)
    modim = convolve(model*isig**2., psfstamp)
    varim[varim <= 1e-14] = 0.  # numerical noise starts to set in around here.
    ivarim = 1./(varim + (varim == 0) * 1e14)
    return sigim*numpy.sqrt(ivarim), modim*numpy.sqrt(ivarim)


def significance_image_multiband(ims, models, isigs, psfs, band_weights=None, sz=19):
    """
    Joint matched-filter significance image for a list of bands, without local background fit

    Input Parameters
    ----------
    ims  : list(ndarray[Nx,Ny])
        Sky-subtracted data images, one per band.
    models  : list(ndarray[Nx,Ny])
        Current model images (same shapes as `images`).
    isigs   : list(ndarray[Nx,Ny])
        Inverse-sigma maps for each band.
    psfs    : list of PSF functions – one per band.
    weights : list(float), optional
        Relative weight for each band in the matched filter. If None, use equal weights (=1/nband).
    sz : int
        Size of the PSF stamp to use in convolution.

    Returns
    -------
    sigim   : 2darray[Nx,Ny]
        Joint significance image (data × PSF / noise).
    modim : 2darray[Nx,Ny]
        Combined model significance image (model × PSF / noise).
    """

    nband = len(ims)
    if band_weights is None: band_weights = [1.0/nband] * nband

    def convolve(im, kernel):
        from scipy.signal import fftconvolve
        return fftconvolve(im, kernel[::-1, ::-1], mode='same')

    sigim, modim, varim = 0.0, 0.0, 0.0
    for w, im, mod, isig, psf in zip(band_weights, ims, models, isigs, psfs):
        psfstamp = psfmod.central_stamp(psf, sz).copy()
        sigim += w * convolve(im   * isig**2, psfstamp)
        modim += w * convolve(mod  * isig**2, psfstamp)
        varim += w**2 * convolve(isig**2, psfstamp**2)

    varim[varim <= 1e-14] = 0.0
    ivarim = 1. / (varim + (varim == 0) * 1e14)

    return sigim * numpy.sqrt(ivarim), modim * numpy.sqrt(ivarim)



def significance_image_lbs(im, model, isig, psf, sz=19):
    """Give significance of PSF at each point, with local background fits."""

    def convolve(im, kernel):
        from scipy.signal import fftconvolve
        return fftconvolve(im, kernel[::-1, ::-1], mode='same')

    def convolve_flat(im, sz):
        from scipy.ndimage.filters import convolve
        filt = numpy.ones(sz, dtype='f4')
        c1 = convolve(im, filt.reshape(1, -1), mode='constant', origin=0)
        return convolve(c1, filt.reshape(-1, 1), mode='constant', origin=0)

    # we need: * convolution of ivar with P^2
    #          * convolution of ivar with flat
    #          * convolution of ivar with P
    #          * convolution of b*ivar with P
    #          * convolution of b*ivar with flat
    ivar = isig**2.
    if sz is None:
        psfstamp = psfmod.central_stamp(psf).copy()
    else:
        psfstamp = psfmod.central_stamp(psf, censize=sz).copy()
    ivarp2 = convolve(ivar, psfstamp**2.)
    ivarp2[ivarp2 < 0] = 0.
    ivarimsimple = 1./(ivarp2 + (ivarp2 == 0) * 1e12)
    ivarf = convolve_flat(ivar, psfstamp.shape[0])
    ivarp = convolve(ivar, psfstamp)
    bivarp = convolve(im*ivar, psfstamp)
    bivarf = convolve_flat(im*ivar, psfstamp.shape[0])
    atcinvadet = ivarp2*ivarf-ivarp**2.
    atcinvadet[atcinvadet <= 0] = 1.e-12
    ivarf[ivarf <= 0] = 1.e-12
    fluxest = (bivarp*ivarf-ivarp*bivarf)/atcinvadet
    fluxisig = numpy.sqrt(atcinvadet/ivarf)
    fluxsig = fluxest*fluxisig
    modim = convolve(model*ivar, psfstamp)
    return fluxsig, modim*numpy.sqrt(ivarimsimple)


def peakfind(im, model, isig, dq, psf, keepsat=False, threshold=5,
             blendthreshold=0.3, psfvalsharpcutfac=0.7, psfsharpsat=0.7,
             maxfiltershape=3, psfsz=59):
    psfstamp = psf(int(im.shape[0]/2.), int(im.shape[1]/2.), deriv=False,
                   stampsz=59)
    sigim, modelsigim = significance_image(im, model, isig, psfstamp,
                                           sz=psfsz)
    sig_max = filters.maximum_filter(sigim, maxfiltershape)
    x, y = numpy.nonzero((sig_max == sigim) & (sigim > threshold) &
                         (keepsat | (isig > 0)))
    fluxratio = im[x, y]/numpy.clip(model[x, y], 0.01, numpy.inf)
    sigratio = (im[x, y]*isig[x, y])/numpy.clip(modelsigim[x, y], 0.01,
                                                numpy.inf)
    sigratio2 = sigim[x, y]/numpy.clip(modelsigim[x, y], 0.01, numpy.inf)
    keepsatcensrc = keepsat & (isig[x, y] == 0)
    m = ((isig[x, y] > 0) | keepsatcensrc)  # ~saturated, or saturated & keep
    if dq is not None and numpy.any(dq[x, y] & nodeblend_maskbit):
        nodeblend = (dq[x, y] & nodeblend_maskbit) != 0
        blendthreshold = numpy.ones_like(x)*blendthreshold
        blendthreshold[nodeblend] = 100
    if dq is not None and numpy.any(dq[x, y] & sharp_maskbit):
        sharp = (dq[x, y] & sharp_maskbit) != 0
        msharp = ~sharp | psfvalsharpcut(
            x, y, sigim, isig, psfstamp, psfvalsharpcutfac=psfvalsharpcutfac,
            psfsharpsat=psfsharpsat)
        # keep if not nebulous region or sharp peak.
        m = m & msharp

    m = m & ((sigratio2 > blendthreshold*2) |
             ((fluxratio > blendthreshold) & (sigratio > blendthreshold/4.) &
              (sigratio2 > blendthreshold)))

    return x[m], y[m]


def peakfind_multiband(ims, models, isigs, dq, psfs, *,
                       band_weights=None, keepsat=False, threshold=5,
                       blendthreshold=0.3, psfvalsharpcutfac=0.7,
                       psfsharpsat=0.7, maxfiltershape=3, psfsz=59):
    """
    Identify significant, non-blended peaks in a *joint* multiband fit.

    Parameters
    ----------
    ims, models, isigs : list of 2-D ndarray (one per band)
        Sky-subtracted data, current model, and inverse-sigma/weight images.
    dq : 2-D ndarray or None
        Combined data-quality mask. 
    psfs : list of PSF functions, one per band
    band_weights : list(float), optional
        Relative weight for each band (must match 'significance_image_multiband').
    keepsat, threshold, blendthreshold, psfvalsharpcutfac, psfsharpsat,
    maxfiltershape, psfsz : see single-band 'peakfind'.

    Returns
    -------
    x, y : 1-D int arrays
        Pixel coordinates of accepted peaks.
    """
    import numpy as np
    nband = len(ims)
    if band_weights is None: band_weights = [1.0 / nband] * nband

    # Joint significance images
    psfstamp1 = psfs[0](int(ims[0].shape[0]/2.), int(ims[0].shape[1]/2.), deriv=False, stampsz=psfsz)
    psfstamp2 = psfs[1](int(ims[1].shape[0]/2.), int(ims[1].shape[1]/2.), deriv=False, stampsz=psfsz)
    psfstamps = [psfstamp1, psfstamp2]
    sigim, modelsigim = significance_image_multiband(ims, models, isigs, psfstamps, band_weights=band_weights, sz=psfsz)

    # Build *weighted* combined image, model, and isig for the flux / significance ratios.
    im_tot    = np.sum([w * im    for w, im    in zip(band_weights, ims)],    axis=0)
    model_tot = np.sum([w * mod   for w, mod   in zip(band_weights, models)], axis=0)
    isig_tot  = np.sqrt(np.sum([(w * isig)**2 for w, isig in zip(band_weights, isigs)], axis=0))
    psf_tot  = np.sum([w * psfstamp for w, psfstamp in zip(band_weights, psfstamps)], axis=0)

    # Candidate peaks: same max-filter trick as single band
    sig_max = filters.maximum_filter(sigim, maxfiltershape)
    x, y = np.nonzero((sig_max == sigim) & (sigim > threshold) & (keepsat | (isig_tot > 0)))

    # # Candidate peaks: keep joint OR single-band detections
    # sig_max_joint = filters.maximum_filter(sigim, maxfiltershape)
    # sig_max_bands = [filters.maximum_filter(s, maxfiltershape) for s in sigims_single]
    # cand_mask = ((sig_max_joint == sigim) & (sigim > threshold))
    # for sig_band, sig_max_band in zip(sigims_single, sig_max_bands):
    #     cand_mask |= ((sig_max_band == sig_band) & (sig_band > threshold))
    # cand_mask &= (keepsat | (isig_tot > 0))
    # x, y = np.nonzero(cand_mask)
    
    fluxratio  = im_tot[x, y] / np.clip(model_tot[x, y], 0.01, np.inf)
    sigratio   = (im_tot[x, y] * isig_tot[x, y]) / np.clip(modelsigim[x, y], 0.01, np.inf)
    sigratio2  = sigim[x, y] / np.clip(modelsigim[x, y], 0.01, np.inf)

    m = ((isig_tot[x, y] > 0) | (keepsat & (isig_tot[x, y] == 0)))    # ~saturated, or saturated & keep

    if dq is not None and np.any(dq[x, y] & nodeblend_maskbit):   # HyperLeda - big galaxies list
        nodeblend = (dq[x, y] & nodeblend_maskbit) != 0
        bt = np.ones_like(x) * blendthreshold
        bt[nodeblend] = 100          
        blendthreshold = bt           
        
    if dq is not None and np.any(dq[x, y] & sharp_maskbit):   # NN nebulosity or very bright stars - sharp them
        sharp = (dq[x, y] & sharp_maskbit) != 0
        msharp = ~sharp | psfvalsharpcut(
            x, y, sigim, isig_tot, psf_tot,
            psfvalsharpcutfac=psfvalsharpcutfac,
            psfsharpsat=psfsharpsat
        )
        m &= msharp

    # Blend / significance conditions (same algebra as single band)
    m &= ((sigratio2 > blendthreshold * 2) |
          ((fluxratio > blendthreshold) &
           (sigratio >  blendthreshold / 4.) &
           (sigratio2 > blendthreshold)))

    return x[m], y[m]

def psfvalsharpcut(x, y, sigim, isig, psf, psfvalsharpcutfac=0.7,
                   psfsharpsat=0.7):
    xl = numpy.clip(x-1, 0, sigim.shape[0]-1)
    xr = numpy.clip(x+1, 0, sigim.shape[0]-1)
    yl = numpy.clip(y-1, 0, sigim.shape[1]-1)
    yr = numpy.clip(y+1, 0, sigim.shape[1]-1)
    # sigim[x, y] should always be >0 from threshold cut.
    psfval1 = 1-(sigim[xl, y]+sigim[xr, y])/(2*sigim[x, y])
    psfval2 = 1-(sigim[x, yl]+sigim[x, yr])/(2*sigim[x, y])
    psfval3 = 1-(sigim[xl, yl]+sigim[xr, yr])/(2*sigim[x, y])
    psfval4 = 1-(sigim[xl, yr]+sigim[xr, yl])/(2*sigim[x, y])
    # in nebulous region, there should be a peak of these around the PSF
    # size, plus a bunch of diffuse things (psfval ~ 0).
    from scipy.signal import fftconvolve
    pp = fftconvolve(psf, psf[::-1, ::-1], mode='same')
    half = psf.shape[0] // 2
    ppcen = pp[half, half]
    psfval1pp = 1-(pp[half-1, half]+pp[half+1, half])/(2*ppcen)
    psfval2pp = 1-(pp[half, half-1]+pp[half, half+1])/(2*ppcen)
    psfval3pp = 1-(pp[half-1, half-1]+pp[half+1, half+1])/(2*ppcen)
    psfval4pp = 1-(pp[half-1, half+1]+pp[half+1, half-1])/(2*ppcen)
    fac = psfvalsharpcutfac*(1-psfsharpsat*(isig[x, y] == 0))
    # more forgiving if center is masked.
    res = ((psfval1 > psfval1pp*fac) & (psfval2 > psfval2pp*fac) &
           (psfval3 > psfval3pp*fac) & (psfval4 > psfval4pp*fac))
    return res


def build_model(x, y, flux, nx, ny, psf=None, psflist=None, psfderiv=False,
                stampsz=59):
    if psf is None and psflist is None:
        raise ValueError('One of psf and psflist must be set')
    if psf is not None and psflist is not None:
        raise ValueError('Only one of psf and psflist must be set')
    if psflist is None:
        psflist = build_psf_list(x, y, psf, stampsz, psfderiv=psfderiv)
        sz = numpy.ones(len(x), dtype='i4')*stampsz
    else:
        sz = numpy.array([tpsf[0].shape[-1] for tpsf in psflist[0]])
        if len(sz) > 0:
            stampsz = numpy.max(sz)

    stampszo2 = stampsz//2
    im = numpy.zeros((nx, ny), dtype='f4')
    im = numpy.pad(im, [stampszo2, stampszo2], constant_values=0.,
                   mode='constant')
    xp = numpy.round(x).astype('i4')
    yp = numpy.round(y).astype('i4')
    # _subtract_ stampszo2 to move from the center of the PSF to the edge
    # of the stamp.
    # _add_ it back to move from the original image to the padded image.
    xe = xp - sz//2 + stampszo2
    ye = yp - sz//2 + stampszo2
    repeat = 3 if psfderiv else 1
    for i in range(len(x)):
        for j in range(repeat):
            im[xe[i]:xe[i]+sz[i], ye[i]:ye[i]+sz[i]] += (
                psflist[j][i][:, :]*flux[i*repeat+j])
    im = im[stampszo2:-stampszo2, stampszo2:-stampszo2]
    return im


def build_psf_list(x, y, psf, sz, psfderiv=True):
    """Make a list of PSFs of the right size, hopefully efficiently."""
    sz = numpy.broadcast_to(sz, x.shape)
    psflist = {}
    for tsz in numpy.unique(sz):
        m = sz == tsz
        res = psf(x[m], y[m], stampsz=tsz, deriv=psfderiv)
        if not psfderiv:
            res = [res]
        psflist[tsz] = res
    counts = {tsz: 0 for tsz in numpy.unique(sz)}
    out = [[] for i in range(3 if psfderiv else 1)]
    for i in range(len(x)):
        for j in range(len(out)):
            out[j].append(psflist[sz[i]][j][counts[sz[i]]])
        counts[sz[i]] += 1
    return out


def in_padded_region(flatcoord, imshape, pad):
    coord = numpy.unravel_index(flatcoord, imshape)
    m = numpy.zeros(len(flatcoord), dtype='bool')
    for c, length in zip(coord, imshape):
        m |= (c < pad) | (c >= length - pad)
    return m


def fit_once(im, x, y, psfs, weight=None,
             psfderiv=False, nskyx=0, nskyy=0,
             guess=None):
    """Fit fluxes for psfs at x & y in image im.

    Args:
        im (ndarray[NX, NY] float): image to fit
        x (ndarray[NS] float): x coord
        y (ndarray[NS] float): y coord
        psf (ndarray[sz, sz] float): psf stamp
        weight (ndarray[NX, NY] float): weight for image
        psfderiv (tuple(ndarray[sz, sz] float)): x, y derivatives of psf image
        nskyx (int): number of sky pixels in x direction (0 or >= 3)
        nskyy (int): number of sky pixels in y direction (0 or >= 3)

    Returns:
        tuple(flux, model, sky)
        flux: output of optimization routine; needs to be refined
        model (ndarray[NX, NY]): best fit model image
        sky (ndarray(NX, NY]): best fit model sky
    """
    # sparse matrix, with rows at first equal to the fluxes at each peak
    # later add in the derivatives at each peak
    sz = numpy.array([tpsf[0].shape[-1] for tpsf in psfs[0]])
    if len(sz) > 0:
        stampsz = numpy.max(sz)
    else:
        stampsz = 19
    stampszo2 = stampsz // 2
    szo2 = sz // 2
    nx, ny = im.shape
    pad = stampszo2 + 1
    im = numpy.pad(im, [pad, pad], constant_values=0.,
                   mode='constant')
    if weight is None:
        weight = numpy.ones_like(im)
    weight = numpy.pad(weight, [pad, pad], constant_values=0.,
                       mode='constant')
    weight[weight == 0.] = 1.e-20
    pix = numpy.arange(stampsz*stampsz, dtype='i4').reshape(stampsz, stampsz) #pixel grid to track where each pixel lands on the image
    # convention: x is the first index, y is the second
    # sorry.
    xpix = pix // stampsz
    ypix = pix % stampsz
    xp = numpy.round(x).astype('i4')
    yp = numpy.round(y).astype('i4')
    # _subtract_ stampszo2 to move from the center of the PSF to the edge
    # of the stamp.
    # _add_ pad back to move from the original image to the padded image.
    xe = xp - stampszo2 + pad
    ye = yp - stampszo2 + pad
    repeat = 1 if not psfderiv else 3
    nskypar = nskyx * nskyy
    npixim = im.shape[0]*im.shape[1]
    zsz = (repeat*numpy.sum(sz*sz) + nskypar*npixim).astype('i8')   #Total number of non-zero entries in the sparse matrix 'A'. Sky parameters take nskypar non-zero entries to be precise, but this way is just easier. 
    if zsz >= 2**32:
        raise ValueError(
            'Number of pixels being fit is too large (>2**32); '
            'failing early.  This usually indicates a problem with '
            'the choice of PSF size & too many sources.')
    xloc = numpy.zeros(zsz, dtype='i4')   #row number (i.e., pixel index) of the non-zero entries
    values = numpy.zeros(len(xloc), dtype='f4')   #values of the non-zero entries
    colnorm = numpy.zeros(len(x)*repeat+nskypar, dtype='f4')   #number of columns in 'A' matrix - Compute the norm (magnitude) of the column vector for stability.
    first = 0   #pointer that tracks where in the xloc and values arrays we will write data.
    for i in range(len(xe)):  #looping over each source
        # f and l crops the stamp if it's greater than the psf. f stands for 'first index' and l stands for 'last index' to pick from the psf stamp.
        f = stampszo2-szo2[i]
        l = stampsz - f
        wt = weight[xe[i]:xe[i]+stampsz, ye[i]:ye[i]+stampsz][f:l, f:l]   #extracts a stampsz x stampsz region from the padded weight map centered around the source. Then crops it to sz
        for j in range(repeat):
            xloc[first:first+sz[i]**2] = (
                numpy.ravel_multi_index(((xe[i]+xpix[f:l, f:l]),    #gives the actual x,y pixel coordinates of each PSF pixel on the padded image.
                                         (ye[i]+ypix[f:l, f:l])),
                                        im.shape)).reshape(-1)
            # yloc[first:first+sz[i]**2] = i*repeat+j
            values[first:first+sz[i]**2] = (
                (psfs[j][i][:, :]*wt).reshape(-1))     #stores PSF × weight for each pixel
            colnorm[i*repeat+j] = numpy.sqrt(
                numpy.sum(values[first:first+sz[i]**2]**2.))
            colnorm[i*repeat+j] += (colnorm[i*repeat+j] == 0)    #Column-wise L2 norm (for scaling and LSQR stability).Otherwise, bright sources (large flux) would dominate the solution.
            values[first:first+sz[i]**2] /= colnorm[i*repeat+j]
            first += sz[i]**2

    if nskypar != 0:
        sxloc, syloc, svalues = sky_parameters(nx+pad*2, ny+pad*2,
                                               nskyx, nskyy, weight)
        startidx = len(x)*repeat   #where sky columns begin in matrix (after source columns)
        nskypix = len(sxloc[0])   #total pixels(rows in A) affected by each sky parameter
        for i in range(len(sxloc)):
            xloc[first:first+nskypix] = sxloc[i]
            # yloc[first:first+nskypix] = startidx+syloc[i]
            colnorm[startidx+i] = numpy.sqrt(numpy.sum(svalues[i]**2.))
            colnorm[startidx+i] += (colnorm[startidx+i] == 0.)
            values[first:first+nskypix] = svalues[i] / colnorm[startidx+i]
            first += nskypix
    shape = (im.shape[0]*im.shape[1], len(x)*repeat+nskypar)  #Final shape of the sparse matrix, A

    from scipy import sparse
    #csc_indptr defines the start and end of each column in the sparse matrix. It grows by sz[i]**2 per source column, and by nskypix per sky column
    csc_indptr = numpy.cumsum([sz[i]**2 for i in range(len(x))
                               for j in range(repeat)])
    csc_indptr = numpy.concatenate([[0], csc_indptr])
    if nskypar != 0:
        csc_indptr = numpy.concatenate([csc_indptr, [
            csc_indptr[-1] + i*nskypix for i in range(1, nskypar+1)]])
    mat = sparse.csc_matrix((values, xloc, csc_indptr), shape=shape,
                            dtype='f4')
    if guess is not None:
        # guess is a guess for the fluxes and sky; no derivatives.
        guessvec = numpy.zeros(len(xe)*repeat+nskypar, dtype='f4')
        guessvec[0:len(xe)*repeat:repeat] = guess[0:len(xe)]
        if nskypar > 0:
            guessvec[-nskypar:] = guess[-nskypar:]
        guessvec *= colnorm
    else:
        guessvec = None
    flux = lsqr_cp(mat, (im*weight).ravel(), atol=1.e-4, btol=1.e-4,
                   guess=guessvec)
    model = mat.dot(flux[0]).reshape(*im.shape)
    flux[0][:] = flux[0][:] / colnorm    #Undo column normalization

    #Crop out the padding region
    im = im[pad:-pad, pad:-pad]
    model = model[pad:-pad, pad:-pad]
    weight = weight[pad:-pad, pad:-pad]
    if nskypar != 0:
        sky = sky_model(flux[0][-nskypar:].reshape(nskyx, nskyy),
                        nx+pad*2, ny+pad*2)
        sky = sky[pad:-pad, pad:-pad]
    else:
        sky = model * 0  #return a blank sky image
        
    model = model / (weight + (weight == 0))   #Undo the weight multiplication applied earlier to the image
    res = (flux, model, sky)
    return res

def fit_once_multiband(images, x, y, psfs, weights=None,
                       psfderiv=False, nskyx=0, nskyy=0, guess=None):
    """
    Multiband fitting of fluxes for shared (x, y) positions across bands.

    Parameters
    ----------
    images : list of ndarray
        List of sky-subtracted images, one per band.
    x, y : ndarray
        Shared source coordinates (length N).
    psfs : list of list of ndarray
        psfs[b][j][i] gives PSF stamp for source i and derivatives j (j<=3) in band b.
    weights : list of ndarray, optional
        One weight map per band. Defaults to uniform if None.
    psfderiv : bool
        Whether PSF derivatives are included (dx, dy).
    nskyx, nskyy : int
        Sky model parameterization per band (0 or >=3).
    guess : ndarray, optional
        Initial guess vector (flux only, no derivatives).

    Returns
    -------
    flux : ndarray
        Solution vector where flux[0] is of shape (B*N + (2*N if psfderiv) + B*nskypar,), ordered as:
        [ per-band fluxes (B*N), shared dx (N), shared dy (N), sky params (B*nskypar) ].
    model : list of ndarray
        Modeled image for each band.
    sky : list of ndarray
        Modeled sky image for each band.
    """

    B = len(images)
    N = len(x)
    sz_b = [numpy.array([psfs[b][0][i].shape[-1] for i in range(N)]) for b in range(B)]
    szo2_b = [sz // 2 for sz in sz_b]
    
    # for safety, I pick a common stampsz across both bands. DOUBT
    stampsz = max(max(sz) for sz in sz_b) if N > 0 else 19
    pad = stampsz // 2 + 1

    images_pad = [numpy.pad(im, [pad, pad], constant_values=0.) for im in images]
    weights_pad = []
    for b in range(B):
        if weights is None or weights[b] is None:
            w = numpy.ones_like(images[b])
        else:
            w = weights[b]
        w = numpy.pad(w, [pad, pad], constant_values=0.)
        w[w == 0] = 1e-20
        weights_pad.append(w)

    xp = numpy.round(x).astype('i4')
    yp = numpy.round(y).astype('i4')
    xe = xp - stampsz // 2 + pad
    ye = yp - stampsz // 2 + pad

    pix = numpy.arange(stampsz*stampsz).reshape(stampsz, stampsz)
    xpix = pix // stampsz
    ypix = pix % stampsz

    nskypar = nskyx * nskyy
    npixim = images_pad[0].shape[0] * images_pad[0].shape[1]

    # ---- parameter count ----
    ncol = B * N                       # fluxes
    if psfderiv:
        ncol += 2 * N                  # shared dx, dy per source
    ncol += B * nskypar                # sky terms

    zsz = (sum(numpy.sum(sz_b[b]**2) for b in range(B)) * (1 + (2 if psfderiv else 0))
           + B * nskypar * npixim).astype('i8')
    
    if zsz >= 2**32: raise ValueError("Too many pixels — check PSF size and sources")

    xloc = numpy.zeros(zsz, dtype='i4')
    values = numpy.zeros(zsz, dtype='f4')
    colnorm = numpy.zeros(ncol, dtype='f4')

    first = 0
    # --- flux columns ---
    for b in range(B):
        for i in range(N):
            f = stampsz // 2 - szo2_b[b][i]
            l = stampsz - f
            wt = weights_pad[b][xe[i]:xe[i]+stampsz, ye[i]:ye[i]+stampsz][f:l, f:l]

            xloc[first:first+sz_b[b][i]**2] = (
                numpy.ravel_multi_index(((xe[i]+xpix[f:l, f:l]),
                                         (ye[i]+ypix[f:l, f:l])),
                                         images_pad[b].shape).reshape(-1)
                + b * npixim
            )
            values[first:first+sz_b[b][i]**2] = (psfs[b][0][i] * wt).reshape(-1)
            col_idx = b * N + i
            colnorm[col_idx] = numpy.sqrt(numpy.sum(values[first:first+sz_b[b][i]**2]**2.))
            colnorm[col_idx] += (colnorm[col_idx] == 0)
            values[first:first+sz_b[b][i]**2] /= colnorm[col_idx]
            first += sz_b[b][i]**2

    # --- shared derivative columns ---
    if psfderiv:
        for i in range(N):
            for d, j in enumerate([1, 2]):  # dx, dy
                col_idx = B * N + 2*i + d
                vals = []
                idxs = []
                for b in range(B):
                    f = stampsz // 2 - szo2_b[b][i]
                    l = stampsz - f
                    wt = weights_pad[b][xe[i]:xe[i]+stampsz, ye[i]:ye[i]+stampsz][f:l, f:l]
                    pix_idx = (
                        numpy.ravel_multi_index(((xe[i]+xpix[f:l, f:l]),
                                                 (ye[i]+ypix[f:l, f:l])),
                                                 images_pad[b].shape).reshape(-1)
                        + b * npixim
                    )
                    if guess is not None:
                        vals.append((guess[b*N + i] * psfs[b][j][i] * wt).reshape(-1))   #Multiply guessflux so we directly get the centroid shifts without math.
                    else:
                        vals.append((psfs[b][j][i] * wt).reshape(-1))   
                    idxs.append(pix_idx)
                idxs = numpy.concatenate(idxs)
                vals = numpy.concatenate(vals)
                colnorm[col_idx] = numpy.sqrt(numpy.sum(vals**2)) or 1.0
                values[first:first+len(vals)] = vals / colnorm[col_idx]
                xloc[first:first+len(vals)] = idxs
                first += len(vals)


    if nskypar > 0:
        for b in range(B):
            sxloc, syloc, svalues = sky_parameters(
                images[b].shape[0] + 2*pad, images[b].shape[1] + 2*pad,
                nskyx, nskyy, weights_pad[b]
            )
            startidx = (B * N + (2*N if psfderiv else 0)) + b * nskypar
            for i in range(len(sxloc)):
                xloc[first:first+len(sxloc[i])] = sxloc[i] + b * npixim
                colnorm[startidx + syloc[i]] = numpy.sqrt(numpy.sum(svalues[i]**2.))
                colnorm[startidx + syloc[i]] += (colnorm[startidx + syloc[i]] == 0)
                values[first:first+len(sxloc[i])] = svalues[i] / colnorm[startidx + syloc[i]]
                first += len(sxloc[i])

    shape = (B * npixim, ncol)

    # csc_indptr = []    # # list to hold the number of nonzeros in each column
    csc_indptr = [0]   # always start at 0
    
    # --- Flux columns (per band)---
    for b in range(B):                  # loop over bands
        for i in range(N):              # loop over sources
            nnz = sz_b[b][i]**2         # each flux column has PSF stamp pixels
            csc_indptr.append(csc_indptr[-1] + nnz)
    
    # --- Shared derivative columns ---
    if psfderiv:
        for i in range(N):              # loop over sources
            # dx column
            nnz_dx = sum(sz_b[b][i]**2 for b in range(B))  # one combined column across all bands
            csc_indptr.append(csc_indptr[-1] + nnz_dx)
    
            # dy column
            nnz_dy = sum(sz_b[b][i]**2 for b in range(B))  # same size as dx
            csc_indptr.append(csc_indptr[-1] + nnz_dy)
    
    # --- Sky columns (per band) ---
    if nskypar > 0:
        for b in range(B):
            for _ in range(nskypar):
                csc_indptr.append(csc_indptr[-1] + npixim)
                # each sky basis spans the whole image, so nnz = npixim

    mat = sparse.csc_matrix((values, xloc, csc_indptr), shape=shape, dtype='f4')

    if guess is not None:
        # total parameter count = ncol = B * N + (2 * N if psfderiv else 0) + B * nskypar
        guessvec = numpy.zeros(ncol, dtype='f4')
        for b in range(B):
            start = b * N; end = (b + 1) * N
            guessvec[start:end] = guess[start:end]  # Fill only the flux slots
    
        # --- sky guesses ---
        if nskypar > 0:
            # take last B*nskypar entries of guess for sky params
            guessvec[-B * nskypar:] = guess[-B * nskypar:]
        guessvec *= colnorm
    else: guessvec = None


    rhs = numpy.concatenate([(images_pad[b]*weights_pad[b]).ravel() for b in range(B)])
    # print("rhs.shape before lsqr_cp:", rhs.shape)
    # print("mat.shape:", mat.shape)
    flux = lsqr_cp(mat, rhs.ravel(), atol=1.e-4, btol=1.e-4, guess=guessvec)
    flux_scaled = flux[0].copy()          # still column-normalised to be used for model
    flux[0] /= colnorm            # Undo column normalization. This will be the returned flux and will be used in sky model

    model = []
    sky = []
    # index where sky parameters start
    sky_offset = B * N + (2 * N if psfderiv else 0)
    
    for b in range(B):
        # --- build model image ---
        band_model = mat[b * npixim:(b + 1) * npixim, :].dot(flux_scaled)
        band_model = band_model.reshape(images_pad[b].shape)
        band_model = band_model[pad:-pad, pad:-pad]
        weight = weights_pad[b][pad:-pad, pad:-pad] if weights else 1.
        band_model /= (weight + (weight == 0))
        model.append(band_model)
    
        # --- sky model ---
        if nskypar > 0:
            sky_params = flux[0][sky_offset + b * nskypar : sky_offset + (b + 1) * nskypar]
            skymap = sky_model(sky_params.reshape(nskyx, nskyy),
                               images[b].shape[0] + 2*pad, images[b].shape[1] + 2*pad)
            sky.append(skymap[pad:-pad, pad:-pad])
        else:
            sky.append(numpy.zeros_like(model[-1]))

    return flux, numpy.array(model), numpy.array(sky)


def unpack_fitpar(guess, nsource, psfderiv):
    """Extract fluxes and sky parameters from fit parameter vector."""
    repeat = 3 if psfderiv else 1
    return guess[0:nsource*repeat:repeat], guess[nsource*repeat:]


def lsqr_cp(aa, bb, guess=None, **kw):
    # implement two speed-ups:
    # 1. "column preconditioning": make sure each column of aa has the same
    #    norm
    # 2. allow guesses

    # column preconditioning is important (substantial speedup), and has
    # been implemented directly in fit_once.

    # allow guesses: solving Ax = b is the same as solving A(x-x*) = b-Ax*.
    # => A(dx) = b-Ax*.  So we can solve for dx instead, then return dx+x*.
    # This improves speed if we reduce the tolerance.
    from scipy.sparse import linalg

    if guess is not None:
        bb2 = bb - aa.dot(guess)
        if 'btol' in kw:
            fac = numpy.sum(bb**2.)**(0.5)/numpy.sum(bb2**2.)**0.5
            kw['btol'] = kw['btol']*numpy.clip(fac, 0.1, 10.)
    else:
        bb2 = bb.copy()

    normbb = numpy.sum(bb2**2.)
    bb2 /= normbb**(0.5)
    par = linalg.lsqr(aa, bb2, **kw)
    # for some reason, everything ends up as double precision after this
    # or lsmr; lsqr seems to be better
    # par[0][:] *= norm**(-0.5)*normbb**(0.5)
    par[0][:] *= normbb**0.5
    if guess is not None:
        par[0][:] += guess
    par = list(par)
    par[0] = par[0].astype('f4')
    par[9] = par[9].astype('f4')
    return par


def compute_centroids(x, y, psflist, flux, im, resid, weight,
                      derivcentroids=False, centroidsize=19):
    # define c = integral(x * I * P * W) / integral(I * P * W)
    # x = x/y coordinate, I = isolated stamp, P = PSF model, W = weight
    # Assuming I ~ P(x-y) for some small offset y and expanding,
    # integrating by parts gives:
    # y = 2 / integral(P*P*W) * integral(x*(I-P)*W)
    # that is the offset we want.

    # we want to compute the centroids on the image after the other sources
    # have been subtracted off.
    # we construct this image by taking the residual image, and then
    # star-by-star adding the model back.

    # ---------------------------
    # KEEP (stamps): build PSF stamps centered to 'centroidsize'
    # ---------------------------

    psfs = [numpy.zeros((len(x), centroidsize, centroidsize), dtype='f4')
            for i in range(len(psflist))]
    
    for j in range(len(psflist)):
        for i in range(len(x)):
            psf_stamp = psfmod.central_stamp(psflist[j][i], censize=centroidsize)
            if numpy.all(psf_stamp == 0):
                # Fallback: rebuild from canonical PSF at (x[i], y[i])
                psf_stamp = psfmod.central_stamp(psflist[j][i], censize=centroidsize, force_center=True)
                if psf_stamp is None or np.all(psf_stamp == 0):
                    print(f"[repair-final] source {i}, comp {j}: could not restore PSF stamp")
            psfs[j][i, :, :] = psf_stamp
        
    
    # for j in range(len(psflist)):
    #     for i in range(len(x)):
    #         psfs[j][i, :, :] = psfmod.central_stamp(psflist[j][i],
    #                                                 censize=centroidsize)

    stampsz = psfs[0].shape[-1]
    stampszo2 = (stampsz-1)//2

    # ---------------------------
    # REMOVE (centroid math): pixel coordinate grids used for integrals
    # ---------------------------
    # dx = numpy.arange(stampsz, dtype='i4')-stampszo2
    # dx = dx.reshape(-1, 1)
    # dy = dx.copy().reshape(1, -1)

    # ---------------------------
    # KEEP (stamps): source centers & padding for stamp extraction
    # ---------------------------
    xp = numpy.round(x).astype('i4')
    yp = numpy.round(y).astype('i4')
    # subtracting to get to the edge of the stamp, adding back to deal with the padded image.
    xe = xp - stampszo2 + stampszo2
    ye = yp - stampszo2 + stampszo2
    resid = numpy.pad(resid, [stampszo2, stampszo2], constant_values=0., mode='constant')
    weight = numpy.pad(weight, [stampszo2, stampszo2], constant_values=0., mode='constant')
    im = numpy.pad(im, [stampszo2, stampszo2], constant_values=0., mode='constant')
    repeat = len(psflist)

    # ---------------------------
    # KEEP (stamps): extract residual/data/weight stamps
    # ---------------------------
    residst = numpy.array([resid[xe0:xe0+stampsz, ye0:ye0+stampsz] for (xe0, ye0) in zip(xe, ye)])
    weightst = numpy.array([weight[xe0:xe0+stampsz, ye0:ye0+stampsz] for (xe0, ye0) in zip(xe, ye)])
    psfst = psfs[0] * flux[:len(x)*repeat:repeat].reshape(-1, 1, 1)
    imst = numpy.array([im[xe0:xe0+stampsz, ye0:ye0+stampsz] for (xe0, ye0) in zip(xe, ye)])

    if len(x) == 0:
        # KEEP (stamps): empty-safe fallbacks
        weightst = psfs[0].copy()
        residst = psfs[0].copy()
        imst = psfs[0].copy()

    # ---------------------------
    # KEEP (stamps): reconstruct model stamp (adds derivative comps if present)
    # ---------------------------
    modelst = psfst.copy()
    if len(psflist) > 1:
        modelst += psfs[1]*flux[1:len(x)*repeat:repeat].reshape(-1, 1, 1)
        modelst += psfs[2]*flux[2:len(x)*repeat:repeat].reshape(-1, 1, 1)

    # ---------------------------
    # REMOVE (centroid math): integral-based centroid estimator and correction
    # ---------------------------
    # cen = []
    # ppw = numpy.sum(modelst*modelst*weightst, axis=(1, 2))
    # pp = numpy.sum(modelst*modelst, axis=(1, 2))
    # for dc in (dx, dy):
    #     xrpw = numpy.sum(dc[None, :, :]*residst*modelst*weightst, axis=(1, 2))
    #     xmmpm = numpy.sum(dc[None, :, :]*(modelst-psfst)*modelst, axis=(1, 2))
    #     cen.append(2*xrpw/(ppw + (ppw == 0.))*(ppw != 0.) +
    #                2*xmmpm/(pp + (pp == 0.))*(pp != 0.))
    # xcen, ycen = cen

    # ---------------------------
    # REMOVE (centroid gating/fallback): psf quality fraction & derivative fallback
    # ---------------------------
    # norm = numpy.sum(modelst, axis=(1, 2))
    # norm = norm + (norm == 0)
    # psfqf = numpy.sum(modelst*(weightst > 0), axis=(1, 2)) / norm
    # how should we really be doing this?  derivcentroids is the first order
    # approximation to the right thing.  the centroid computation that I do
    # otherwise should be unbiased but noisier than optimal for significantly
    # offset peaks.  Vakili, Hogg (2016) say that I should convolve with the
    # PSF and interpolate to the brightest point with some polynomial.  I
    # expected this to be slow (convolving thousands of stamps individually
    # with the PSF each iteration), but the spread_model code worked pretty
    # well, so this is probably a worthwhile thing to try.  if it worked, it
    # would obviate some of the code mess above, and be optimal, so that
    # sounds worthwhile.
    
    # if not derivcentroids:
    #     m = psfqf < 0.5
    # else:
    #     m = numpy.ones(len(xcen), dtype='bool')
    # xcen[m] = 0.
    # ycen[m] = 0.
    # if (len(psflist) > 1) and numpy.sum(m) > 0:
    #     ind = numpy.flatnonzero(m)
    #     # just use the derivative-based centroids for this case.
    #     fluxnz = flux[repeat*ind]
    #     fluxnz = fluxnz + (fluxnz == 0)
    #     xcen[ind] = flux[repeat*ind+1]/fluxnz
    #     ycen[ind] = flux[repeat*ind+2]/fluxnz

    # ---------------------------
    # NEW: centroid outputs are zeros; stamps are unchanged
    # ---------------------------
    xcen = numpy.zeros(len(x), dtype='f4')
    ycen = numpy.zeros(len(x), dtype='f4')

    if numpy.any([numpy.all(w == 0) for w in weightst]):
        print("Warning: empty weight stamp for some sources")


    # stamps: 0: neighbor-subtracted images,
    # 1: images,
    # 2: psfs with shifts
    # 3: weights
    # 4: psfs without shifts
    # print("modelst = psfstack from compute_centroids. Prinitng modelst[0][0], modelst[1][0], modelst[2][0], modelst[3][0] in order", modelst[0][0], modelst[1][0], modelst[2][0], modelst[3][0])
    res = (xcen, ycen, (modelst+residst, imst, modelst, weightst, psfst))
    return res


def compute_centroids_multiband(x, y, psflists, flux, images, resids, weights, 
                                derivcentroids=False, centroidsize=19):
    """
    Stamp-only multiband wrapper for centroid computation.

    Parameters
    ----------
    images : list of ndarray
        Sky-subtracted images, one per band.
    x, y : ndarray
        Shared source coordinates (length N).
    psflists : list of list of ndarray
        psflists[b][j][i] gives PSF stamp for source i and derivative j in band b.
    weights : list of ndarray
        One weight map per band.
    flux : ndarray
        Flattened flux vector from fit_once_multiband.
        Layout is:
          [ f^(1)_1...f^(1)_N, f^(2)_1...f^(2)_N, ..., f^(B)_1...f^(B)_N,
            (optional) dx_1, dy_1, dx_2, dy_2, ..., dx_N, dy_N,
            (optional) sky params ... ]
    resids : list of ndarray
        Residual images per band (= data - model - sky).
    derivcentroids : bool
        If True, include derivative PSFs (not actually used now, centroids forced to 0).
    centroidsize : int
        Side length of cutout stamps.

    Returns
    -------
    xcen, ycen : ndarray
        Dummy centroids (zeros).
    stamps : tuple of ndarrays
        Each element has shape (B, N, S, S):
        (model+resid, data, model, weight, psf*flux).
    """
    import numpy as np
    B = len(images)
    N = len(x)
    base_flux_len = B * N
    has_deriv = len(flux) >= base_flux_len + 2 * N
    deriv_offset = base_flux_len if has_deriv else None

    stamps_b = []
    for b in range(B):
        # --- slice flux block for this band ---
        start = b * N
        stop  = (b + 1) * N
        flux_band = flux[start:stop]

        # --- repack into stride-3 style expected by compute_centroids ---
        # inside compute_centroids_multiband, per band b
        Kb = len(psflists[b])   # 1 (last iter) or 3 (earlier iters)
        
        if Kb == 3:
            # stride-3 interleaved (f, dx, dy) as I have now
            flux_b = np.zeros(3 * N, dtype='f4')
            flux_b[0:3*N:3] = flux_band
            # only fill dx,dy if they truly exist in the global flux:
            if len(flux) >= base_flux_len + 2*N:
                dx_block = flux[base_flux_len : base_flux_len + 2*N : 2]
                dy_block = flux[base_flux_len+1 : base_flux_len + 2*N : 2]
                flux_b[1:3*N:3] = dx_block
                flux_b[2:3*N:3] = dy_block
        else:
            # PSF-only: pass a length-N vector, no padding with zeros
            flux_b = flux_band

        if has_deriv:
            dx_block = flux[deriv_offset : deriv_offset + 2*N : 2]
            dy_block = flux[deriv_offset+1 : deriv_offset + 2*N : 2]
            flux_b[1:3*N:3] = dx_block
            flux_b[2:3*N:3] = dy_block

        # pass to single-band centroid computation
        _, _, stamps = compute_centroids(
            x, y, psflists[b], flux_b,
            images[b], resids[b], weights[b],
            derivcentroids=derivcentroids, centroidsize=centroidsize
        )
        stamps_b.append(stamps)

    # stack per element across bands -> (B, N, S, S)
    stacked = tuple( np.stack([st[idx] for st in stamps_b], axis=0) for idx in range(len(stamps_b[0])))    # collect element idx across all bands

    return np.zeros(N, dtype='f4'), np.zeros(N, dtype='f4'), stacked


def estimate_sky_background(im):
    """Find peak of count distribution; pretend this is the sky background."""
    # for some reason, I have found this hard to work robustly.  Replace with
    # median at the moment.

    return numpy.median(im)


def sky_im(im, weight=None, npix=20, order=1):
    """Remove sky from image."""
    nbinx, nbiny = (numpy.ceil(sh/1./npix).astype('i4') for sh in im.shape)
    xg = numpy.linspace(0, im.shape[0], nbinx+1).astype('i4')
    yg = numpy.linspace(0, im.shape[1], nbiny+1).astype('i4')
    val = numpy.zeros((nbinx, nbiny), dtype='f4')
    usedpix = numpy.zeros((nbinx, nbiny), dtype='f4')
    if weight is None:
        weight = numpy.ones_like(im, dtype='f4')
    if numpy.all(weight == 0):
        return im*0
    # annoying!
    for i in range(nbinx):
        for j in range(nbiny):
            use = weight[xg[i]:xg[i+1], yg[j]:yg[j+1]] > 0
            usedpix[i, j] = numpy.sum(use)
            if usedpix[i, j] > 0:
                val[i, j] = estimate_sky_background(
                    im[xg[i]:xg[i+1], yg[j]:yg[j+1]][use])
    val[usedpix < 20] = 0.
    usedpix[usedpix < 20] = 0.
    from scipy.ndimage.filters import gaussian_filter
    count = 0
    while numpy.any(usedpix == 0):
        sig = 0.4
        valc = gaussian_filter(val*(usedpix > 0), sig, mode='constant')
        weightc = gaussian_filter((usedpix != 0).astype('f4'), sig,
                                  mode='constant')
        m = (usedpix == 0) & (weightc > 1.e-10)
        val[m] = valc[m]/weightc[m]
        usedpix[m] = 1
        count += 1
        if count > 100:
            m = usedpix == 0
            val[m] = numpy.median(im)
            print('Sky estimation failed badly.')
            break
    x = numpy.arange(im.shape[0])
    y = numpy.arange(im.shape[1])
    xc = (xg[:-1]+xg[1:])/2.
    yc = (yg[:-1]+yg[1:])/2.
    from scipy.ndimage import map_coordinates
    xp = numpy.interp(x, xc, numpy.arange(len(xc), dtype='f4'))
    yp = numpy.interp(y, yc, numpy.arange(len(yc), dtype='f4'))
    xpa = xp.reshape(-1, 1)*numpy.ones(len(yp)).reshape(1, -1)
    ypa = yp.reshape(1, -1)*numpy.ones(len(xp)).reshape(-1, 1)
    coord = [xpa.ravel(), ypa.ravel()]
    bg = map_coordinates(val, coord, mode='nearest', order=order)
    bg = bg.reshape(im.shape)
    return bg


def get_sizes(x, y, imbs, weight=None, blist=None, blistsz=299,
              cutofflist=None):
    if cutofflist is None:
        cutofflist = [
            (-numpy.inf, 19), (1000, 59), (20000, 149)]
    x = numpy.round(x).astype('i4')
    y = numpy.round(y).astype('i4')
    peakbright = imbs[x, y]

    if weight is not None:
        # treat saturated / off edge sources as very bright.
        peakbright[weight[x, y] == 0] = cutofflist[-1][0] + 1

    sz = numpy.zeros(len(x), dtype='i4')
    nbright = list()
    for cutoff, tsz in cutofflist:
        m = peakbright > cutoff
        sz[m] = tsz
        nbright.append(numpy.sum(m))

    if ((len(nbright) > 2) and (nbright[-1] > 100) and
            (nbright[-1] > nbright[-2] / 2)):
        print('Too many bright sources, using smaller PSF stamp size...')
        sz[peakbright > cutofflist[-2][0]] = cutofflist[-2][1]

    # sources near listed sources get very big PSF
    if blist is not None and len(x) > 0:
        for xb, yb in zip(blist[0], blist[1]):
            dist2 = (x-xb)**2 + (y-yb)**2
            indclose = numpy.argmin(dist2)
            if dist2[indclose] < 5**2:
                sz[indclose] = blistsz
    return sz


def fit_im_force(im, x, y, psf, weight=None, dq=None, psfderiv=True,
                 nskyx=0, nskyy=0, refit_psf=False,
                 niter=4, blist=None, derivcentroids=False, refit_sky=True,
                 startsky=numpy.nan):
    repeat = 3 if psfderiv else 1
    guessflux = None
    msky = 0
    model = 0

    if len(x) == 0:
        raise ValueError('must force some sources')

    if derivcentroids and not psfderiv:
        raise ValueError('derivcentroids only makes sense when psfderiv '
                         'is true')

    for titer in range(niter):
        for c, s in zip((x, y), im.shape):
            if numpy.any((c < -0.499) | (c > s-0.501)):
                c[:] = numpy.clip(c, -0.499, s-0.501)
                print('Some positions within 0.01 pix of edge of image '
                      'clipped back to 0.01 pix inside image.')
        if (refit_sky and
                ((titer > 0) or numpy.any(~numpy.isfinite(startsky)))):
            sky = sky_im(im-model, weight=weight, npix=100)
        else:
            sky = startsky
        sz = get_sizes(x, y, im-sky-msky, weight=weight, blist=blist)
        minsz = numpy.min(sz)
        psfs = [numpy.zeros((len(x), minsz, minsz), dtype='f4')
                for i in range(repeat)]
        if guessflux is not None:
            guess = guessflux.copy()
        else:
            guess = None
        # should really only be done once in refit_psf=False case
        psfsfull = build_psf_list(x, y, psf, sz, psfderiv=psfderiv)
        # need to package some "tiling" around this eventually, probably?
        flux, model, msky = fit_once(
                im-sky, x, y, psfsfull,
                psfderiv=psfderiv, weight=weight, guess=guess,
                nskyx=nskyx, nskyy=nskyy)
        import gc
        gc.collect()
        flux = flux[0]
        skypar = flux[len(x)*repeat:]
        guessflux = flux[:len(x)*repeat:repeat]
        for i in range(repeat):
            psfs[i][...] = [psfmod.central_stamp(psfsfull[i][j], minsz)
                            for j in range(len(psfsfull[i]))]
        centroids = compute_centroids(x, y, psfs, flux, im-(sky+msky),
                                      im-model-sky,
                                      weight, derivcentroids=derivcentroids)
        xcen, ycen, stamps = centroids
        if refit_psf:
            psf, x, y = refit_psf_from_stamps(psf, x, y, xcen, ycen,
                                              stamps)
            # we are letting the positions get updated, even when
            # psfderiv is false, only for the mean shift that
            # gets introduced when we recentroid all the stars.
            # we could eliminate this by replacing the above with
            # psf, _, _ = refit_psf_from_stamps(...)
            # for WISE at the moment, this should _mostly_ introduce
            # a mean shift, and potentially also a small subpixel-offset
            # related shift.
        if psfderiv:
            if derivcentroids:
                maxstep = 1
            else:
                maxstep = 3
            dcen = numpy.sqrt(xcen**2 + ycen**2)
            m = dcen > maxstep
            xcen[m] /= dcen[m]
            ycen[m] /= dcen[m]
            x, y = (numpy.clip(c, -0.499, s-0.501)
                    for c, s in zip((x+xcen, y+ycen), im.shape))
        print('Iteration %d, median sky %6.2f' %
              (titer+1, numpy.median(sky+msky)))

    stats = compute_stats(x-numpy.round(x), y-numpy.round(y),
                          stamps[0], stamps[2], stamps[3], stamps[1], flux)
    if dq is not None:
        stats['flags'] = extract_im(x, y, dq).astype('i4')
    stats['sky'] = extract_im(x, y, sky+msky).astype('f4')

    stars = OrderedDict([('x', x), ('y', y), ('flux', flux),
                         ('deltx', xcen), ('delty', ycen)] +
                        [(f, stats[f]) for f in stats])
    dtypenames = list(stars.keys())
    dtypeformats = [stars[n].dtype for n in dtypenames]
    dtype = dict(names=dtypenames, formats=dtypeformats)
    stars = numpy.fromiter(zip(*stars.values()),
                           dtype=dtype, count=len(stars['x']))
    res = (stars, model+sky, sky+msky, psf)
    return res


def refit_psf_from_stamps(psf, x, y, xcen, ycen, stamps, name=None,
                          plot=False):
    # how far the centroids of the model PSFs would
    # be from (0, 0) if instantiated there
    # this initial definition includes the known offset (since
    # we instantiated off a pixel center), and the model offset
    xe, ye = psfmod.simple_centroid(
        psfmod.central_stamp(stamps[4], censize=stamps[0].shape[-1]))
    # now we subtract the known offset
    xe -= x-numpy.round(x)
    ye -= y-numpy.round(y)
    if hasattr(psf, 'fitfun'):
        psffitfun = psf.fitfun
        npsf = psffitfun(x, y, xcen+xe, ycen+ye, stamps[0],
                         stamps[1], stamps[2], stamps[3], nkeep=200,
                         name=name, plot=plot)
        if npsf is not None:
            npsf.fitfun = psffitfun
    else:
        shiftx = xcen + xe + x - numpy.round(x)
        shifty = ycen + ye + y - numpy.round(y)
        npsf = find_psf(x, shiftx, y, shifty,
                        stamps[0], stamps[3], stamps[1])
        # we removed the centroid offset of the model PSFs;
        # we need to correct the positions to compensate
    if npsf is not None:
        xnew = x + xe
        ynew = y + ye
        psf = npsf
    else:
        xnew = x
        ynew = y
    return psf, xnew, ynew



def fit_im(im, psf, weight=None, dq=None, psfderiv=True,
           nskyx=0, nskyy=0, refit_psf=False,
           verbose=False, miniter=4, maxiter=10, blist=None,
           maxstars=40000, derivcentroids=False,
           ntilex=1, ntiley=1, fewstars=100, threshold=5,
           ccd=None, plot=False, titer_thresh=2, blendthreshu=2,
           psfvalsharpcutfac=0.7, psfsharpsat=0.7):
    
    if isinstance(weight, int):
        weight = numpy.ones_like(im)*weight

    model = numpy.zeros_like(im)
    xa = numpy.zeros(0, dtype='f4')
    ya = xa.copy()
    lsky = numpy.median(im[weight > 0])
    hsky = numpy.median(im[weight > 0])
    msky = numpy.zeros_like(im)
    passno = numpy.zeros(0, dtype='i4')
    guessflux, guesssky = None, None
    titer = -1
    lastiter = -1
    skypar = {}  # best sky parameters so far.

    roughfwhm = psfmod.neff_fwhm(psf(im.shape[0]//2, im.shape[1]//2))
    roughfwhm = numpy.max([roughfwhm, 3.])

    while True:
        titer += 1
        hsky = sky_im(im-model, weight=weight, npix=20)
        lsky = sky_im(im-model, weight=weight, npix=50*roughfwhm)
        if titer != lastiter:
            # in first passes, do not split sources!
            blendthresh = blendthreshu if titer < titer_thresh else 0.2


            xn, yn = peakfind(im-model-hsky,
                              model-msky, weight, dq, psf,
                              keepsat=(titer == 0),
                              blendthreshold=blendthresh,
                              threshold=threshold,
                              psfvalsharpcutfac=psfvalsharpcutfac,
                              psfsharpsat=psfsharpsat)

            # print("Iter", titer, "new peak count:", len(xn))
            # print("Residual stats:", numpy.mean(im - model - hsky), numpy.std(im - model - hsky))
            
            if len(xa) > 0 and len(xn) > 0:
                keep = neighbor_dist(xn, yn, xa, ya) > 1.5   ### DOUBT - How to decide the threshold?
                xn, yn = (c[keep] for c in (xn, yn))
            if (titer == 0) and (blist is not None):
                xnb, ynb = add_bright_stars(xn, yn, blist, im)
                xn = numpy.concatenate([xn, xnb]).astype('f4')
                yn = numpy.concatenate([yn, ynb]).astype('f4')

            xa, ya = (numpy.concatenate([xa, xn]).astype('f4'),
                      numpy.concatenate([ya, yn]).astype('f4'))
            passno = numpy.concatenate([passno, numpy.zeros(len(xn))+titer])
        else:
            xn, yn = numpy.zeros(0, dtype='f4'), numpy.zeros(0, dtype='f4')

        if titer != lastiter:
            if (titer == maxiter-1) or (
                    (titer >= miniter-1) and (len(xn) < fewstars)) or (
                    len(xa) > maxstars):
                lastiter = titer + 1
        # we probably don't want the sizes to change very much.  hsky certainly
        # will change a bit from iteration to iteration, though.
        sz = get_sizes(xa, ya, im-hsky-msky, weight=weight, blist=blist)
        if guessflux is not None:
            guess = numpy.concatenate([guessflux, numpy.zeros_like(xn)])
        else:
            guess = None
        sky = hsky if titer >= 2 else lsky

        # in final iteration, no longer allow shifting locations; just fit
        # centroids.
        tpsfderiv = psfderiv if lastiter != titer else False
        repeat = 1+tpsfderiv*2
        if len(sz) != 0:
            minsz = numpy.min(sz)
        else:
            minsz = 19
        psfs = [numpy.zeros((len(xa), minsz, minsz), dtype='f4')
                for i in range(repeat)]
        flux = numpy.zeros(len(xa)*repeat, dtype='f4')
        if verbose:
            subreg_iter = 0
            t0 = time.time()
            print("Starting subregion iterations")
        for (bdxf, bdxl, bdxaf, bdxal, bdyf, bdyl, bdyaf, bdyal) in (
                subregions(im.shape, ntilex, ntiley)):
            if verbose:
                print(f"Subregion iteration {subreg_iter} starting; "
                      f"dt={time.time()-t0}", flush=True)
                subreg_iter += 1
            mbda = in_bounds(xa, ya, [bdxaf-0.5, bdxal-0.5],
                             [bdyaf-0.5, bdyal-0.5])
            mbd = in_bounds(xa, ya, [bdxf-0.5, bdxl-0.5],
                            [bdyf-0.5, bdyl-0.5])
            psfsbda = build_psf_list(xa[mbda], ya[mbda], psf, sz[mbda],
                                     psfderiv=tpsfderiv)
            sall = numpy.s_[bdxaf:bdxal, bdyaf:bdyal]
            spri = numpy.s_[bdxf:bdxl, bdyf:bdyl]
            dx, dy = bdxal-bdxaf, bdyal-bdyaf
            sfit = numpy.s_[bdxf-bdxaf:dx+bdxl-bdxal,
                            bdyf-bdyaf:dy+bdyl-bdyal]
            weightbda = weight[sall] if weight is not None else None
            guessmbda = guess[mbda] if guess is not None else None
            guesssky = skypar.get((bdxf, bdyf))
            guessmbda = (numpy.concatenate([guessmbda, guesssky])
                         if guessmbda is not None else None)
            tflux, tmodel, tmsky = fit_once(
                im[sall]-sky[sall], xa[mbda]-bdxaf, ya[mbda]-bdyaf, psfsbda,
                psfderiv=tpsfderiv, weight=weightbda, guess=guessmbda,
                nskyx=nskyx, nskyy=nskyy)
            if numpy.all(numpy.isnan(tmodel)):
                raise ValueError("Model is all NaNs")
            model[spri] = tmodel[sfit]
            msky[spri] = tmsky[sfit]
            ind = numpy.flatnonzero(mbd)
            ind2 = numpy.flatnonzero(mbd[mbda])
            for i in range(repeat):
                flux[ind*repeat+i] = tflux[0][ind2*repeat+i]
            skypar[(bdxf, bdyf)] = flux[numpy.sum(mbda)*repeat:]
            for i in range(repeat):
                if len(ind2) == 0:
                    continue
                psfs[i][mbd] = [psfmod.central_stamp(psfsbda[i][tind], minsz)
                                for tind in ind2]
            # try to free memory!  Not sure where the circular reference
            # could be, but this makes a factor of a few difference
            # in peak memory usage on fields with lots of stars with
            # large models...
            del psfsbda
            import gc
            gc.collect()

        ## Old centroiding call (redundant after derivative-based centroids in fit_once). We perform the next block instead.
        # centroids = compute_centroids(xa, ya, psfs, flux, im-(sky+msky),
        #                               im-model-sky,
        #                               weight, derivcentroids=derivcentroids)
        # xcen, ycen, stamps = centroids
        
        ## ----- NEW: centroid shifts from derivative coefficients -----
        ## Compute xcen,ycen from the linear-fit derivative amplitudes (Fx/F, Fy/F)
        ## rather than from compute_centroids() integrals.
        N = len(xa)
        if N == 0:
            xcen = numpy.zeros(0, dtype='f4')
            ycen = numpy.zeros(0, dtype='f4')
        else:
            # repeat corresponds to this iteration's derivative setting
            # (already set above as: repeat = 1 + tpsfderiv*2)
            if repeat > 1:
                F  = flux[0:N*repeat:repeat]
                Fx = flux[1:N*repeat:repeat]
                Fy = flux[2:N*repeat:repeat]
                eps = 1e-20
                xcen = (Fx / (F + eps)).astype('f4')
                ycen = (Fy / (F + eps)).astype('f4')
            else:
                # Final pass (tpsfderiv=False) or no-deriv mode: no shift
                xcen = numpy.zeros(N, dtype='f4')
                ycen = numpy.zeros(N, dtype='f4')

        ## ----- KEEP: build stamps (compute_centroids is now stamp-only) -----
        ## We still need stamps for compute_stats, PSF refit, and fluxunc below.
        _, _, stamps = compute_centroids(
            xa, ya, psfs, flux,
            im-(sky+msky),          # 'im' argument (data minus sky)
            im-model-sky,           # 'resid' argument (neighbor-subtracted residual)
            weight,
            derivcentroids=derivcentroids
        )

        if titer == lastiter:
            stats = compute_stats(xa-numpy.round(xa), ya-numpy.round(ya),
                                  stamps[0], stamps[1],
                                  stamps[2], stamps[3],
                                  flux)
            if dq is not None:
                stats['flags'] = extract_im(xa, ya, dq).astype('i4')
            stats['sky'] = extract_im(xa, ya, sky+msky).astype('f4')

            # print("N candidates before pruning:", len(xa))
            # print("Kept after brightenough:", numpy.sum(brightenough))
            # print("Kept after isolatedenough:", numpy.sum(isolatedenough))
            # print("Final:", numpy.sum(keep))

            break
        guessflux = flux[:len(xa)*repeat:repeat]
        if refit_psf and len(xa) > 0:
            psf, xa, ya = refit_psf_from_stamps(psf, xa, ya, xcen, ycen, stamps, name=(titer, ccd), plot=plot)
            
        # enforce maximum step
        if derivcentroids:
            maxstep = 1
        else:
            maxstep = 3
        dcen = numpy.sqrt(xcen**2 + ycen**2)
        m = dcen > maxstep
        xcen[m] /= dcen[m]
        ycen[m] /= dcen[m]
        xa, ya = (numpy.clip(c, -0.499, s-0.501)
                  for c, s in zip((xa+xcen, ya+ycen), im.shape))
        fluxunc = numpy.sum(stamps[2]**2.*stamps[3]**2., axis=(1, 2))
        fluxunc = fluxunc + (fluxunc == 0)*1e-20
        fluxunc = (fluxunc**(-0.5)).astype('f4')
        # for very bright stars, fluxunc is unreliable because the entire
        # (small) stamp is saturated.
        # these stars all have very bright inferred fluxes
        # i.e., 50k saturates, so we can cut there.
        brightenough = (guessflux/fluxunc > threshold*3/5.) | (guessflux > 1e5)     
        isolatedenough = cull_near(xa, ya, guessflux)

        keep = brightenough & isolatedenough
        xa, ya = (c[keep] for c in (xa, ya))
        passno = passno[keep]
        guessflux = guessflux[keep]
        if verbose:
            print('Extension %s, iteration %2d, found %6d sources; %4d close and '
                  '%4d faint sources removed.' %
                  (ccd, titer+1, len(xn),
                   numpy.sum(~isolatedenough),
                   numpy.sum(~brightenough & isolatedenough)))

        # should probably also subtract these stars from the model image
        # which is used for peak finding.  But the faint stars should
        # make little difference?

    # This is the end of the internal iteration loops
    # Prepares found sources for export
    stars = OrderedDict([('x', xa), ('y', ya), ('flux', flux),
                         ('passno', passno)] +
                        [(f, stats[f]) for f in stats])
    dtypenames = list(stars.keys())
    dtypeformats = [stars[n].dtype for n in dtypenames]
    dtype = dict(names=dtypenames, formats=dtypeformats)
    stars = numpy.fromiter(zip(*stars.values()),
                           dtype=dtype, count=len(stars['x']))
    res = (stars, model+sky, sky+msky, psf)
    return res


def fit_im_multiband(images, psfb, weights=None, dq
                     =None, band_weights=None, psfderiv=True,
                     nskyx=0, nskyy=0, refit_psf=False,
                     verbose=False, miniter=4, maxiter=10, blist=None,
                     maxstars=40000, derivcentroids=False,
                     ntilex=1, ntiley=1, fewstars=100, threshold=5,
                     ccd=None, plot=False, titer_thresh=2, blendthreshu=2,
                     psfvalsharpcutfac=0.7, psfsharpsat=0.7):
    """
    Multiband version of fit_im:
    - Fits fluxes for shared source positions across multiple bands.
    - Uses fit_once_multiband, compute_centroids_multiband and compute_stats_multiband.
    """

    import numpy as np
    from collections import OrderedDict
    import gc, time

    B = len(images)   # number of bands
    shape = images[0].shape

    models = [np.zeros_like(im) for im in images]
    mskys  = [np.zeros_like(im) for im in images]
    xa = np.zeros(0, dtype='f4')
    ya = xa.copy()
    passno = np.zeros(0, dtype='i4')
    guessflux, guesssky = None, None
    titer, lastiter = -1, -1
    skypar = {}  # per-tile sky params
    nskypar = nskyx * nskyy

    # Estimate rough FWHM from first band’s PSF
    roughfwhm = psfmod.neff_fwhm(psfb[0](shape[0]//2, shape[1]//2))
    roughfwhm = np.max([roughfwhm, 3.])

    while True:
        titer += 1

        # Estimate sky backgrounds separately per band
        hskys, lskys = [], []
        for b in range(B):
            hskys.append(sky_im(images[b]-models[b], weight=weights[b], npix=20))
            lskys.append(sky_im(images[b]-models[b], weight=weights[b], npix=50*roughfwhm))

        if titer != lastiter:
            # --- Peak finding ---
            blendthresh = blendthreshu if titer < titer_thresh else 0.2

            xn, yn = peakfind_multiband([images[b] - models[b] - hskys[b] for b in range(B)],
                              [models[b] - mskys[b] for b in range(B)], weights, dq, psfb,
                              band_weights = band_weights,
                              keepsat=(titer == 0),
                              blendthreshold=blendthresh,
                              threshold=threshold,
                              psfvalsharpcutfac=psfvalsharpcutfac,
                              psfsharpsat=psfsharpsat)
            
            # Remove duplicates near existing sources
            if len(xa) > 0 and len(xn) > 0:
                keep = neighbor_dist(xn, yn, xa, ya) > 1.5   
                xn, yn = (c[keep] for c in (xn, yn))

            # Add bright-star list if provided
            if (titer == 0) and (blist is not None):
                xnb, ynb = add_bright_stars(xn, yn, blist, coadd_im)
                xn = np.concatenate([xn, xnb]).astype('f4')
                yn = np.concatenate([yn, ynb]).astype('f4')

            xa = np.concatenate([xa, xn]).astype('f4')
            ya = np.concatenate([ya, yn]).astype('f4')
            passno = np.concatenate([passno, np.zeros(len(xn))+titer])
        else:
            xn, yn = np.zeros(0, dtype='f4'), np.zeros(0, dtype='f4')

        # Decide stopping criterion
        if titer != lastiter:
            if (titer == maxiter-1) or (
                (titer >= miniter-1) and (len(xn) < fewstars)) or (
                len(xa) > maxstars):
                lastiter = titer + 1
            
        # --- Prepare PSFs and flux arrays per band ---
        minsz_b, sz_b = [], []
        for b in range(B):
            resid  = images[b] - hskys[b] - mskys[b]
            sz = get_sizes(xa, ya, resid, weight=weights[b], blist=blist)
            sz_b.append(sz)
            minsz_b.append(np.min(sz) if len(sz) else 19)


        N = len(xa)
        if guessflux is not None:
            guess = np.concatenate([guessflux, np.zeros_like(xn).repeat(B)])   # (B*N,)
        else: guess = None

        skys = hskys if titer >= 2 else lskys

        # I only use this to decide how many PSF components we build (1 or 3).
        tpsfderiv = psfderiv if lastiter != titer else False
        n_psfcomps = 3 if tpsfderiv else 1

        # PSF stamps per band
        psf_stamps = [[np.zeros((len(xa), minsz_b[b], minsz_b[b]), dtype='f4')
                       for _ in range(n_psfcomps)] for b in range(B)]

        # [B*N fluxes] + [(2*N) shared deriv if enabled] + [B*nskypar sky]
        ncol_global = B*N + (2*N if tpsfderiv else 0) + B*nskypar
        flux = np.zeros(ncol_global, dtype='f4')

        if verbose:
            subreg_iter = 0
            t0 = time.time()
            print("Starting subregion iterations")
                    
        # --- Loop over image subregions ---
        for (bdxf, bdxl, bdxaf, bdxal, bdyf, bdyl, bdyaf, bdyal) in (
                subregions(shape, ntilex, ntiley)):
        
            if verbose:
                print(f"Subregion iteration {subreg_iter} starting; dt={time.time()-t0}", flush=True)
                subreg_iter += 1
        
            # Sources in this subregion (shared positions across all bands)
            mbda_sources = in_bounds(xa, ya, [bdxaf-0.5, bdxal-0.5], [bdyaf-0.5, bdyal-0.5])
            mbd = in_bounds(xa, ya, [bdxf-0.5, bdxl-0.5], [bdyf-0.5, bdyl-0.5])
        
            # Expand to band-space indices for flux/guess (flat [B*N])
            mbda = np.concatenate([b*len(xa) + mbda_sources for b in range(B)])
        
            # Extract cutouts per band
            sall = np.s_[bdxaf:bdxal, bdyaf:bdyal]
            spri = np.s_[bdxf:bdxl, bdyf:bdyl]
        
            dx, dy = bdxal-bdxaf, bdyal-bdyaf
            sfit = np.s_[bdxf-bdxaf:dx+bdxl-bdxal, bdyf-bdyaf:dy+bdyl-bdyal]
        
            # Build PSF lists, images and weights tiles per band
            psfsbda = [build_psf_list(xa[mbda_sources], ya[mbda_sources], psfb[b], sz_b[b][mbda_sources], psfderiv=tpsfderiv) for b in range(B)]
            imgs_tile    = [images[b][sall]  - skys[b][sall] for b in range(B)]
            weights_tile = [weights[b][sall] for b in range(B)]
        
            # initialize skypar entry for this tile if missing
            if (bdxf, bdyf) not in skypar:
                skypar[(bdxf, bdyf)] = np.zeros(B * nskypar, dtype='f4')

            if guess is not None:
                ind = np.flatnonzero(mbda_sources)   # fitted sources (reuse same 'ind')
                M   = len(ind)
                guess_local_flux = []
                for b in range(B):
                    band_block = guess[b*N:(b+1)*N]  # (N,)
                    guess_local_flux.append(band_block[ind].astype('f4'))  # (M,)
                guess_local_flux = np.concatenate(guess_local_flux)        # (B*M,)
                guessmbda = (np.concatenate([guess_local_flux, skypar[(bdxf, bdyf)]])
                             if nskypar > 0 else guess_local_flux)
            else:
                guessmbda = None

        
            # --- Run joint multiband fit ---
            tflux, tmodels, tmskys = fit_once_multiband(
                imgs_tile,
                xa[mbda_sources] - bdxaf,   # positions (shared across bands)
                ya[mbda_sources] - bdyaf,
                psfsbda,
                psfderiv=tpsfderiv,
                weights=weights_tile,
                guess=guessmbda,
                nskyx=nskyx, nskyy=nskyy,
            )
        
            # --- Accumulate ---
            for b in range(B):
                models[b][spri] = tmodels[b][sfit]
                mskys[b][spri]  = tmskys[b][sfit]
        
            # Store fluxes back into global flat array
            ind  = np.flatnonzero(mbda_sources)                # global source indices in tile
            ind2 = np.arange(len(ind))
            M = len(ind2)

            for b in range(B):
                loc_start = b*M
                loc_stop  = (b+1)*M
                # insert into global [b*N : (b+1)*N] at positions 'ind'
                glob_start = b*N
                flux[glob_start + ind] = tflux[0][loc_start:loc_stop]

            # Scatter *shared derivatives* (if present): global deriv block starts at B*N
            if tpsfderiv and M > 0:
                deriv_off_glob = B * N
                deriv_off_loc  = B * M      # after the B*M flux block locally

                # dx (even), dy (odd) interleaved per source locally
                dx_loc = tflux[0][deriv_off_loc     : deriv_off_loc + 2*M : 2]
                dy_loc = tflux[0][deriv_off_loc + 1 : deriv_off_loc + 2*M : 2]

                flux[deriv_off_glob + 2*ind    ] = dx_loc
                flux[deriv_off_glob + 2*ind + 1] = dy_loc

            # --- Update per-tile sky parameters (B*nskypar) ---
            if nskypar > 0:
                skypar[(bdxf, bdyf)] = tflux[0][-B*nskypar:].copy()

            for b in range(B):
                if M > 0:
                    for k in range(n_psfcomps):
                        psf_block = np.stack(
                            [psfmod.central_stamp(psfsbda[b][k][tind], minsz_b[b]) for tind in ind2],
                            axis=0
                        ).astype('f4')  # (M, S, S)
            
                        # Insert into the global psf_stamps
                        psf_stamps[b][k][mbda_sources] = psf_block

            # free memory
            del psfsbda
            gc.collect()
        
        # --- Build stamps for compute_stats ---
        _, _, stamps = compute_centroids_multiband(
            xa, ya, psf_stamps, flux,
            [images[b]-(skys[b]+mskys[b]) for b in range(B)],    # 'im' argument
            [images[b]-models[b]-skys[b] for b in range(B)],     # 'resid' argument
            [weights[b] for b in range(B)], derivcentroids=derivcentroids)
        # breakpoint()

        if titer == lastiter:
            print("LASTITER", lastiter)
            stats = compute_stats_multiband(xa-np.round(xa), ya-np.round(ya), stamps, flux)
            if dq is not None:
                stats['flags'] = extract_im(xa, ya, dq).astype('i4')
            for b in range(B):
                stats[f'sky_b{b}'] = extract_im(xa, ya, skys[b]+mskys[b]).astype('f4')
            break
                
        # --- Centroiding --- 
        N = len(xa)
        has_deriv_glob = (len(flux) >= B*N + 2*N)
        if N > 0 and has_deriv_glob:
            # Shared derivative parameters are exactly centroid shifts (dx, dy) now
            xcen = flux[B*N : B*N + 2*N : 2].astype('f4')   # (N,)
            ycen = flux[B*N + 1 : B*N + 2*N : 2].astype('f4')   # (N,)
        else:
            xcen = np.zeros(N, dtype='f4')
            ycen = np.zeros(N, dtype='f4')

        # Optional per-band PSF refit (mirroring single-band API)
        if refit_psf and N > 0:
            # stamps from compute_centroids_multiband: each entry has shape (B, N, S, S)
            # We need per-band single-band-like tuples for refit_psf_from_stamps.
            # stamps = (impsf, im, model, weight, psf_only)
            for b in range(B):
                # Slice per-band stamp tensors
                impsf_b  = stamps[0][b]
                im_b     = stamps[1][b]
                model_b  = stamps[2][b]
                weight_b = stamps[3][b]
                psfonly_b = stamps[4][b]
                stamps_b = (impsf_b, im_b, model_b, weight_b, psfonly_b)
        
                psfb[b], xa, ya = refit_psf_from_stamps(
                    psfb[b], xa, ya, xcen, ycen, stamps_b,
                    name=(titer, f"{ccd}_b{b}" if ccd is not None else f"b{b}"),
                    plot=plot
                )
        
        # Enforce maximum centroid step and update (x, y)
        maxstep = 1.0 if derivcentroids else 3.0
        dcen = np.sqrt(xcen**2 + ycen**2)
        m = dcen > maxstep
        if np.any(m):
            xcen[m] /= dcen[m]
            ycen[m] /= dcen[m]
        
        # Apply shift and clip to image bounds 
        xa = np.clip(xa + xcen, -0.499, shape[0] - 0.501)
        ya = np.clip(ya + ycen, -0.499, shape[1] - 0.501)

        # Build per-band flux uncertainty (B, N) from stamps
        fluxunc_b = np.sum(stamps[2]**2 * stamps[3]**2, axis=(2, 3))  # (B, N) and stamps shape = (B, N, S, S)
        fluxunc_b = fluxunc_b + (fluxunc_b == 0) * 1e-20
        fluxunc_b = (fluxunc_b**(-0.5)).astype('f4')

        ## Checking for weird dflux values
        if np.any(fluxunc_b > 1e8): print("Huge dflux:", fluxunc_b.max())
        
        guessflux_b = np.vstack([flux[b*N:(b+1)*N] for b in range(B)])   # (B, N)

        # Joint SNR across bands
        snr_b = guessflux_b / (fluxunc_b + 1e-20)            # (B, N)
        snr_joint = np.sqrt(np.sum(snr_b**2, axis=0))        # (N,)

        # Keep if significant jointly
        bright_joint   = snr_joint > threshold*3/5.
        bright_satur   = np.any(guessflux_b > 1e5, axis=0)
        brightenough =  bright_joint | bright_satur

        # --- Isolation cut across bands ---
        guessflux_joint = np.max(guessflux_b, axis=0)      # (N,)
        isolatedenough  = cull_near(xa, ya, snr_joint)
        
        keep = brightenough & isolatedenough             # (N,)  <-- 1-D mask
        xa = xa[keep]
        ya = ya[keep]
        passno = passno[keep]

        guessflux_b = guessflux_b[:, keep]
        guessflux   = guessflux_b.ravel(order='C')   # (B*N_keep,)

        if verbose:
            print('Extension %s, iteration %2d, found %6d sources; %4d close and %4d faint sources removed.' %
                  (ccd, titer+1, len(xn),
                   numpy.sum(~isolatedenough),
                   numpy.sum(~brightenough & isolatedenough)))

    flux_b = [flux[b*N:(b+1)*N] for b in range(B)]
    stars = OrderedDict([ ('x', xa), ('y', ya),] +
                        [(f'flux_b{b}', flux_b[b].astype('f4')) for b in range(B)] + 
                        [('passno', passno),] +
                        [(f, stats[f]) for f in stats])


    dtypenames = list(stars.keys())
    dtypeformats = [stars[n].dtype for n in dtypenames]
    dtype = dict(names=dtypenames, formats=dtypeformats)
    stars = np.fromiter(zip(*stars.values()),dtype=dtype, count=len(stars['x']))

    return stars, [models[b] + skys[b] for b in range(B)], [skys[b]+mskys[b] for b in range(B)], psfb




def compute_stats(xs, ys, impsfstack, imstack, psfstack, weightstack, flux):
    """
    Compute per-source diagnostics (errors, chi2, flux fractions, morphology)
    from postage stamp cutouts.

    Parameters
    ----------
    xs, ys : ndarray
        Source centroid offsets (subpixel x,y positions).
    impsfstack : ndarray (N,S,S)
        Neighbor-subtracted data stamps (model + residual).
    psfstack : ndarray (N,S,S)
        Model PSF × fitted flux stamps for each source NOT the psf template stamp which stamp[4] element
    weightstack : ndarray (N,S,S)
        Weight (inverse noise) stamps.
    imstack : ndarray (N,S,S)
        Raw image data stamps.
    flux : ndarray (N,)
        Fitted source flux amplitudes.

    Returns
    -------
    stats : OrderedDict
        Dictionary of per-source measurements:
        - dx, dy : position uncertainties
        - dflux  : flux uncertainty
        - qf     : quality factor (fraction of valid PSF footprint)
        - rchi2  : reduced chi2 of fit
        - fracflux : fraction of flux modeled
        - fluxlbs/dfluxlbs : large-box summed flux and error
        - fluxiso,xiso,yiso : isophotal flux and centroid
        - fwhm   : effective PSF width
        - spread_model/dspread_model : star/galaxy separator
    """

    # Residuals = (image data - model)
    residstack = impsfstack - psfstack

    # Normalize PSF stamps so they integrate to 1
    norm = numpy.sum(psfstack, axis=(1, 2))

    # print("psf norm", norm[:10])
    # print("psfstack", psfstack[1][0])

    psfstack = psfstack / (norm + (norm == 0)).reshape(-1, 1, 1)

    # Quality factor how “complete” the measurement is = fraction of the PSF footprint that overlaps with good/unmasked pixels.
    qf = numpy.sum(psfstack*(weightstack > 0), axis=(1, 2))

    # Flux uncertainty = inverse sqrt(sum(PSF^2 * weight^2))
    fluxunc = numpy.sum(psfstack**2.*weightstack**2., axis=(1, 2))
    fluxunc = fluxunc + (fluxunc == 0)*1e-20
    fluxunc = (fluxunc**(-0.5)).astype('f4')

    if numpy.any(fluxunc > 1e8):
        print("compute_stats: huge fluxunc", fluxunc.max(), "min denom:", numpy.min(numpy.sum(psfstack**2.*weightstack**2., axis=(1,2))))

    mask_empty = numpy.sum(weightstack > 0, axis=(1,2)) == 0
    if numpy.any(mask_empty):
        print("DEBUG: sources with empty weights:", numpy.nonzero(mask_empty)[0])

    # Initialize arrays for positional uncertainties
    posunc = [numpy.zeros(len(qf), dtype='f4'),
              numpy.zeros(len(qf), dtype='f4')]

    # Derivatives of the normalized PSF (for centroid uncertainty)
    psfderiv = numpy.gradient(-psfstack, axis=(1, 2))
    for i, p in enumerate(psfderiv):
        # Position error sigma_x,sigma_y ∝ 1/sqrt(sum((dPSF * flux * weight)^2))
        dp = numpy.sum((p*weightstack*flux[:, None, None])**2., axis=(1, 2))
        dp = dp + (dp == 0)*1e-40
        dp = dp**(-0.5)
        posunc[i][:] = dp

    # Reduced chi2 of the fit, weighted by PSF footprint
    rchi2 = numpy.sum(residstack**2.*weightstack**2.*psfstack,
                      axis=(1, 2)) / (qf + (qf == 0.)*1e-20).astype('f4')

    # Flux fraction: how much of the flux is explained by the model. Best for point sources, high S/N.
    fracfluxn = numpy.sum(impsfstack*(weightstack > 0)*psfstack, axis=(1, 2))
    fracfluxd = numpy.sum(imstack*(weightstack > 0)*psfstack, axis=(1, 2))
    fracfluxd = fracfluxd + (fracfluxd == 0)*1e-20
    fracflux = (fracfluxn / fracfluxd).astype('f4')

    # Alternative flux estimators
    # - LBS flux: large box summation - big box aperture flux. 
    # Robust sanity check (if PSF model is wrong, we notice a discrepancy).
    fluxlbs, dfluxlbs = compute_lbs_flux(impsfstack, psfstack, weightstack, flux/(norm+(norm == 0)))
    fluxlbs = fluxlbs.astype('f4'); dfluxlbs = dfluxlbs.astype('f4')

    # - Isophotal flux/centroid using PSF derivatives. Flux within brightness contour.
    # Useful for galaxies, extended emission.
    fluxiso, xiso, yiso = compute_iso_fit(impsfstack, psfstack, weightstack, flux/(norm+(norm == 0)), psfderiv)

    # Effective PSF FWHM for this source
    fwhm = psfmod.neff_fwhm(psfstack).astype('f4')

    # Spread model (star/galaxy separator)
    spread, dspread = spread_model(impsfstack, psfstack, weightstack)

    return OrderedDict([('dx', posunc[0]), ('dy', posunc[1]),
                        ('dflux', fluxunc),
                        ('qf', qf), ('rchi2', rchi2), ('fracflux', fracflux),
                        ('fluxlbs', fluxlbs), ('dfluxlbs', dfluxlbs),
                        ('fwhm', fwhm), ('spread_model', spread),
                        ('dspread_model', dspread),
                        ('fluxiso', fluxiso), ('xiso', xiso), ('yiso', yiso)])


def compute_stats_multiband(xs, ys, stamps_mb, flux_flat):
    """
    Multiband wrapper around single-band compute_stats.

    Parameters
    ----------
    xs, ys : ndarray of length N
        Source centroid offsets (subpixel positions).
    stamps_mb : tuple of 5 arrays, each shaped (B, N, S, S)
        (impsfstack, imstack, psfstack, weightstack, psf*flux) for all bands.
    flux_flat : ndarray
        Flattened flux vector from fit_once_multiband.
        Layout is:
          [f^(1)_1...f^(1)_N, f^(2)_1...f^(2)_N, ..., f^(B)_1...f^(B)_N,
           (optional) dx,dy block,
           (optional) sky params]

    Returns
    -------
    stats : OrderedDict
        Per-band fields (with suffix _b{b}) and joint fields:
        - dx_joint, dy_joint : combined position uncertainties
        - rchi2_joint        : combined reduced chi^2
        - snr_joint          : joint SNR across bands
    """
    import numpy as np
    from collections import OrderedDict

    impsf_b, im_b, psf_b, w_b, _ = stamps_mb  # each (B, N, S, S)
    B, N, _, _ = psf_b.shape

    # detect if derivatives present
    base_flux_len = B * N
    has_deriv = len(flux_flat) >= base_flux_len + 2 * N

    # --- slice fluxes only ---
    flux_blocks = [flux_flat[b * N:(b + 1) * N] for b in range(B)]

    dx_list, dy_list, qf_list, rchi2_list, dflux_list, flux_list = [], [], [], [], [], []
    out = OrderedDict()

    for b in range(B):
        flux_b = flux_blocks[b]

        # call single-band compute_stats
        sb = compute_stats(xs, ys,
                           impsf_b[b],     # impsfstack
                           im_b[b],        # imstack (data)
                           psf_b[b],       # psfstack (model)
                           w_b[b],         # weightstack
                           flux_b)         # just fluxes for this band

        for k, v in sb.items():
            out[f"{k}_b{b}"] = v.astype('f4')

        # keep for joint metrics
        dx_list.append(sb['dx'])
        dy_list.append(sb['dy'])
        qf_list.append(sb['qf'])
        rchi2_list.append(sb['rchi2'])
        dflux_list.append(sb['dflux'])
        flux_list.append(flux_b)

    # convert to arrays (B, N)
    dx_arr = np.vstack(dx_list); dy_arr = np.vstack(dy_list)
    qf_arr = np.vstack(qf_list); rchi2_arr = np.vstack(rchi2_list)
    dflux_arr = np.vstack(dflux_list); flux_arr = np.vstack(flux_list)

    # ---- Aggregate joint metrics ----

    # 1) Joint reduced-chi^2:
    #    Each band’s rchi2 = Σ(resid^2 * w^2 * psf) / qf
    #    To combine, sum numerators and denominators across bands:
    #    numerator_b = rchi2_b * qf_b
    num_joint = np.sum(rchi2_arr * qf_arr, axis=0)
    den_joint = np.sum(qf_arr, axis=0)
    rchi2_joint = num_joint / (den_joint + (den_joint == 0)*1e-20)

    # 2) Joint position uncertainties:
    #    Single-band dx,dy are 1σ position errors.
    #    For independent measurements, combine via inverse-variance:
    #    1/σ_joint^2 = Σ (1/σ_b^2)
    inv2_dx = np.sum(1.0 / (dx_arr**2 + 1e-20), axis=0)
    inv2_dy = np.sum(1.0 / (dy_arr**2 + 1e-20), axis=0)
    dx_joint = (1.0 / (inv2_dx + (inv2_dx == 0)*1e-20))**0.5
    dy_joint = (1.0 / (inv2_dy + (inv2_dy == 0)*1e-20))**0.5

    # 3) Joint SNR:
    #    Per-band SNR = F / σ_F
    #    Independent SNRs add in quadrature:
    #    SNR_joint = sqrt( Σ SNR_b^2 )
    snr_bands = flux_arr / (dflux_arr + 1e-20)
    snr_joint = np.sqrt(np.sum(snr_bands**2, axis=0))

    # NOTE:
    # - dx_b{b}, dy_b{b}, dx_joint, dy_joint = *uncertainties* (not centroid shifts).
    # - Actual centroid shifts (dx_fit, dy_fit) come from flux_flat derivative block,
    #   but are not reported here to keep behavior consistent with single-band stats.

    # ---- Add joint metrics to output ----
    out['dx_joint'] = dx_joint.astype('f4')
    out['dy_joint'] = dy_joint.astype('f4')
    out['rchi2_joint'] = rchi2_joint.astype('f4')
    out['snr_joint'] = snr_joint.astype('f4')
    bad = np.where(dflux_arr[b] > 1e8)[0]
    print("Band", b, "bad sources:", bad[:10])
    print("flux_b[bad]", flux_blocks[b][bad][:10])
    print("qf[bad]", sb['qf'][bad][:10])
    print("weight sum[bad]", np.sum(stamps_mb[3][b][bad], axis=(1,2))[:10])

    return out




def spread_model(impsfstack, psfstack, weightstack):
    # need to convolve psfs with 1/16 FWHM exponential
    # can get FWHM from n_eff
    # better way?  n_eff can be a bit annoying; not necessarily what one
    # expects if there's a sharp peak on a broad background.
    # spread_model is on the whole a bit goofy: one sixteenth of a FWHM is very
    # little.  So this is really more like the significance of the derivative
    # of the PSF with radius, which I would compute a bit differently.
    # still, other people compute spread_model, and it's well defined, so...
    import crowdsource.galconv as galconv
    fwhm = psfmod.neff_fwhm(psfstack)
    sigma = fwhm/16.
    re = sigma * 1.67834699
    expgalstack = galconv.gal_psfstack_conv(re, 0, 0, galconv.ExpGalaxy,
                                            numpy.eye(2), 0, 0, psfstack)
    GWp = numpy.sum(expgalstack*weightstack**2*impsfstack, axis=(1, 2))
    PWp = numpy.sum(psfstack*weightstack**2*impsfstack, axis=(1, 2))
    GWP = numpy.sum(expgalstack*weightstack**2*psfstack, axis=(1, 2))
    PWP = numpy.sum(psfstack**2*weightstack**2, axis=(1, 2))
    GWG = numpy.sum(expgalstack**2*weightstack**2, axis=(1, 2))
    spread = (GWp/(PWp+(PWp == 0)) - GWP/(PWP+(PWP == 0)))
    dspread = numpy.sqrt(numpy.clip(
        PWp**2*GWG + GWp**2*PWP - 2*GWp*PWp*GWP, 0, numpy.inf)
                         /(PWp + (PWp == 0))**4)
    return spread, dspread


def extract_im(xa, ya, im, sentinel=999):
    m = numpy.ones(len(xa), dtype='bool')
    for c, sz in zip((xa, ya), im.shape):
        m = m & (c > -0.5) & (c < sz - 0.5)
    res = numpy.zeros(len(xa), dtype=im.dtype)
    res[~m] = sentinel
    xp, yp = (numpy.round(c[m]).astype('i4') for c in (xa, ya))
    res[m] = im[xp, yp]
    return res


def compute_lbs_flux(stamp, psf, isig, apcor):
    sumisig2 = numpy.sum(isig**2, axis=(1, 2))
    sumpsf2isig2 = numpy.sum(psf*psf*isig**2, axis=(1, 2))
    sumpsfisig2 = numpy.sum(psf*isig**2, axis=(1, 2))
    det = numpy.clip(sumisig2*sumpsf2isig2 - sumpsfisig2**2, 0, numpy.inf)
    det = det + (det == 0)
    unc = numpy.sqrt(sumisig2/det)
    flux = (sumisig2*numpy.sum(psf*stamp*isig**2, axis=(1, 2)) -
            sumpsfisig2*numpy.sum(stamp*isig**2, axis=(1, 2)))/det
    flux *= apcor
    unc *= apcor
    return flux, unc


def compute_iso_fit(impsfstack, psfstack, weightstack, apcor, psfderiv):
    nstar = len(impsfstack)
    par = numpy.zeros((nstar, 3), dtype='f4')
    for i in range(len(impsfstack)):
        aa = numpy.array([psfstack[i]*weightstack[i],
                          psfderiv[0][i]*weightstack[i],
                          psfderiv[1][i]*weightstack[i]])
        aa = aa.reshape(3, -1).T
        par[i, :] = numpy.linalg.lstsq(
            aa, (impsfstack[i]*weightstack[i]).reshape(-1), rcond=None)[0]
    zeroflux = par[:, 0] == 0
    return (par[:, 0],
            (1-zeroflux)*par[:, 1]/(par[:, 0]+zeroflux),
            (1-zeroflux)*par[:, 2]/(par[:, 0]+zeroflux))


def sky_model_basis(i, j, nskyx, nskyy, nx, ny):
    from crowdsource import basisspline
    if (nskyx < 3) or (nskyy < 3):
        raise ValueError('Invalid sky model.')
    expandx = (nskyx-1.)/(3-1)
    expandy = (nskyy-1.)/(3-1)
    xg = -expandx/3. + i*2/3.*expandx/(nskyx-1.)
    yg = -expandy/3. + j*2/3.*expandy/(nskyy-1.)
    x = numpy.linspace(-expandx/3.+1/6., expandx/3.-1/6., nx).reshape(-1, 1)
    y = numpy.linspace(-expandy/3.+1/6., expandy/3.-1/6., ny).reshape(1, -1)
    return basisspline.basis2dq(x-xg, y-yg)


def sky_model(coeff, nx, ny):
    # minimum sky model: if we want to use the quadratic basis functions we
    # implemented, and we want to allow a constant sky over the frame, then we
    # need at least 9 basis polynomials: [-0.5, 0.5, 1.5] x [-0.5, 0.5, 1.5].
    nskyx, nskyy = coeff.shape
    if (coeff.shape[0] == 1) & (coeff.shape[1]) == 1:
        return coeff[0, 0]*numpy.ones((nx, ny), dtype='f4')
    if (coeff.shape[0] < 3) or (coeff.shape[1]) < 3:
        raise ValueError('Not obvious what to do for <3')
    im = numpy.zeros((nx, ny), dtype='f4')
    for i in range(coeff.shape[0]):
        for j in range(coeff.shape[1]):
            # missing here: speed up available from knowing that
            # the basisspline is zero over a large area.
            im += coeff[i, j] * sky_model_basis(i, j, nskyx, nskyy, nx, ny)
    return im


def sky_parameters(nx, ny, nskyx, nskyy, weight):
    # yloc: just add rows to the end according to the current largest row
    # in there
    nskypar = nskyx * nskyy
    xloc = [numpy.arange(nx*ny, dtype='i4')]*nskypar
    # for the moment, don't take advantage of the bounded support.
    yloc = [i*numpy.ones((nx, ny), dtype='i4').ravel()
            for i in range(nskypar)]
    if (nskyx == 1) & (nskyy == 1):
        values = [(numpy.ones((nx, ny), dtype='f4')*weight).ravel()
                  for yl in yloc]
    else:
        values = [(sky_model_basis(i, j, nskyx, nskyy, nx, ny)*weight).ravel()
                  for i in range(nskyx) for j in range(nskyy)]
    return xloc, yloc, values


def cull_near(x, y, flux):
    """Delete faint sources within 1 pixel of a brighter source.

    Args:
        x (ndarray, int[N]): x coordinates for N sources
        y (ndarray, int[N]): y coordinates
        flux (ndarray, int[N]): fluxes

    Returns:
        ndarray (bool[N]): mask array indicating sources to keep
    """
    if len(x) == 0:
        return numpy.ones(len(x), dtype='bool')
    m1, m2, dist = match_xy(x, y, x, y, neighbors=6)
    m = (dist < 1) & (flux[m1] < flux[m2]) & (m1 != m2)
    keep = numpy.ones(len(x), dtype='bool')
    keep[m1[m]] = 0
    return keep


def neighbor_dist(x1, y1, x2, y2):
    """Return distance of nearest neighbor to x1, y1 in x2, y2"""
    m1, m2, d12 = match_xy(x2, y2, x1, y1, neighbors=1)
    return d12


def match_xy(x1, y1, x2, y2, neighbors=1):
    """Match x1 & y1 to x2 & y2, neighbors nearest neighbors.

    Finds the neighbors nearest neighbors to each point in x2, y2 among
    all x1, y1."""
    from scipy.spatial import cKDTree
    vec1 = numpy.array([x1, y1]).T
    vec2 = numpy.array([x2, y2]).T
    kdt = cKDTree(vec1)
    dist, idx = kdt.query(vec2, neighbors)
    m1 = idx.ravel()
    m2 = numpy.repeat(numpy.arange(len(vec2), dtype='i4'), neighbors)
    dist = dist.ravel()
    dist = dist
    m = m1 < len(x1)  # possible if fewer than neighbors elements in x1.
    return m1[m], m2[m], dist[m]


def add_bright_stars(xa, ya, blist, im):
    xout = []
    yout = []
    for x, y, mag in zip(*blist):
        if ((x < -0.499) or (x > im.shape[0]-0.501) or
            (y < -0.499) or (y > im.shape[1]-0.501)):
            continue
        if len(xa) > 0:
            mindist2 = numpy.min((x-xa)**2 + (y-ya)**2)
        else:
            mindist2 = 9999
        if mindist2 > 5**2:
            xout.append(x)
            yout.append(y)
    return (numpy.array(xout, dtype='f4'), numpy.array(yout, dtype='f4'))


# This is almost entirely deprecated for the psf.py module... go look there.
def find_psf(xcen, shiftx, ycen, shifty, psfstack, weightstack,
             imstack, stampsz=59, nkeep=100):
    """Find PSF from stamps."""
    # let's just go ahead and correlate the noise
    xr = numpy.round(shiftx)
    yr = numpy.round(shifty)
    psfqf = (numpy.sum(psfstack*(weightstack > 0), axis=(1, 2)) /
             numpy.sum(psfstack, axis=(1, 2)))
    totalflux = numpy.sum(psfstack, axis=(1, 2))
    timflux = numpy.sum(imstack, axis=(1, 2))
    toneflux = numpy.sum(psfstack, axis=(1, 2))
    tmedflux = numpy.median(psfstack, axis=(1, 2))
    tfracflux = toneflux / numpy.clip(timflux, 100, numpy.inf)
    tfracflux2 = ((toneflux-tmedflux*psfstack.shape[1]*psfstack.shape[2]) /
                  numpy.clip(timflux, 100, numpy.inf))
    okpsf = ((numpy.abs(psfqf - 1) < 0.03) &
             (tfracflux > 0.5) & (tfracflux2 > 0.2))
    if numpy.sum(okpsf) > 0:
        shiftxm = numpy.median(shiftx[okpsf])
        shiftym = numpy.median(shifty[okpsf])
        okpsf = (okpsf &
                 (numpy.abs(shiftx-shiftxm) < 1.) &
                 (numpy.abs(shifty-shiftym) < 1.))
    if numpy.sum(okpsf) <= 5:
        print('Fewer than 5 stars accepted in image, keeping original PSF')
        return None
    if numpy.sum(okpsf) > nkeep:
        okpsf = okpsf & (totalflux > -numpy.sort(-totalflux[okpsf])[nkeep-1])
    psfstack = psfstack[okpsf, :, :]
    weightstack = weightstack[okpsf, :, :]
    totalflux = totalflux[okpsf]
    xcen = xcen[okpsf]
    ycen = ycen[okpsf]
    shiftx = shiftx[okpsf]
    shifty = shifty[okpsf]
    for i in range(psfstack.shape[0]):
        psfstack[i, :, :] = shift(psfstack[i, :, :], [-shiftx[i], -shifty[i]])
        if (numpy.abs(xr[i]) > 0) or (numpy.abs(yr[i]) > 0):
            weightstack[i, :, :] = shift(weightstack[i, :, :],
                                         [-xr[i], -yr[i]],
                                         mode='constant', cval=0.)
        # our best guess as to the PSFs & their weights
    # select some reasonable sample of the PSFs
    totalflux = numpy.sum(psfstack, axis=(1, 2))
    psfstack /= totalflux.reshape(-1, 1, 1)
    weightstack *= totalflux.reshape(-1, 1, 1)
    tpsf = numpy.median(psfstack, axis=0)
    tpsf = psfmod.center_psf(tpsf)
    if tpsf.shape == stampsz:
        return tpsf
    xc = numpy.arange(tpsf.shape[0]).reshape(-1, 1)-tpsf.shape[0]//2
    yc = xc.reshape(1, -1)
    rc = numpy.sqrt(xc**2.+yc**2.)
    stampszo2 = psfstack[0].shape[0] // 2
    wt = numpy.clip((stampszo2+1-rc)/4., 0., 1.)
    overlap = (wt != 1) & (wt != 0)

    def objective(par):
        mod = psfmod.moffat_psf(par[0], beta=2.5, xy=par[2], yy=par[3],
                                deriv=False, stampsz=tpsf.shape[0])
        mod /= numpy.sum(mod)
        return ((tpsf-mod)[overlap]).reshape(-1)
    from scipy.optimize import leastsq
    par = leastsq(objective, [4., 3., 0., 1.])[0]
    modpsf = psfmod.moffat_psf(par[0], beta=2.5, xy=par[2], yy=par[3],
                               deriv=False, stampsz=stampsz)
    modpsf /= numpy.sum(psfmod.central_stamp(modpsf))
    npsf = modpsf.copy()
    npsfcen = psfmod.central_stamp(npsf, tpsf.shape[0])
    npsfcen[:, :] = tpsf*wt+(1-wt)*npsfcen[:, :]
    npsf /= numpy.sum(npsf)
    return psfmod.SimplePSF(npsf, normalize=-1)


def subregions(shape, nx, ny, overlap=149):
    # ugh.  I guess we want:
    # starts and ends of each _primary_ fit region
    # starts and ends of each _entire_ fit region
    # should be nothing else?
    # need this for both x and y: 8 things to return.
    nx = nx if nx > 0 else 1
    ny = ny if ny > 0 else 1
    bdx = numpy.round(numpy.linspace(0, shape[0], nx+1)).astype('i4')
    bdlx = numpy.clip(bdx - overlap, 0, shape[0])
    bdrx = numpy.clip(bdx + overlap, 0, shape[0])
    bdy = numpy.round(numpy.linspace(0, shape[1], ny+1)).astype('i4')
    bdly = numpy.clip(bdy - overlap, 0, shape[1])
    bdry = numpy.clip(bdy + overlap, 0, shape[1])
    xf = bdx[:nx]
    xl = bdx[1:]
    xaf = bdlx[:nx]
    xal = bdrx[1:]
    yf = bdy[:nx]
    yl = bdy[1:]
    yaf = bdly[:nx]
    yal = bdry[1:]
    for i in range(nx):
        for j in range(ny):
            yield (xf[i], xl[i], xaf[i], xal[i], yf[j], yl[j], yaf[j], yal[j])


def in_bounds(x, y, xbound, ybound):
    return ((x > xbound[0]) & (x <= xbound[1]) &
            (y > ybound[0]) & (y <= ybound[1]))
