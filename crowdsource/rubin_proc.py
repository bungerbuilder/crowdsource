import numpy as np
import argparse, os, pdb
import crowdsource.psf as psfmod
from crowdsource import crowdsource_base
from lsst.daf.butler import Butler
from functools import partial


def process(visitId, detector, nx = 4, ny = 4, maxstars = 10000000, fewstars = 60, **kw):
    """Use the RSP Butler to find the Rubin image"""
    
    butler = Butler("dp1", collections="LSSTComCam/DP1")
    
    dataset_refs = list(butler.query_datasets(
        "visit_image",
        where="visit.id = :visitId AND detector.id = :detector",
        bind={"visitId": visitId, "detector": detector},
    ))

    if len(dataset_refs) == 0:
        raise RuntimeError(
            f"No visit_image found for visit={visitId}, detector={detector}")

    else:
        print(f"Visit image found for visit={visitId}, detector={detector}")
            
    ref = list(dataset_refs)[0]
    visit_image = butler.get(ref)

    """Get the PSF using wise_psf_fit"""
    
    rubin_psf = visit_image.getPsf()
    visit_center = visit_image.getBBox().getCenter()
    psf_stamp_visit = rubin_psf.computeImage(visit_center).array
    
    stamp = np.clip(np.array(psf_stamp_visit), 1e-10, np.inf)
    stamp = stamp / np.sum(stamp)
    
    psf = psfmod.SimplePSF(stamp)
    psf.fitfun = partial(psfmod.wise_psf_fit, psfstamp=stamp)

    """Making the variables for crowdsource"""
    
    im = visit_image.image.array.astype(np.float32)
    
    var = visit_image.variance.array
    sqivar = np.where(var > 0, 1.0 / np.sqrt(var), 0.0).astype(np.float32)
    
    mask = visit_image.mask
    
    bad_bits = 0
    for plane in ("BAD", "SAT", "CR", "NO_DATA", "EDGE", "INTRP"):
        if plane in mask.getMaskPlaneDict():
            bad_bits |= mask.getPlaneBitMask(plane)
            
    flag = (mask.array & bad_bits).astype(np.uint32)
    sqivar[(mask.array & bad_bits) != 0] = 0.0

    """Run CROWDSOURCE on an image"""
    
    res = crowdsource_base.fit_im(im, psf, sqivar, dq=flag, refit_psf=True,
                                  verbose = True, ntilex=nx, ntiley=ny, **kw)

    print("CROWDSOURCE is done!")
    
    return res, visit_image
