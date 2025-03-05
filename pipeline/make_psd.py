import numpy as np
from stableemrifisher.noise import noise_PSD_AE
from LISAfom.lisatools import build_lisa_noise
from few.summation.interpolatedmodesum import CubicSplineInterpolant

def make_psd_file(filename="example_psd.npy"):
    freqs = np.linspace(0, 1, 100001)[1:]

    psd = noise_PSD_AE(freqs, TDI = 'TDI2')

    np.save(filename,np.vstack((freqs, psd)).T)


def build_psd_interp(args, logger, xp=np):
    noise = build_lisa_noise(args, logger)
    if args.foreground:
        wd = args.tobs 
        logger.info("Adding the WD confusion foreground with Tobs=%s years", wd)
    else:
        wd = 0.0

    noise.set_wdconfusion(wd)

    noise_psd = xp.asarray(noise.psd(noise.freq, option='A', tdi2=args.tdi2))
    min_psd = noise_psd.min()
    max_psd = noise_psd.max()
    psd_interp =  CubicSplineInterpolant(xp.asarray(noise.freq), noise_psd)

    def psd_clipped(f, **kwargs):
        f = np.clip(f, 0.00001, 1.0)
        return np.clip(psd_interp(f), min_psd, max_psd)

    return psd_clipped