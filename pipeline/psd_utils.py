from typing import Union
import numpy as np
import os
#from stableemrifisher.noise import noise_PSD_AE
from few.summation.interpolatedmodesum import CubicSplineInterpolant
from lisatools.sensitivity import *
from lisatools.utils.constants import lisaLT

from scipy.interpolate import make_interp_spline as interp1d_cpu
try:
    from cupyx.scipy.interpolate import make_interp_spline as interp1d_gpu
except ImportError:
    interp1d_gpu = None 

def write_psd_file(model='scirdv1', 
                    channels='AET', 
                    tdi2=True,
                    include_foreground=False,
                    filename="example_psd.npy",
                    **kwargs
                   ):
    """
    Write a PSD file for a given model.

    Parameters
    ----------
    model : str
        The noise model to use. Default is 'scirdv1'.
    channels : str
        The channels to include in the PSD. Default is 'AET'. if None, the sensitivity curve without projections is computed.
    tdi2 : bool 
        Whether to use Second generation TDI. Default is True.
    include_foreground : bool
        Whether to include the foreground noise. Default is False. This is just an extra check, the actual
        argument is in the kwargs.
    filename : str
        The name of the file to save the PSD to. Default is 'example_psd.npy'.
    **kwargs : dict
        Additional keyword arguments to pass to the PSD generation function.
    """
    
    assert channels in [None,  'A', 'AE', 'AET'], "channels must be None, 'A', 'AE', or 'AET'"
    if include_foreground:
        assert 'stochastic_params' in kwargs.keys(), "`stochastic_params = Tobs [s]` must be provided if include_foreground is True"

    freqs = np.linspace(0, 1, 100001)[1:]
    
    if channels is None:
        sens_fns = [LISASens]

        default_kwargs = dict(
        return_type='PSD',
        average=False
    )
        
    elif 'A' in channels:
        sens_fns = [A1TDISens]
        if 'E' in channels:
            sens_fns.append(E1TDISens)
        if 'T' in channels:
            sens_fns.append(T1TDISens)
        
        default_kwargs = dict(
            return_type='PSD',
        )

    updated_kwargs = default_kwargs | kwargs

    Sn = [get_sensitivity(freqs, sens_fn=sens_fn, model=model, **updated_kwargs) for sens_fn in sens_fns]

    if tdi2:
        x = 2.0 * np.pi * lisaLT * freqs
        tdi_factor = 4 * np.sin(2*x)**2
        Sn = [sens*tdi_factor for sens in Sn]

    Sn = np.array(Sn)
    np.save(filename,np.vstack((freqs, Sn)).T)


def load_psd_from_file(psd_file, xp=np, smooth=True, clip=False):
    """
    Load the PSD from a file and return an interpolant.

    Parameters
    ----------
    psd_file : str
        The name of the file to load the PSD from.
    xp : module
        The module to use for array operations. Default is np.
    smooth : bool
        Whether to smooth the PSD values. Default is True.
    clip : bool
        Whether to clip the PSD values to the range found in the file. Default is False.
    
    Returns
    -------
    psd_interpolant : function
        A function that takes a frequency and returns the PSD at that frequency. If smooth is True, the PSD values are smoothed before interpolation. If clip is True, the PSD values are clipped to the range found in the file.
    """

    psd_in = np.load(psd_file).T
    freqs, values = psd_in[0], np.atleast_2d(psd_in[1:])

    #convert to cupy if needed
    freqs = xp.asarray(freqs)
    values = xp.asarray(values)

    backend = 'cpu' if xp is np else 'gpu'
    print(f"Using {backend} backend for PSD interpolation")
    #min_psd = values[:,0]#np.min(values, axis=1)
    min_psd = np.min(values[:, freqs < 1e-2], axis=1) # compatible with both tdi 1 and tdi 2
    max_psd = np.max(values, axis=1)
    print("PSD range", min_psd, max_psd)
    if smooth:
        psd_interp = get_psd_smoothed_interpolant(freqs, values, xp=xp)
    else:
        psd_interp = CubicSplineInterpolant(freqs, values, force_backend=backend)
    
    if clip:
        def psd_clipped(f, **kwargs):
            f = xp.clip(f, 0.00001, 1.0)

            out = xp.array(
                [
                    xp.clip(xp.atleast_2d(psd_interp(f))[i], min_psd[i], max_psd[i]) for i in range(len(values))
                ]
            )
            return xp.squeeze(out) # remove the extra dimension if there is only one channel
        #breakpoint()
        return psd_clipped    
    else:
        return psd_interp

def get_psd_smoothed_interpolant(freqs: Union[np.ndarray, Any], 
                                psd: Union[np.ndarray, Any], 
                                num_bins: int = 300,
                                xp=np):
    """
    Docstring for get_psd_smoothed_interpolant.

    Credits for the original snippet: Ollie Burke.
    Current implementation from https://gitlab.esa.int/lisa-sgs/l2d/sandbox/emrianalysistools/-/blob/main
    
    :param freqs: Frequencies at which the PSD is evaluated
    :type freqs: Union[np.ndarray, Any]
    :param psd: Power Spectral Density values corresponding to the frequencies
    :type psd: Union[np.ndarray, Any]
    :param num_bins: Number of bins to use for smoothing the PSD
    :type num_bins: int
    :param xp: The module to use for array operations. Default is np.
    """

    # Get xp arrays
    freqs = xp.asarray(freqs)
    psd_vals = xp.atleast_2d(psd)

    # Mask out zero or negative values (for log). psd_vals should be 2D with shape (n_channels, n_freqs)
    mask = (freqs > 0) & (psd_vals > 0).all(axis=0)
    if not mask.any():
        raise ValueError("No valid frequencies or PSD values found after masking.")
    # Apply mask to both freqs and psd_vals
       
    freqs = freqs[mask]
    psd_vals = psd_vals[:, mask]

    nchannels = psd_vals.shape[0]

    # Log-spaced frequency bins
    log_freq_bins = xp.logspace(xp.log10(freqs.min()), xp.log10(freqs.max()), num_bins + 1)
    bin_centers = xp.sqrt(log_freq_bins[:-1] * log_freq_bins[1:])  # geometric mean

    # Digitize and bin average
    digitized = xp.digitize(freqs, log_freq_bins)

    smoothing_loop = [
        psd_vals[:, digitized == i].mean(axis=1) if xp.any(digitized == i) else xp.array([xp.nan] * nchannels)
        for i in range(1, len(log_freq_bins))
    ]

    binned_psd = xp.array(smoothing_loop).T

    # Remove NaNs
    valid = ~xp.isnan(binned_psd[0])
    bin_centers = bin_centers[valid]
    binned_psd = xp.take_along_axis(binned_psd, xp.where(valid)[0][None, :], axis=1)
    # Interpolate back to original frequency grid
    interp1d = interp1d_gpu if xp.__name__ == 'cupy' else interp1d_cpu
    interpolator = interp1d(bin_centers, binned_psd, k=1, axis=-1)
    
    def wrap(x, **kwargs):
        return xp.squeeze(interpolator(x))
    return wrap

def load_psd(
            logger,
            model='scirdv1', 
            channels='AET', 
            tdi2=True, 
            include_foreground=False,
            filename="example_psd.npy",
            xp=np,
            smooth=True,
            clip=False,
            **kwargs
            ):
    """
    Load the PSD from a file and returns an interpolant. If the file does not exist, it will be created.

    Parameters
    ----------
    model : str
        The noise model to use. Default is 'scirdv1'.
    channels : str
        The channels to include in the PSD. Default is 'AET'.
    tdi2 : bool 
        Whether to use Second generation TDI. Default is True.
    include_foreground : bool
        Whether to include the foreground noise. Default is False. This is just an extra check, the actual
        argument is in the kwargs.
    filename : str
        The name of the file to save the PSD to. Default is 'example_psd.npy'.
    xp : module
        The module to use for array operations. Default is np.
    smooth : bool
        Whether to smooth the PSD. Default is True.
    clip : bool
        Whether to clip the PSD values. Default is False.
    **kwargs : dict
        Additional keyword arguments to pass to the PSD generation function.

    Returns
    -------
    psd_interpolant : function
        A function that takes a frequency and returns the PSD at that frequency
    """
    if filename is None or filename == "None":
        tdi_gen = 'tdi2' if tdi2 else 'tdi1'
        foreground = 'wd' if include_foreground else 'no_wd'    
        filename = f"noise_psd_{model}_{channels}_{tdi_gen}_{foreground}.npy"
    if not os.path.exists(filename):
        logger.warning(f"PSD file {filename} does not exist. Creating it now.")
        write_psd_file(model=model, channels=channels, tdi2=tdi2, include_foreground=include_foreground, filename=filename, **kwargs)
    
    logger.info(f"Loading PSD from {filename}")
    return load_psd_from_file(filename, xp=xp, smooth=smooth, clip=clip)

def get_psd_kwargs(kwargs):
    """
    Return a dictionary of default settings for PSD generation. Use the input dictionary to override the defaults.
    """
    default_settings = {
        "model": "scirdv1",
        "channels": "A",
    }
    return default_settings | kwargs

def compute_snr2(freqs, tdi_channels, psd_fn, xp=np):
    """
    """
    df = freqs[2] - freqs[1]
    
    prefactor = 4 * df
    snr2 = prefactor * xp.sum(
        xp.abs(tdi_channels)**2 / psd_fn(freqs),
        axis=(0,1)
    )

    return snr2


if __name__ == "__main__":

    # Example usage: create the three interpolant functions
    psd_emri_1p5 = load_psd_from_file("TDI2_AE_psd_emri_background_1.5_yr.npy", smooth=False, clip=True)
    psd_emri_4p5 = load_psd_from_file("TDI2_AE_psd_emri_background_4.5_yr.npy", smooth=False, clip=True)
    psd_nominal = load_psd_from_file("TDI2_AE_psd.npy", smooth=False, clip=True)
    psd_smoothed = load_psd_from_file("TDI2_AE_psd.npy", smooth=True, clip=False)

    print("Created PSD functions:", psd_emri_1p5, psd_emri_4p5, psd_nominal, psd_smoothed)

    # plot example
    import matplotlib.pyplot as plt
    import scienceplots

    plt.style.use(['science'])

    plot_params = {
        "figure.dpi": "200",
        "axes.labelsize": 20,
        "axes.linewidth": 1.5,
        "axes.titlesize": 20,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.title_fontsize": 16,
        "legend.fontsize": 16,
        "xtick.major.size": 3.5,
        "xtick.major.width": 1.5,
        "xtick.minor.size": 2.5,
        "xtick.minor.width": 1.5,
        "ytick.major.size": 3.5,
        "ytick.major.width": 1.5,
        "ytick.minor.size": 2.5,
        "ytick.minor.width": 1.5,
    }
    plt.rcParams.update(plot_params)
    freqs = np.logspace(-4, 0.0, 1000)
    plt.figure(figsize=(8,4))
    plt.loglog(freqs, psd_nominal(freqs), label="Instrumental Only")
    plt.loglog(freqs, psd_emri_1p5(freqs), "--", label="Mission Duration = 1.5 yr")
    plt.loglog(freqs, psd_emri_4p5(freqs), ":",label="Mission Duration = 4.5 yr")
    plt.loglog(freqs, psd_smoothed(freqs), "-.", label="Smoothed Instrumental Only")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD [1/Hz]")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("tdi2AE_psd.pdf")
    # plt.show()