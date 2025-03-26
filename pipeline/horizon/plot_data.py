# plot horizon data for the Kerr eccentric equatorial case

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.lines as mlines
# from seaborn import color_palette

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.model_selection import cross_val_predict

from scipy.interpolate import interp1d
import argparse

import pickle as pkl


default_rcParams = {
        'text.usetex': True,
        'font.family': 'serif',
        "font.serif": ["Computer Modern"],
        'font.weight':'medium',
        'mathtext.fontset': 'cm',
        'text.latex.preamble': r"\usepackage{amsmath}",
        'font.size': 16,
        'figure.figsize': (7, 7),
        'figure.titlesize': 'large',
        'axes.formatter.use_mathtext': True,
        'axes.formatter.limits': [-2, 4],
        'axes.titlesize': 'large',
        'axes.labelsize': 'large',
        'xtick.top': True,
        'xtick.major.size': 5,
        'xtick.minor.size': 3,
        'xtick.major.width': 0.8,
        'xtick.minor.visible': True,
        'xtick.direction': 'in',
        'xtick.labelsize': 'medium',
        'ytick.right': True,
        'ytick.major.size': 5,
        'ytick.minor.size': 3,
        'ytick.major.width': 0.8,
        'ytick.minor.visible': True,
        'ytick.direction': 'in',
        'ytick.labelsize': 'medium',
        'legend.frameon': True,
        'legend.framealpha': 1,
        'legend.fontsize': 'medium',
        'legend.scatterpoints' : 3,
        #'lines.color': 'k',
        'lines.linewidth': 2,
        'patch.linewidth': 1,
        'hatch.linewidth': 1,
        'grid.linestyle': 'dashed',
        'savefig.dpi' : 200,
        'savefig.format' : 'pdf',
        'savefig.bbox' : 'tight',
        'savefig.transparent' : True,
    }

plt.rcParams.update(default_rcParams)


parser = argparse.ArgumentParser(description="Plot horizon data")
parser.add_argument("-base", "--base_name", type=str, default='horizon', help="base name of the data file")
parser.add_argument("-datadir", "--datadir", type=str, default='data/', help="directory where the data is stored")
parser.add_argument("-plotdir", "--plotdir", type=str, default='figures/', help="directory where the plots are stored")
parser.add_argument("-interp", "--interp", default=False, help="interpolate data", action='store_true')
parser.add_argument("-fill", "--fill", default=False, help="fill between data", action='store_true')

args = vars(parser.parse_args())


def add_plot(M_axis, data, ls='solid', frame='detector', colors='k', fill=False, interp=None, interp_kwargs={}, plot_kwargs={}, fig=None, axs=None, use_gpr=True):
    if fig is None or axs is None:
        fig, axs = plt.subplots(ncols=1, nrows=1, sharex=True)

    if isinstance(colors, str):
        colors = [colors] * len(data.keys())

    #Mgrid = np.logspace(5, 8, 1000)
    zmin = 0.0

    M_edge =[] # where mu = 1
    z_edge = [] # where M = 1

    for key, color in zip(data.keys(), colors):

        if isinstance(M_axis, dict):
            M = M_axis[key]
        elif isinstance(M_axis, np.ndarray):
            M = M_axis
        else:
            raise ValueError('M_axis must be either a dictionary or a numpy array')
        
        z_here = data[key][:]
        if frame == 'detector':
            M_source = to_M_source(M, z_here)
        elif frame == 'source':
            M_source = M
        else:
            raise ValueError('frame must be either detector or source')
        
        if interp:

            if 'fill_value' in interp_kwargs.keys() and interp_kwargs['fill_value'] == 'extrapolate':
                    xlow, xhigh = 4, 8
            else:
                xlow, xhigh = np.log10(M_source[0]), np.log10(M_source[-1]*0.99)

            if use_gpr:
                kernel = 1.0 * RBF(length_scale=1e-1, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-10, 1e1))

                gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=30)

                gp.fit(np.log10(M_source.reshape(-1, 1)), z_here)

                x = np.linspace(xlow, xhigh, 1000)
                y, sigma = gp.predict(x.reshape(-1, 1), return_std=True)
                x = 10**x
            else:
                interpolant = interp(M_source, z_here, **interp_kwargs) 
                
                x = np.logspace(xlow, xhigh, 1000)
                y, sigma = interpolant(x), None

        else:
            x = M_source
            y, sigma = z_here, None

        #breakpoint()

        if fill:
            axs.semilogx(x, y, ls=ls, color=color, label=key, **plot_kwargs)
            #rescale y to the same range as the previous plot

            axs.fill_between(x, zmin, y, alpha=0.3, zorder=1, hatch='', color=color, rasterized=True)
            zmin = y
            x_prev = x
        else:
            axs.semilogx(x, y, ls=ls, color=color, label=key, **plot_kwargs)

            if sigma is not None:   
                axs.fill_between(x, y-sigma, y+sigma, alpha=0.3, zorder=1, hatch='', color=color, rasterized=True)
        
    
    axs.set_xlabel(r'$M_{\rm source} \, [M_\odot]$')
    axs.set_ylabel(r'$\bar{z}$')

    return fig, axs


def to_logM_source(logM, z):
    return logM - np.log(1+z) 

def to_M_source(M, z):
    return M / (1+z)

def pastel_map(cmap, c=0.2, n=6):
    """
    Create a lighter version of a colormap
    Arguments:
    cmap : colormap
    c : scale factor for the lighter colors
    n : number of colors to return
    """
    if type(cmap) == str:
        cmap = plt.get_cmap(cmap)
    colors = ((1. - c) * cmap(np.linspace(0., 1., n)) + c * np.ones((n, 4)))
    return colors
    

if __name__ == '__main__':
    base_name = args['base_name']
    
    datadir = args['datadir']
    if datadir[-1] != '/':
        datadir += '/'
    
    plotdir = args['plotdir']
    if plotdir[-1] != '/':
        plotdir += '/'

    datastring = 'so3-horizon-z.0_-1.pkl'

    savename = plotdir + base_name + '_' + datastring[:-3] + 'pdf'
    
    linestyles = ['-', '--', '-.']

    interp = args['interp']
    if interp:
        interp = interp1d
    fill = args['fill']
    interp_kwargs = dict(fill_value='extrapolate')
    cbar_label = 'Mass ratio'

    with open(datadir + datastring, 'rb') as f:
        data = pkl.load(f)

    M = np.array([el[0] for el in data])
    mu = np.array([el[1] for el in data])
    z = np.array([el[2] for el in data])

    sanity_check = z < 100
    M = M[sanity_check]
    mu = mu[sanity_check]
    z = z[sanity_check]

    q = mu / M
    #make sure there are no mistakenly different values of q because of numerical errors (ie 9.9999999e-6 vs 1.0e-5)
    q = np.round(q, 6)

    q_unique, q_indeces, q_inverse, q_counts = np.unique(q, return_index=True, return_inverse=True, return_counts=True)

    #q_unique = q_unique[q_unique != 1e-3]
    # divide the arrays based on the value of q
    z_data = {}
    M_data = {}
    for i, qval in enumerate(q_unique):
        z_data[qval] = z[q_inverse == i]      
        M_data[qval] = M[q_inverse == i]  

    zmax = max([max(z_data[qval]) for qval in q_unique])
    plot_kwargs = dict(rasterized=True)

    qs_str = [r'$10^{' + str(int(np.log10(qval))) + '}$' for qval in q_unique]
    nlines = len(qs_str)
    cpal = pastel_map('Blues', c=0.2, n=nlines+1)[1:]#[::-1]

    #fig, axs = plt.subplots(ncols=1, nrows=1, sharex=True)
    fig, axs = add_plot(M_data, z_data, frame='source', colors=cpal, fill=fill, interp=interp, interp_kwargs=interp_kwargs, use_gpr=True)#, plot_kwargs=plot_kwargs, fig=fig, axs=axs)

    boundaries = list(q_unique) + [q_unique[-1] * 1.15]
    boundaries_shift = [boundaries[i] - 0.05 for i in range(len(boundaries))]
    norm = mpl.colors.BoundaryNorm(boundaries_shift, nlines)

    cmap = mpl.colors.ListedColormap(cpal[:nlines])
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)

    # Add colorbar axes
    cbar_ax = fig.add_axes([0.92, 0.09, 0.03, 0.8])

    # Create colorbar with explicit tick locations
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical', 
                    extend='both', drawedges=True)

    # Calculate tick positions - center of each color segment
    tick_locs = [(boundaries_shift[i] + boundaries_shift[i+1])/2 
                for i in range(len(boundaries_shift)-1)]

    # Set ticks and labels
    cbar.set_ticks(tick_locs)
    cbar.set_ticklabels(qs_str)
    cbar.ax.minorticks_off()

    # Customize colorbar appearance
    cbar.set_label(cbar_label, fontsize=18, labelpad=22)
    cbar.ax.yaxis.label.set_rotation(270)

    # Set other plot parameters
    # axs.set_title(r'$e_0=0.5, \, a=0.0$', fontsize=18)
    axs.set_xlim(1e4, 2e7)
    axs.set_ylim(0., zmax + 0.2)

    # Save figure
    fig.savefig(savename, dpi=300)
