"""
Standalone script to generate the bound_beta.pdf plot.
Requires: powerlaw_data.h5 in the same directory.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator
import h5py
import os
from math import log10, floor


def format_smart(v):
    """Format value: use scientific notation only for pure powers of 10 >= 100."""
    if v == 0:
        return '0'
    magnitude = floor(log10(abs(v)))
    coeff = v / (10**magnitude)
    if coeff == 1.0 and v >= 100:
        if magnitude == 1:
            return '10'
        else:
            return rf'10^{{{magnitude}}}'
    else:
        if v == int(v):
            return rf'{int(v)}'
        else:
            return rf'{v}'


def format_mass_pair(m1, m2):
    """Format (m1, m2) pair for legend."""
    m1_str = format_smart(m1)
    m2_str = format_smart(m2)
    return rf'${m1_str},\, {m2_str}$'


def get_phi_n(pn_order, m1, m2, M, nu, chi_1, chi_2):
    """Calculate phi_n from PN order and masses."""
    gamma_E = 0.57721566490153286060  # Euler's constant
    delta = (m1 - m2) / M
    chi_S = (chi_1 + chi_2) / 2
    chi_A = (chi_1 - chi_2) / 2
    if int(2*pn_order) == 2:
        return 3715/756 + 55*nu/9
    elif int(2*pn_order) == 3:
        return -16*np.pi + 113*delta*chi_A/3 + (113/3 - 76*nu/3)*chi_S
    elif int(2*pn_order) == 4:
        return 15293365/508032 + 27145*nu/504 + 3085*nu**2/72 + (-405/8 + 200*nu)*chi_A**2 - 405*delta*chi_A*chi_S/4 + (-405/8 + 5*nu/2)*chi_S**2
    elif int(2*pn_order) == 5:
        return 38645*np.pi/756 - 65*np.pi*nu/9 + (-732985/2268 - 140*nu/9)*delta*chi_A + (-732985/2268 + 24260*nu/81 + 340*nu**2/9)*chi_S
    elif int(2*pn_order) == 6:
        return 11583231236531/4694215680 - 6848*np.log(4)/21 - 640*np.pi**2/3 + 6848*gamma_E/21 + (-15737765635/3048192 + 2255*np.pi**2/12)*nu + 76055*nu**2/1728 - 127825*nu**3/1296 + 2270*np.pi*delta*chi_A/3 + (2270*np.pi/3 - 520*np.pi*nu)*chi_S + (75515/288 - 547945*nu/504 - 8455*nu**2/24)*chi_A**2 + (75515/144 - 8225*nu/18)*delta*chi_A*chi_S + (75515/288 - 126935*nu/252 + 19235*nu**2/72)*chi_S**2
    elif int(2*pn_order) == 7:
        return 77096675*np.pi/254016 + 378515*np.pi*nu/1512 - 74045*np.pi*nu**2/756 + (-25150083775/3048192 + 26804935*nu/6048 - 1985*nu**2/48)*delta*chi_A + (-25150083775/3048192 + 10566655595*nu/762048 - 1042165*nu**2/3024 + 5345*nu**3/36)*chi_S
    else:
        return 1.0


def get_beta_dphi_from_B(B, pn_order, M, mu, a):
    """Calculate beta and delta_phi from B coupling."""
    eta = mu * M / (mu + M)**2  # symmetric mass ratio
    beta = -15/32 * 1/(4-pn_order) * 1/(5-2*pn_order) * B * eta**(-2*pn_order/5)
    b = 2*pn_order - 5  # power ppE
    delta_phi = 128/3 * beta * eta**(2*pn_order/5) / get_phi_n(pn_order, M, mu, M+mu, eta, a, 0.0)
    return beta, delta_phi


def main():
    # Load data from HDF5
    h5_file = os.path.join(os.path.dirname(__file__), 'powerlaw_data.h5')
    
    if not os.path.exists(h5_file):
        raise FileNotFoundError(f"Data file not found: {h5_file}")
    
    config_data = {}
    
    with h5py.File(h5_file, 'r') as f:
        for config_name in f.keys():
            grp = f[config_name]
            config = (
                float(grp.attrs['m1']),
                float(grp.attrs['m2']),
                float(grp.attrs['z']),
                float(grp.attrs['T_val'])
            )
            config_data[config] = {
                'nr': np.array(grp['nr']),
                'median_absA': np.array(grp['median_absA']),
                'p16_absA': np.array(grp['p16_absA']),
                'p84_absA': np.array(grp['p84_absA']),
            }
    
    print(f"Loaded {len(config_data)} configurations")
    
    # Build styling information from configurations
    unique_systems = sorted(set((config[0], config[1], config[3]) for config in config_data.keys()))
    
    mass_pair_colors = {}
    for m1, m2, T_val in unique_systems:
        if (m1, m2) not in mass_pair_colors:
            if m1 <= 200000:
                mass_pair_colors[(m1, m2)] = '#ff7f0e'  # orange
            else:
                mass_pair_colors[(m1, m2)] = 'C0'  # blue
    
    styles = {}
    for m1, m2, T_val in unique_systems:
        ls = '-' if T_val == 4.5 else '--'
        alpha = 1.0 if T_val == 4.5 else 0.75
        styles[(m1, m2, T_val)] = {'color': mass_pair_colors[(m1, m2)], 'linestyle': ls, 'alpha': alpha}
    
    # Create legend elements
    legend_elements_emri = []
    for m1, m2, T_val in unique_systems:
        style = styles[(m1, m2, T_val)]
        T_label = rf'{format_mass_pair(m1, m2)}, {T_val}'
        legend_elements_emri.append(
            Line2D([0], [0],
                   label=T_label,
                   linestyle=style['linestyle'],
                   linewidth=1,
                   color=style['color'],
                   alpha=1.0)
        )
    
    # Create the plot
    default_width = 3.25  # in inches
    default_ratio = (np.sqrt(5.0) - 1.0) / 2.0  # golden mean
    
    fig, axs = plt.subplots(1, 1, figsize=(default_width, default_width * default_ratio * 2))
    
    for config, data in sorted(config_data.items()):
        m1, m2, z, T_val = config
        nr_vals = np.array(data['nr'])
        mask = (nr_vals >= -2) & (nr_vals <= 6)
        nr_vals = nr_vals[mask]
        median_absA = np.array(data['median_absA'])[mask] / 10**(nr_vals)
        
        beta_list = np.abs([
            get_beta_dphi_from_B(median_absA[ii], -nr_vals[ii], m1*(1+z), m2*(1+z), 0.99)[0]
            for ii in range(len(nr_vals))
        ])
        
        style = styles[(m1, m2, T_val)]
        arg_sort = np.argsort(nr_vals)
        
        label_ = rf'{format_mass_pair(m1, m2)}, {T_val}'
        axs.semilogy(
            -nr_vals[arg_sort], beta_list[arg_sort], 
            #'o',
            alpha=0.8, ms=5,
            color=style['color'],
            linestyle=style['linestyle'],
            label=label_
        )
    
    # # Add reference points
    # beta_1e6 = np.abs(get_beta_dphi_from_B(4.5e-6 / 10, -1, 1e6, 100, 0.99)[0])
    # beta_1e5 = np.abs(get_beta_dphi_from_B(3.6e-6 / 10, -1, 1e5, 10, 0.99)[0])
    # axs.scatter([-1.15], [beta_1e6], marker='P', color='C0', s=25, zorder=10, alpha=0.7)
    # axs.scatter([-1.15], [beta_1e5], marker='P', color='#ff7f0e', s=25, zorder=10, alpha=0.7)
    
    # Labels and formatting
    axs.set_xlabel(r'PN order', fontsize=15)
    axs.set_ylabel(r'$\beta$', fontsize=15)
    # axs.yaxis.set_major_locator(LogLocator(base=10, numticks=30))
    
    with h5py.File("bounds_pn.h5", 'r') as f:
        nlist = f['nlist'][()]
        GW150914_list = f['GW150914_list'][()]
        SMBH_list = f['SMBH_list'][()]
        EMRI_list = f['EMRI_list'][()]
    
    nlist = [-6, -5.5, -5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 3.5]

    gw150914 = [1.5115799999999997e-11, 7.40908e-11, 3.6529500000000003e-10, 1.81273e-9, 9.0588707266887e-9, 4.5607804951279e-8, 2.3136298509287e-7, 0.0000011823341293477, 0.0000060827263113931, 0.000031477209462677, 0.00016379572246311, 0.00085871257238174, 0.0045667778652951, 0.025027190358874, 0.14582298935005, 0.9643785088657, 8.7331364719302, 223.4104562681, 793.6472199764]

    smbh = [7.348901260895008e-17, 5.903670781505224e-16, 4.780635892725636e-15, 3.912077307551759e-14, 3.246227113918589e-13, 2.7444914572666486e-12, 2.3798570002085215e-11, 2.1367915072166816e-10, 2.0133563608479247e-9, 2.0254466763101922e-8, 2.1950386161665746e-7, 0.000002369645221080775, 0.000019893681185811375, 0.00014220280281751775, 0.0012089464016267263, 0.016322315924577568, 1.6085887048538947, 2.3202340869377203, 6.244360472012985]

    EMRI_10_1e5 = [5.532805023619079e-27, 2.106681498914941e-25, 8.119474529334953e-24, 3.1763611485365388e-22, 1.2658894416541605e-20, 5.1653319516531845e-19, 2.173253099589045e-17, 9.527724680974622e-16, 4.4254823308387826e-14, 2.242352644745364e-12, 1.3170305981124245e-10, 1.0716015884577922e-8, 0.000025958456116095097, 0.00002349319453000577, 0.0006681871937246003, 0.03647668569573771, 8.726242682788738, 959.679894207078, 12113.161149784763]

    EMRI_100_1e6 = [6.6491344350183635e-25, 1.908027791868682e-23, 5.5508220288543975e-22, 1.6421540852835757e-20, 4.960266021456334e-19, 1.5381384495644757e-17, 4.933960828587789e-16, 1.655655225587799e-14, 5.915344743153766e-13, 2.3208823784603186e-11, 1.0670297137234044e-9, 7.003374478879782e-8, 0.00003740590232948982, 0.00008052613444777824, 0.0018229727406370083, 0.0696700685883451, 3.8964616513777326, 623.5460481406232, 9027.750005612766]

    axs.scatter(nlist, EMRI_10_1e5, marker='X', color='C1', s=25, zorder=10, alpha=0.7, label='EMRI')
    axs.scatter(nlist, EMRI_100_1e6, marker='X', color='C0', s=25, zorder=10, alpha=0.7)
    
    label_ = rf'{format_mass_pair(1e5, 10)}, 4.5 (PN)'
    legend_elements_emri.append(Line2D([0], [0], label=label_, marker='X', color='C1', linestyle='', markersize=5, alpha=0.7))
    label_ = rf'{format_mass_pair(1e6, 100)}, 4.5 (PN)'
    legend_elements_emri.append(Line2D([0], [0], label=label_, marker='X', color='C0', linestyle='', markersize=5, alpha=0.7))
    
    # Legend
    leg1 = axs.legend(
        handles=legend_elements_emri,
        loc='upper left', ncols=1,
        bbox_to_anchor=(0.0, 0.99),
        title=r'$m_1[M_\odot], m_2[M_\odot], T[\mathrm{yr}]$',
        frameon=False, framealpha=1.0,
        fontsize=7, title_fontsize=6
    )
    axs.add_artist(leg1)
    
    plt.savefig(os.path.join(os.path.dirname(__file__), 'bound_beta.pdf'), bbox_inches='tight')
    print("Plot saved: bound_beta.pdf")
    plt.close()


if __name__ == '__main__':
    main()
