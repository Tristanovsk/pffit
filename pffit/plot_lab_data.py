import os
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.ioff()

import pffit

fit = pffit.phase_function_models.inversion()
m = pffit.phase_function_models.models()

opj = os.path.join
dir = pffit.__path__[0]
dirfig = opj(dir, 'fig')

pfs=pd.read_csv(opj(dir,'data/tabulated_phase_function_feb2015_full.txt'),
                skiprows=11,index_col=0,na_values='NA', sep='\s+')
pfs= pfs[pfs.index>0.5]
samples=['Arizona dust', 'C. autotrophica', 'C. closterium',
       'D. salina', 'K. mikimotoi', 'S. cf. costatum']
names=['Arizona dust', r'$\it{C. autotrophica}$', r'$\it{C. closterium}$', r'$\it{D. salina}$',
         r'$\it{K. mikimotoi}$', r'$\it{S. cf. costatum}$']
theta_ = np.logspace(-2, np.log10(180), 1000)

def semilog(ax, size=3):
    ax.loglog()
    ax.set_xlim((0.01, 10))
    divider = make_axes_locatable(ax)
    axlin = divider.append_axes("right", size=size, pad=0, sharey=ax)
    ax.spines['right'].set_visible(False)
    axlin.spines['left'].set_linestyle('--')
    # axlin.spines['left'].set_linewidth(1.8)
    axlin.spines['left'].set_color('grey')
    axlin.yaxis.set_ticks_position('right')
    axlin.yaxis.set_visible(False)
    axlin.xaxis.set_visible(False)
    axlin.set_xscale('linear')
    axlin.set_xlim((10, 190))
    ax.semilogy()
    ax.xaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0, numticks=4))
    ax.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0, numticks=10))
    ax.xaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, numticks=10, subs=np.arange(10) * 0.1))
    ax.yaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, numticks=10, subs=np.arange(10) * 0.1))
    return ax, axlin

rows, cols = 3, 2
for model in (fit.TTRM_fit,fit.FFRM_fit, fit.FF_fit, fit.RM_fit, fit.TTFF_fit ):
    fig, axs_ = plt.subplots(rows, cols, figsize=(cols*5, rows*4), sharex=True, sharey=True)
    axs=axs_.ravel()
    axslin = [0 for x in range(rows*cols)]
    for i,name in enumerate(names):
        pf_ = pfs['pf_'+samples[i]]
        std = pfs['pf_std_'+samples[i]]

        # remove NaN

        pf = pf_[~ pf_.isna()]
        theta = pf.index
        rad = np.radians(theta)
        integ = np.trapz(pf*np.sin(rad),rad)*2*np.pi
        std = std[~ pf_.isna()]/integ
        pf=pf/integ/2


        min1, func = model(theta, pf.values)
        out1 = min1.least_squares()  # max_nfev=30, xtol=1e-7, ftol=1e-4)
        out1.params.pretty_print()

        x = out1.x

        ax, axlin = semilog(axs[i])
        axslin[i] = axlin
        for ax_ in (ax, axlin):
            ax_.plot(theta,pf, color='black')
            ax_.fill_between(theta,pf-std,pf+std)
            ax_.plot(theta_, func(theta_, *x), '--', color='red')

        axs[i].set_ylim(ymin=0.0003, ymax=30 ** 2)
        axs[i].set_title(name)


    for icol in range(-2, 0):
        axslin[icol].xaxis.set_visible(True)
        axslin[icol].set_xlabel('Scattering angle (deg)')
    for irow in range(rows):
        axs_[irow, 0].set_ylabel(r'Phase function $(sr^{-1})$')

    plt.tight_layout()
    fig.subplots_adjust(hspace=0.12, wspace=0.065)
    plt.suptitle('')
    plt.savefig(opj(dirfig, 'pf_allinstru_fit'+model.__name__+'.png'), dpi=300)

    # for i in range(rows*cols):
    #     axs[i].semilogx()
    # plt.savefig(opj(dirfig, 'pf_allinstru_fit'+model.__name__+'_loglog.png'), dpi=300)



#plt.show()
