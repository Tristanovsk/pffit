import os
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
plt.ioff()
import study_cases.vsf as vsf

opj = os.path.join
dirfig = opj(vsf.__path__[0], 'fig')
fit = vsf.phase_function_models.inversion()

pfs=pd.read_csv('./data/tabulated_phase_function_feb2015_full.txt',
                skiprows=11,index_col=0,na_values='NA', sep='\s+')
pfs= pfs[pfs.index>0.5]
names=['Arizona dust', 'C. autotrophica', 'C. closterium',
       'D. salina', 'K. mikimotoi', 'S. cf. costatum']


rows, cols = 3, 2
for model in (fit.TTRM_fit,fit.FFRM_fit, fit.FF_fit, fit.RM_fit, fit.TTFF_fit ):
    fig, axs_ = plt.subplots(rows, cols, figsize=(cols*5, rows*4), sharex=True, sharey=True)
    axs=axs_.ravel()
    for i,name in enumerate(names):
        pf_ = pfs['pf_'+name]
        std = pfs['pf_std_'+name]

        # remove NaN

        pf = pf_[~ pf_.isna()]
        theta = pf.index
        rad = np.radians(theta)
        integ = np.trapz(pf*np.sin(rad),rad)*2*np.pi
        std = std[~ pf_.isna()]/integ
        pf=pf/integ


        min1, func = model(theta, pf.values)
        out1 = min1.least_squares()  # max_nfev=30, xtol=1e-7, ftol=1e-4)
        out1.params.pretty_print()

        x = out1.x

        axs[i].plot(theta,pf, color='black')
        axs[i].fill_between(theta,pf-std,pf+std)
        axs[i].plot(theta, func(theta, *x), '--', color='red')

        axs[i].set_title(name)
        axs[i].semilogy()


    for irow in range(rows):
        axs_[irow, 0].set_ylabel(r'Phase function $(sr^{-1})$')
    for icol in range(cols):
        axs_[-1,icol].set_xlabel('Scattering angle (deg)')
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.12, wspace=0.065)
    plt.suptitle('')
    plt.savefig(opj(dirfig, 'pf_allinstru_fit'+model.__name__+'.png'), dpi=300)
    for i in range(rows*cols):
        axs[i].semilogx()
    plt.savefig(opj(dirfig, 'pf_allinstru_fit'+model.__name__+'_loglog.png'), dpi=300)



#plt.show()
