import os

opj = os.path.join
import numpy as np
import pandas as pd
import glob

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

color_cycle = ['dimgrey', 'firebrick', 'darkorange', 'olivedrab',
               'dodgerblue', 'magenta']
plt.ioff()
plt.rcParams.update({'font.family': 'serif',
                     'font.size': 16, 'axes.labelsize': 20,
                     'mathtext.fontset': 'stix',
                     'axes.prop_cycle': plt.cycler('color', color_cycle)})
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

import pffit

fit = pffit.phase_function_models.inversion()
m = pffit.phase_function_models.models()

dir = pffit.__path__[0]
dirdata = opj(dir, 'data')

trunc = False
if trunc:
    dirfig = opj(dir, 'fig', 'truncated')
    angular_range_t = [3, 150]
else:
    dirfig = opj(dir, 'fig','article' )
    angular_range = [3, 173]

# -------------------
# fitting section
# -------------------

theta_ = np.linspace(0, 180, 1000)
# remove 0deg to comply with FF model (not defined at 0)
theta_ = theta_[1:]
theta_ = np.logspace(-2, np.log10(180), 1000)

# load petzold
file = opj(dirdata, 'petzold_data.txt')
df_petzold = pd.read_csv(file, skiprows=3, sep='\s+', index_col=0, skipinitialspace=True, na_values='inf')

models = (fit.FF_fit, fit.RM_fit, fit.TTFF_fit, fit.TTRM_fit)


def process(wl, data, model, angular_range=[0, 180], x_only=False, theta_=theta_):
    '''
    Execute non-linear fitting for the given model and phase function data
    :param wl:
    :param data:
    :param model:
    :param angular_range:
    :param x_only:
    :param theta_:
    :return:
    '''
    model_ = model.__name__
    N_ang = len(theta_)
    back_ang = theta_[theta_ > 90]
    group_ = data.dropna()
    group_ = group_[
        (group_.index >= angular_range[0]) & (group_.index <= angular_range[1])]  # [group_.index<140]
    theta, vsf = group_.index.values, group_.values

    min1, func = model(theta, vsf)
    out1 = min1.least_squares()  # max_nfev=30, xtol=1e-7, ftol=1e-4)
    x = out1.x

    if x_only:
        return x

    res_ = pd.DataFrame(data={'model': [model_],'sample': [sample], 'name': [names[irow]], 'wavelength': [wl]})

    df_ = pd.DataFrame(data={'model': model_, 'sample': [sample] * N_ang, 'name': [names[irow]] * N_ang,
                             'wavelength': [wl] * N_ang, 'theta': theta_})

    res_['cost'] = out1.residual.__abs__().mean()
    for c in ('redchi', 'bic', 'aic'):
        res_[c] = out1.__getattribute__(c)

    for name, param in out1.params.items():
        res_[name] = param.value
        res_[name + '_std'] = param.stderr
    pf = func(theta_, *x)
    raw = np.interp(theta_, theta, vsf, left=np.nan, right=np.nan)
    df_['pf_raw'] = raw
    df_['pf_fit'] = pf
    norm = np.trapz(func(theta_[1:], *x) * np.sin(np.radians(theta_[1:])), np.radians(theta_[1:])) * np.pi * 2
    bb_tilde = np.trapz(func(back_ang, *x) * np.sin(np.radians(back_ang)),
                        np.radians(back_ang)) * np.pi * 2 / norm
    cos_ave = np.trapz(func(theta_[1:], *x) * np.sin(np.radians(theta_[1:]) * np.cos(np.radians(theta_[1:]))),
                       np.radians(theta_[1:])) * np.pi * 2
    res_['norm'] = norm
    res_['bb_ratio'] = bb_tilde
    res_['asymmetry_factor'] = cos_ave

    return res_, df_


files = glob.glob(opj(dirdata, 'normalized_vsf*txt'))
samples = ['PF_clear', 'PF_coast', 'PF_turbid', 'PF_avg-part',
           'Arizona', 'Chlorella', 'Cylindrotheca', 'Dunaliella', 'Karenia', 'Skeletonema']
names = ['Petzold clear', 'Petzold coast', 'Petzold turbid', 'Petzold average',
         'Arizona dust', r'$\it{C. autotrophica}$', r'$\it{C. closterium}$', r'$\it{D. salina}$',
         r'$\it{K. mikimotoi}$', r'$\it{S. cf. costatum}$']
file_pattern = opj(dirdata, 'normalized_vsf_lov_experiment2015_xxx.txt')
fitdf = []
res = []
for icol, model in enumerate(models):
    model_ = model.__name__

    for irow, sample in enumerate(samples):

        if 'PF' in sample:
            # ===============
            # Petzold data
            # ===============
            print(model_, sample)
            group = df_petzold[sample]
            wl = 514
            # set range to removce extrapolated values and uncertain forward scatt data
            angular_range = [10, 170]
            res_, df_ = process(wl, group, model, angular_range=angular_range)
            res.append(res_)
            fitdf.append(df_)

        else:
            # ===============
            # Harmel et al 2016 data
            # ===============
            file = file_pattern.replace('xxx', sample)
            df = pd.read_csv(file, skiprows=8, sep='\t', index_col=0, skipinitialspace=True, na_values='inf')
            angular_range = [3, 173]
            if trunc:
                angular_range = angular_range_t
            # if trunc: # to truncate phase function and verify consistency over different scatt. angle range
            #     dirfig = opj(dir, 'fig', 'truncated')
            #     angular_range = [3, 150]
            for i, (label, group) in enumerate(df.iteritems()):
                print(model_, sample, label)
                wl = int(label.split('.')[-1])
                res_, df_ = process(wl, group, model, angular_range=angular_range)
                res.append(res_)
                fitdf.append(df_)

res = pd.concat(res)
res.to_csv(opj(dirdata, 'fit_res_all.csv'), index=False)

fitdf = pd.concat(fitdf)
fitdf.to_csv(opj(dirdata, 'fitted_data_all.csv'), index=False)

# -------------------
# plotting section
# -------------------

# ===============
# Performances
# ===============

for param in ('redchi', 'bic', 'aic', 'bb_ratio', 'asymmetry_factor'):
    fig, axs = plt.subplots(2, 2, figsize=(10, 9), sharex=True)
    fig.subplots_adjust(bottom=0.175, top=0.96, left=0.1, right=0.98,
                        hspace=0.25, wspace=0.27)
    axs = axs.ravel()
    for icol, model in enumerate(models):
        model_ = model.__name__
        #res = pd.read_csv(opj(dirdata, 'fit_res_' + model_ + '.csv')).sort_values(['sample', 'wavelength'])
        res_=res[res['model']==model_]
        ax = axs[icol]
        ax.set_title(model_)
        for sample, group in res_.groupby('sample'):
            name = group.name.values[0]
            print(name)
            if 'Petzold' in name:
                continue
            if icol == 3:

                ax.plot(group.wavelength, group[param], label=name, linestyle='dashed', lw=2, marker='o', mec='grey',
                        ms=12, alpha=0.6)
            else:
                ax.plot(group.wavelength, group[param], linestyle='dashed', lw=2, marker='o', mec='grey', ms=12,
                        alpha=0.6)
            if param == "redchi":
                ax.set_ylabel(r'${\chi_\nu^2}$')
            else:
                ax.set_ylabel(param)

    axs[-1].set_xlabel('Wavelength (nm)')
    axs[-2].set_xlabel('Wavelength (nm)')

    fig.legend(loc='upper center', bbox_to_anchor=(0.535, .115),
               fancybox=True, shadow=True, ncol=3, handletextpad=0.5, fontsize=20)
    # fig.tight_layout()
    plt.savefig(opj(dirfig, param + '_fitting_performances.png'), dpi=300)

# ===============
# TTRM parameters
# ===============

fig, axs = plt.subplots(4, 2, figsize=(10, 12), sharex=True)
res_TTRM = res[res['model']=='TTRM_fit']
axs = axs.ravel()
labels = ['$\gamma$', '$g_1$', '$g_2$', r'$\alpha _1$', r'$\alpha_2$', '$\~b_b$', r'$<cos\theta >$']
for i, param in enumerate(['gamma', 'g1', 'g2', 'alpha1', 'alpha2', 'bb_ratio', 'asymmetry_factor']):
    ax = axs[i]
    ax.set_ylabel(labels[i])
    for name, group in res_TTRM.groupby('name'):
        if 'Petzold' in name:
            continue
        # ax.errorbar(group.wavelength,group[param],yerr=group[param+'_std'],label=name,linestyle='dashed',lw=2, marker='o',mec='grey',ms=12,alpha=0.6)
        ax.errorbar(group.wavelength, group[param], linestyle='dashed', lw=2, marker='o', mec='grey', ms=12, alpha=0.6)
for sample, group in res_TTRM.groupby('sample'):
    name = group.name.values[0]
    if 'Petzold' in name:
        continue
    axs[-1].errorbar(group.wavelength, group[param], label=name, linestyle='dashed', lw=2, marker='o', mec='grey',
                     ms=12, alpha=0.6)

axs[-1].set_visible(False)

axs[-2].set_xlabel('Wavelength (nm)')
axs[-3].set_xlabel('Wavelength (nm)')
axs[-3].tick_params(axis='x', labelbottom='on')

fig.legend(loc='lower left', bbox_to_anchor=(0.57, 0.04),
           fancybox=True, shadow=True, ncol=1, handletextpad=0.5, fontsize=17)
plt.tight_layout()
fig.subplots_adjust(hspace=0.065)  # , wspace=0.065)
plt.savefig(opj(dirfig, 'TTRM_fitting_parameters.png'), dpi=300)


# ===============
# PF Fitting per sample
# ===============
def semilog(ax, size=4):
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


color = ['black', 'blue', 'green', 'red']

samples = ['PF',
           'Arizona', 'Chlorella', 'Cylindrotheca', 'Dunaliella', 'Karenia', 'Skeletonema']
for sample in samples:
    print(sample)
    basename = sample
    df = fitdf[fitdf['sample'].str.contains(sample)]

    by_ = 'wavelength'
    title = df.name.values[0]
    if sample == 'PF':
        by_ = 'name'
        title = "Petzold measurements"

    # if not '3µm' in basename:
    #     continue

    fig, axs_ = plt.subplots(2, 2, figsize=(15, 12), sharex=True, sharey=True)
    axslin = [0 for x in range(4)]
    axs = axs_.ravel()
    for im, (model, group) in enumerate(df.groupby('model')):
        print(model)
        ax = axs[im]
        ax.loglog()
        ax, axlin = semilog(ax)
        axslin[im] = axlin

        for i, (label, g_) in enumerate(group.groupby(by_)):
            res_ = res[res['sample'].str.contains(sample) & (res['model'] == model) & (res[by_] == label)]
            if by_ == 'wavelength':
                label = str(label) + ' nm'
            print(label)


            for ax_ in (ax, axlin):
                ax_.plot(g_.theta, g_.pf_raw, color=color[i], label=label)
                ax_.plot(g_.theta, g_.pf_fit, '--', color=color[i])

            bp_tilde = res_.bb_ratio.values[0]
            asym = res_.asymmetry_factor.values[0]

            axlin.text(0.95, 0.95-(i*0.08), r'$\~b_b=${:6.4f}, $<cos \theta > =${:6.3f}'.format(bp_tilde,asym),
                       size=20, color=color[i],transform=axlin.transAxes, ha="right", va="top", )

            ax.set_title(model)

    ax.set_ylim(ymin=0.0003, ymax=30 ** 2)
    plt.legend(loc='upper center', bbox_to_anchor=(-0.5, -0.14),
           fancybox=True, shadow=True, ncol=4, handletextpad=0.5, fontsize=17)

    for irow in range(2):
        axs_[irow, 0].set_ylabel(r'Phase function $(sr^{-1})$')
    for icol in range(-2, 0):
        axslin[icol].xaxis.set_visible(True)
        axslin[icol].set_xlabel('Scattering angle (deg)')
    fig.subplots_adjust(hspace=0.085, wspace=0.085)
    plt.suptitle(title, fontsize=24)
    plt.savefig(opj(dirfig, basename + '.png'), dpi=300)

# ===============
# PF Fitting summary
# ===============
color_cycle = ['white','dimgrey', 'firebrick',  'olivedrab',
               ]
samples = ['PF',
           'Arizona', 'Chlorella', 'Dunaliella']

rows, cols = 4, 4
axslin = [[0 for x in range(cols)] for x in range(rows)]
names=[]
fig, axs = plt.subplots(rows, cols, figsize=(22, 17), sharex=True, sharey=True)

for irow, sample in enumerate(samples):
    print(sample)
    basename = sample
    df = fitdf[fitdf['sample'].str.contains(sample)]

    by_ = 'wavelength'
    name = df.name.values[0]

    if sample == 'PF':
        by_ = 'name'
        name = "Petzold meas."
    names.append(name)

    for im, (model, group) in enumerate(df.groupby('model')):
        print(model)
        axs[0, im].set_title(model)
        ax = axs[irow,im]
        ax.loglog()
        ax, axlin = semilog(ax,size=3.1)
        axslin[irow][im] = axlin

        for i, (label, g_) in enumerate(group.groupby(by_)):
            res_ = res[res['sample'].str.contains(sample) & (res['model'] == model) & (res[by_] == label)]
            if by_ == 'wavelength':
                label = str(label) + ' nm'
            print(label)


            for ax_ in (ax, axlin):
                ax_.plot(g_.theta, g_.pf_raw, color=color[i], label=label)
                ax_.plot(g_.theta, g_.pf_fit, '--', color=color[i])



    ax.set_ylim(ymin=0.0003, ymax=30 ** 2)
    plt.legend(loc='upper right', bbox_to_anchor=(0.975, 0.97),
           fancybox=True, shadow=True, ncol=1, handletextpad=0.5, fontsize=16)
for irow, sample in enumerate(samples):
    axslin[irow][0].text(0.95, 0.95, names[irow], size=20,
                         transform=axslin[irow][0].transAxes, ha="right", va="top",
                         bbox=dict(boxstyle="round",
                                   ec=(0.1, 0.1, 0.1),
                                   fc=plt.matplotlib.colors.to_rgba(color_cycle[irow], 0.3),
                                   ))
    axs[irow, 0].set_ylabel(r'Phase function $(sr^{-1})$')
for icol, model in enumerate(models):
    axslin[-1][icol].xaxis.set_visible(True)
    axslin[-1][icol].set_xlabel('Scattering angle (deg)')
plt.tight_layout()
fig.subplots_adjust(hspace=0.065, wspace=0.065)
plt.suptitle('')
plt.savefig(opj(dirfig, 'Figure_1.png'), dpi=300)


