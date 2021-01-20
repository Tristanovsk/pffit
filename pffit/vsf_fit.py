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
    angular_range = [3, 150]
else:
    dirfig = opj(dir, 'fig', 'test')
    angular_range = [3, 173]

files = glob.glob(opj(dirdata, 'normalized_vsf*txt'))

# -------------------
# fitting section
# -------------------
models = (fit.TTFF_fit, fit.RM_fit, fit.FFRM_fit, fit.TTRM_fit)
samples = ['Arizona', 'Chlorella', 'Cylindrotheca', 'Dunaliella', 'Karenia', 'Skeletonema']
names = ['Arizona dust', r'$\it{C. autotrophica}$', r'$\it{C. closterium}$', r'$\it{D. salina}$',
         r'$\it{K. mikimotoi}$', r'$\it{S. cf. costatum}$']
file_pattern = '/home/harmel/Dropbox/work/git/vrtc/RTxploitation/RTxploitation/../study_cases/vsf/data/normalized_vsf_lov_experiment2015_xxx.txt'

theta_ = np.linspace(0, 180, 100000)
back_ang = theta_[theta_ > 90]


for icol, model in enumerate(models):
    model_ = model.__name__
    res = []
    for irow, sample in enumerate(samples):
        file = file_pattern.replace('xxx', sample)
        df = pd.read_csv(file, skiprows=8, sep='\t', index_col=0, skipinitialspace=True, na_values='inf')

        for i, (label, group) in enumerate(df.iteritems()):
            print(label)
            wl_ = int(label.split('.')[-1])

            group_ = group.dropna()
            group_ = group_[
                (group_.index >= angular_range[0]) & (group_.index <= angular_range[1])]  # [group_.index<140]
            theta, vsf = group_.index.values, group_.values

            min1, func = model(theta, vsf)
            out1 = min1.least_squares()  # max_nfev=30, xtol=1e-7, ftol=1e-4)
            x = out1.x

            res_ = pd.DataFrame(data={'sample': [sample], 'name': [names[irow]], 'wavelength': [wl_]})
            res_['cost'] = out1.residual.__abs__().mean()
            for c in ('redchi', 'bic', 'aic'):
                res_[c] = out1.__getattribute__(c)

            for name, param in out1.params.items():
                res_[name] = param.value
                res_[name + '_std'] = param.stderr

            norm = np.trapz(func(theta_[1:], *x) * np.sin(np.radians(theta_[1:])), np.radians(theta_[1:])) * np.pi * 2
            bb_tilde = np.trapz(func(back_ang, *x) * np.sin(np.radians(back_ang)),
                                np.radians(back_ang)) * np.pi * 2 / norm
            cos_ave = np.trapz(func(theta_[1:], *x) * np.sin(np.radians(theta_[1:]) * np.cos(np.radians(theta_[1:]))),
                               np.radians(theta_[1:])) * np.pi * 2
            res_['norm'] = norm
            res_['bb_ratio'] = bb_tilde
            res_['asymmetry_factor'] = cos_ave
            if model_ == 'TTRM_fit':
                x = out1.x
                #                cov = out1.covar[:3, :3]
                L1 = m.asym_RM(x[1], x[3])
                mu_1 = (2 * x[1] * x[3] * L1 - (1 + x[1] ** 2)) / (2 * x[1] * (x[3] - 1))
                L2 = m.asym_RM(x[2], x[4])
                mu_2 = (2 * x[2] * x[4] * L2 - (1 + x[2] ** 2)) / (2 * x[2] * (x[4] - 1))
                mu_ = x[0] * mu_1 + (1 - x[0]) * mu_2
                print('result: ', x[0], mu_1, mu_2, mu_)
                J = np.array([x[1] - x[2], x[0], 1 - x[0]])
            #                np.matmul(J, np.matmul(cov, J.T))

            res.append(res_)
    res = pd.concat(res)
    res.to_csv(opj(dirdata, 'fit_res_' + model_ + '.csv'))

# -------------------
# plotting section
# -------------------
for param in ('redchi',):  # 'bic','aic','bb_ratio','asymmetry_factor'):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10), sharex=True)
    fig.subplots_adjust(bottom=0.175, top=0.96, left=0.1, right=0.98,
                        hspace=0.25, wspace=0.27)
    axs = axs.ravel()
    for icol, model in enumerate(models):
        model_ = model.__name__
        res = pd.read_csv(opj(dirdata, 'fit_res_' + model_ + '.csv'))

        ax = axs[icol]
        ax.set_title(model_)
        for name, group in res.groupby('name'):
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
# \mathit


fig, axs = plt.subplots(4, 2, figsize=(10, 12), sharex=True)

axs = axs.ravel()
labels = ['$\gamma$', '$g_1$', '$g_2$', r'$\alpha _1$', r'$\alpha_2$', '$\~b_b$', r'$<cos\theta >$']
for i, param in enumerate(['gamma', 'g1', 'g2', 'alpha1', 'alpha2', 'bb_ratio', 'asymmetry_factor']):
    ax = axs[i]
    ax.set_ylabel(labels[i])
    for name, group in res.groupby('name'):
        # ax.errorbar(group.wavelength,group[param],yerr=group[param+'_std'],label=name,linestyle='dashed',lw=2, marker='o',mec='grey',ms=12,alpha=0.6)
        ax.errorbar(group.wavelength, group[param], linestyle='dashed', lw=2, marker='o', mec='grey', ms=12, alpha=0.6)
for name, group in res.groupby('name'):
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

plt.legend()
plt.show()

color = ['black', 'blue', 'green', 'red']

for file in files:
    df = pd.read_csv(file, skiprows=8, sep='\t', index_col=0, skipinitialspace=True, na_values='inf')
    basename = os.path.basename(file).replace('.txt', '')
    if not '3µm' in basename:
        continue
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    axs = axs.ravel()
    for ax, model in zip(axs, models):
        for i, (label, group) in enumerate(df.iteritems()):
            print(label)
            group_ = group.dropna()
            group_ = group_[(group_.index >= angular_range[0]) & (group_.index <= angular_range[1])]
            theta, vsf = group_.index.values, group_.values

            min1, func = model(theta, vsf)
            out1 = min1.least_squares()  # max_nfev=30, xtol=1e-7, ftol=1e-4)
            out1.params.pretty_print()

            x = out1.x
            ax.plot(theta, vsf, color=color[i], label=label)
            ax.plot(theta, func(theta, *x), '--', color=color[i])
            ax.set_xlabel('Scattering angle (deg)')
            ax.set_ylabel('Phase function $(sr^{-1})$')
            ax.semilogy()
            ax.set_title(model.__name__)
    plt.legend()
    plt.suptitle(basename)
    plt.savefig(opj(dirfig, basename + '.png'), dpi=300)

# fig all

samples = ['Arizona', 'Chlorella', 'Cylindrotheca', 'Dunaliella', 'Karenia', 'Skeletonema']
file_pattern = '/home/harmel/Dropbox/work/git/vrtc/RTxploitation/RTxploitation/../study_cases/vsf/data/normalized_vsf_lov_experiment2015_xxx.txt'

rows, cols = 6, 4
axslin = [[0 for x in range(cols)] for x in range(rows)]

irow = 0
fig, axs = plt.subplots(rows, cols, figsize=(22, 25), sharex=True, sharey=True)
for irow, sample in enumerate(samples):
    file = file_pattern.replace('xxx', sample)
    df = pd.read_csv(file, skiprows=8, sep='\t', index_col=0, skipinitialspace=True, na_values='inf')
    for icol, model in enumerate(models):
        ax = axs[irow, icol]
        ax.loglog()

        axs[0, icol].set_title(model.__name__)
        ax.set_xlim((0.01, 10))
        divider = make_axes_locatable(ax)
        axlin = divider.append_axes("right", size=3, pad=0, sharey=ax)
        axslin[irow][icol] = axlin
        ax.spines['right'].set_visible(False)
        axlin.spines['left'].set_linestyle('--')
        # axlin.spines['left'].set_linewidth(1.8)
        axlin.spines['left'].set_color('grey')
        axlin.yaxis.set_ticks_position('right')
        axlin.yaxis.set_visible(False)
        axlin.xaxis.set_visible(False)
        axlin.set_xscale('linear')
        axlin.set_xlim((10, 190))
        for i, (label_, group) in enumerate(df.iteritems()):
            label = label_.split('.')[-1] + ' nm'
            print(label)
            group_ = group.dropna()
            group_ = group_[(group_.index >= angular_range[0]) & (group_.index <= angular_range[1])]
            theta, vsf = group_.index.values, group_.values

            min1, func = model(theta, vsf)
            out1 = min1.least_squares()  # xtol=1e-15,ftol=1e-15)  # max_nfev=30, xtol=1e-7, ftol=1e-4)
            out1.params.pretty_print()

            x = out1.x

            for ax_ in (ax, axlin):
                ax_.plot(theta, vsf, color=color[i], label=label)
                ax_.plot(theta_, func(theta_, *x), '--', color=color[i])

            # norm = (np.trapz(func(theta_[1:], *x) * np.sin(np.radians(theta_[1:])), np.radians(theta_[1:])) * np.pi * 2)
            # bp_tilde = np.trapz(func(back_ang, *x) * np.sin(np.radians(back_ang)), np.radians(back_ang)) * np.pi * 2 / norm
            # axlin.text(0.95, 0.75, '$\~b_b=${:6.4f}'.format(bp_tilde), size=20,
            #            transform=axlin.transAxes, ha="right", va="top", )
        ax.xaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0, numticks=4))
        ax.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0, numticks=10))
        ax.xaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, numticks=10, subs=np.arange(10) * 0.1))
        ax.yaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, numticks=10, subs=np.arange(10) * 0.1))
    irow += 1

    # ax.plot(theta, vsf, color=color[i], label=label)
    # ax.plot(theta_, func(theta_, *x), '--', color=color[i])
    # norm = (np.trapz( func(theta_[1:], *x)*np.sin(np.radians(theta_[1:])),np.radians(theta_[1:]))*np.pi*2)
    # bp_tilde = np.trapz( func(back_ang, *x)*np.sin(np.radians(back_ang)),np.radians(back_ang))*np.pi*2 / norm
    # ax.text(0.95, 0.75, '$\~b_b=${:6.4f}'.format(bp_tilde), size=20,
    #  transform=ax.transAxes, ha="right", va="top",)

plt.legend()
ax.set_ylim(ymin=0.0003, ymax=30 ** 2)
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
plt.savefig(opj(dirfig, 'all_vsf_fit.png'), dpi=300)

# ax.set_xlim(0.01, 200)
# for ax in axs.ravel():
#     ax.semilogx()
# plt.savefig(opj(dirfig, 'all_vsf_fit_loglog.png'), dpi=300)
