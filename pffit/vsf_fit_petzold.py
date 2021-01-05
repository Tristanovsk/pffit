import os

opj = os.path.join
import numpy as np
import pandas as pd
import glob

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

color_cycle = ['dodgerblue', 'olivedrab', 'darkorange', 'magenta']
plt.ioff()
plt.rcParams.update({'font.family': 'Times New Roman',
                     'font.size': 16, 'axes.labelsize': 20,
                     'mathtext.default': 'regular',
                     'xtick.minor.visible': True,
                     'xtick.major.size': 5,
                     'ytick.minor.visible': True,
                     'ytick.major.size': 5,
                     'axes.prop_cycle': plt.cycler('color', color_cycle)})

import lmfit as lm

import RTxploitation as rtx
import study_cases.vsf as vsf

m = vsf.phase_function_models.models()

dir = opj(rtx.__path__[0], '..', 'study_cases', 'vsf', )
dirdata = opj(dir, 'data')
dirfig = opj(dir, 'fig')


def RM_fit(theta, vsf):
    model = m.P_RM

    def objfunc(x, theta, vsf):
        '''
        Objective function to be minimized
        :param x: vector of unknowns
        :param theta: scattering angle
        :param vsf: phase function
        '''
        g1, alpha1 = np.array(list(x.valuesdict().values()))
        simu = model(theta, g1, alpha1)
        return np.log(vsf) - np.log(simu)

    pars = lm.Parameters()

    pars.add('g1', value=0.9, min=-1, max=1)
    pars.add('alpha1', value=0.5, min=-0.5, max=10)

    return lm.Minimizer(objfunc, pars, fcn_args=(theta, vsf)), model


def HG_fit(theta, vsf):
    model = m.P_RM

    def objfunc(x, theta, vsf):
        '''
        Objective function to be minimized
        :param x: vector of unknowns
        :param theta: scattering angle
        :param vsf: phase function
        '''
        g1 = np.array(list(x.valuesdict().values()))
        simu = model(theta, g1)
        return np.log(vsf) - np.log(simu)

    pars = lm.Parameters()

    pars.add('g1', value=0.9, min=-1, max=1)

    return lm.Minimizer(objfunc, pars, fcn_args=(theta, vsf)), model


def FF_fit(theta, vsf):
    model = m.P_FF

    def objfunc(x, theta, vsf):
        '''
        Objective function to be minimized
        :param x: vector of unknowns
        :param theta: scattering angle
        :param vsf: phase function
        '''
        n1, m1 = np.array(list(x.valuesdict().values()))
        simu = model(theta, n1, m1)
        return np.log(vsf) - np.log(simu)

    pars = lm.Parameters()

    pars.add('n1', value=1.05, min=-1, max=1.35)
    pars.add('m1', value=4., min=3.5, max=5)

    return lm.Minimizer(objfunc, pars, fcn_args=(theta, vsf)), model


def TTRM_fit(theta, vsf):
    model = m.P_TTRM

    def objfunc(x, theta, vsf):
        '''
        Objective function to be minimized
        :param x: vector of unknowns
        :param theta: scattering angle
        :param vsf: phase function
        '''
        gamma, g1, g2, alpha1, alpha2 = np.array(list(x.valuesdict().values()))
        simu = model(theta, gamma, g1, g2, alpha1, alpha2)
        return np.log(vsf) - np.log(simu)

    pars = lm.Parameters()
    pars.add('gamma', value=0.99, min=0., max=1)
    pars.add('g1', value=0.9, min=0, max=1)
    pars.add('g2', value=-0.9, min=-1, max=-0.05)
    pars.add('alpha1', value=0.5, min=0, max=2.5)
    pars.add('alpha2', value=0.5, min=0, max=2.5)
    return lm.Minimizer(objfunc, pars, fcn_args=(theta, vsf)), model


def TTFF_fit(theta, vsf):
    model = m.P_TTFF

    def objfunc(x, theta, vsf):
        '''
        Objective function to be minimized
        :param x: vector of unknowns
        :param theta: scattering angle
        :param vsf: phase function
        '''
        gamma, n1, m1, n2, m2 = np.array(list(x.valuesdict().values()))
        simu = model(theta, gamma, n1, m1, n2, m2)
        return np.log(vsf) - np.log(simu)

    pars = lm.Parameters()
    pars.add('gamma', value=0.7, min=0, max=1)
    pars.add('n1', value=1.05, min=-1, max=1.35)
    pars.add('m1', value=4., min=3.5, max=5)
    pars.add('n2', value=1.15, min=-1, max=1.35)
    pars.add('m2', value=4.5, min=3.5, max=5)

    return lm.Minimizer(objfunc, pars, fcn_args=(theta, vsf)), model


# -------------------
# fitting section
# -------------------
models = (FF_fit, RM_fit, TTFF_fit, TTRM_fit)
file = '/home/harmel/Dropbox/work/git/vrtc/RTxploitation/RTxploitation/../study_cases/vsf/data/petzold_data.txt'
df = pd.read_csv(file, skiprows=3, sep='\s+', index_col=0, skipinitialspace=True, na_values='inf')

angular_range = [10, 170]


def L_RM(g, alpha):
    gp = (1 + g) ** (2 * alpha)
    gm = (1 - g) ** (2 * alpha)
    return (gp + gm) / (gp - gm)


for icol, model in enumerate(models):
    model_ = model.__name__
    res = []

    for i, (label, group) in enumerate(df.iteritems()):
        if not 'PF' in label:
            continue
        print(model_, label)

        group_ = group.dropna()
        group_ = group_[(group_.index >= angular_range[0]) & (group_.index <= angular_range[1])]  # [group_.index<140]
        theta, vsf = group_.index.values, group_.values

        min1, func = model(theta, vsf)
        out1 = min1.least_squares()  # max_nfev=30, xtol=1e-7, ftol=1e-4)

        res_ = pd.DataFrame(data={'sample': [label], 'name': [label], 'wavelength': [514]})
        res_['cost'] = out1.residual.__abs__().mean()
        for c in ('redchi', 'bic', 'aic'):
            res_[c] = out1.__getattribute__(c)

        for name, param in out1.params.items():
            res_[name] = param.value
            res_[name + '_std'] = param.stderr

        if model_ == 'TTRM_fit':
            x = out1.x
            cov = out1.covar[:3, :3]
            L1 = L_RM(x[1], x[3])
            mu_1 = (2 * x[1] * x[3] * L1 - (1 + x[1] ** 2)) / (2 * x[1] * (x[3] - 1))
            L2 = L_RM(x[2], x[4])
            mu_2 = (2 * x[2] * x[4] * L2 - (1 + x[2] ** 2)) / (2 * x[2] * (x[4] - 1))
            mu_ = x[0] * mu_1 + (1 - x[0]) * mu_2
            print('rseult: ', x[0], mu_1, mu_2, mu_)
            J = np.array([x[1] - x[2], x[0], 1 - x[0]])
            np.matmul(J, np.matmul(cov, J.T))

        res.append(res_)
    res = pd.concat(res)
    res.to_csv(opj(dirdata, 'fit_res_petzold_' + model_ + '.csv'))

# -------------------
# plotting section
# -------------------
samples = ['PF_clear', 'PF_coast',
           'PF_turbid', 'PF_avg-part']
names = ['Petzold clear', 'Petzold coast',
         'Petzold turbid', 'Petzold average']
res_ = []
for icol, model in enumerate(models):
    model_ = model.__name__
    res = pd.read_csv(opj(dirdata, 'fit_res_petzold_' + model_ + '.csv'))
    res['model'] = model_.split('_')[0]
    res_.append(res)
res = pd.concat(res_)

for param in ('redchi', 'bic', 'aic'):
    fig, axs = plt.subplots(2, 2, figsize=(10, 7.5), sharex=True)
    fig.subplots_adjust(bottom=0.175, top=0.96, left=0.1, right=0.98,
                        hspace=0.25, wspace=0.27)
    axs = axs.ravel()

    for ip, name in enumerate(names):
        ax = axs[ip]
        ax.set_title(name)
        res_ = res[res.name == samples[ip]]

        markerLines, stemLines, baseLines = ax.stem(res_['model'], res_[param])

        ax.tick_params(axis='x', which='minor', bottom=False)

        if param == "redchi":
            ax.set_ylabel(r'${\chi_\nu^2}$')
        else:
            ax.set_ylabel(param)
        for tick in ax.get_xticklabels():
            tick.set_rotation(55)
        color = plt.matplotlib.colors.to_rgba(color_cycle[ip])
        plt.setp(markerLines, color=color, markersize=12, zorder=1, alpha=0.7,
                 markeredgecolor='grey', markeredgewidth=2)
        plt.setp(stemLines, color=color, linewidth=3, alpha=0.5)
        plt.setp(baseLines, alpha=0)
        # ax.set_ylim([0,0.025])

    plt.tight_layout()
    plt.savefig(opj(dirfig, param + 'petzold_fitting_performances.png'), dpi=300)

fig, axs = plt.subplots(2, 2, figsize=(12, 10))
axs = axs.ravel()
for i, param in enumerate(['gamma', 'g1', 'g2', 'alpha1']):
    ax = axs[i]
    ax.set_title(param)
    for name, group in res.groupby('name'):
        ax.errorbar(group.wavelength, group[param], yerr=group[param + '_std'], label=name, linestyle='dashed', lw=2,
                    marker='o', mec='grey', ms=12, alpha=0.6)
plt.legend()
plt.show()

color = ['black', 'blue', 'green', 'red']
models = (FF_fit, TTFF_fit, RM_fit, TTRM_fit)
for i, (label, group) in enumerate(df.iteritems()):
    if not 'PF' in label:
        continue
    print(model_, label)
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    axs = axs.ravel()
    for ax, model in zip(axs, models):
        group_ = group.dropna()
        group_ = group_[(group_.index >= angular_range[0]) & (group_.index <= angular_range[1])]
        theta, vsf = group_.index.values, group_.values

        min1, func = model(theta, vsf)
        out1 = min1.least_squares()  # max_nfev=30, xtol=1e-7, ftol=1e-4)
        out1.params.pretty_print()

        x = out1.x
        ax.plot(theta, vsf, label=label)
        ax.plot(theta, func(theta, *x), '--')  # , color=color[i])color=color[i],
        ax.set_xlabel('Scattering angle (deg)')
        ax.set_ylabel('Phase function $(sr^{-1})$')
        ax.semilogy()
        ax.set_title(model.__name__)
    plt.legend()
    plt.suptitle(label)
    plt.savefig(opj(dirfig, label + '_pf_fit.png'), dpi=300)

# fig all
theta_ = np.linspace(0, 180, 100000)
back_ang = theta_[theta_ > 90]

models = (FF_fit, RM_fit, TTFF_fit, TTRM_fit)

rows, cols = 4, 4
axslin = [[0 for x in range(rows)] for x in range(cols)]

irow = 0
fig, axs = plt.subplots(rows, cols, figsize=(20, 17), sharex=True, sharey=True)
for i, (label, group) in enumerate(df.iteritems()):
    if not 'PF' in label:
        continue

    print(model_, label)
    for icol, model in enumerate(models):

        ax = axs[irow, icol]
        ax.loglog()

        axs[0, icol].set_title(model.__name__)

        group_ = group.dropna()
        group_ = group_[(group_.index >= angular_range[0]) & (group_.index <= angular_range[1])]
        theta, vsf = group_.index.values, group_.values

        min1, func = model(theta, vsf)
        out1 = min1.least_squares()  # xtol=1e-15,ftol=1e-15)  # max_nfev=30, xtol=1e-7, ftol=1e-4)
        out1.params.pretty_print()

        x = out1.x

        ax.set_xlim((0.01, 10))
        divider = make_axes_locatable(ax)
        axlin = divider.append_axes("right", size=3, pad=0, sharey=ax)
        axslin[irow][icol] = axlin
        axlin.spines['left'].set_visible(False)
        axlin.yaxis.set_ticks_position('right')
        axlin.yaxis.set_visible(False)
        axlin.xaxis.set_visible(False)
        axlin.set_xscale('linear')
        axlin.set_xlim((10, 180))
        for ax_ in (ax, axlin):
            ax_.plot(theta, vsf, label='measured', color='black')
            ax_.plot(theta_, func(theta_, *x), '--', color='red', label='fitted')

        norm = (np.trapz(func(theta_[1:], *x) * np.sin(np.radians(theta_[1:])), np.radians(theta_[1:])) * np.pi * 2)
        bp_tilde = np.trapz(func(back_ang, *x) * np.sin(np.radians(back_ang)), np.radians(back_ang)) * np.pi * 2 / norm
        asym = np.trapz(func(theta_[1:], *x) * np.sin(np.radians(theta_[1:]) * np.cos(np.radians(theta_[1:]))),
                               np.radians(theta_[1:])) * np.pi * 2
        axlin.text(0.95, 0.75, '$\~b_b=${:6.4f}'.format(bp_tilde), size=20,
                   transform=axlin.transAxes, ha="right", va="top", )
        axlin.text(0.95, 0.65, r'$<cos \theta > =${:6.3f}'.format(asym), size=20,
                   transform=axlin.transAxes, ha="right", va="top", )
        ax.xaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0, numticks=4))
        ax.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0, numticks=10))
        ax.xaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, numticks=10,subs=np.arange(10)*0.1))
        ax.yaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, numticks=10,subs=np.arange(10)*0.1))
    irow += 1

plt.legend()
ax.set_ylim(ymin=0.0007, ymax=10 ** 4)

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
plt.savefig(opj(dirfig, 'petzold_pf_fit.png'), dpi=300)
