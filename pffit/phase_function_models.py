import os

opj = os.path.join
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import lmfit as lm

import RTxploitation as rtx

dir = opj(rtx.__path__[0], '..', 'study_cases', 'vsf', )
dirdata = opj(dir, 'data')
dirfig = opj(dir, 'fig')

files = glob.glob(opj(dirdata, 'normalized_vsf*txt'))


class models:
    def __init__(self):
        pass

    # -------------------------------------------
    # single term models
    # -------------------------------------------

    def P_RM(self, theta, g, alpha=0.5):
        '''
        Phase function from Reynold and McCormick (J. Opt. Soc. Am., 1980,
        70, 1206–1212), phase function approximation which described
        highly anisotropic angular scattering distributions
        :param theta: scattering angle (deg)
        :param g: asymmetry factor
        :param alpha: fitting parameter
        :return:
        '''
        theta_rad = np.radians(theta)
        num = alpha * g * (1. - g ** 2) ** (2 * alpha)
        denom = np.pi * (1 + g ** 2 - 2 * g * np.cos(theta_rad)) ** (alpha + 1) * \
                ((1 + g) ** (2 * alpha) - (1 - g) ** (2 * alpha))
        return num / denom

    def P_HG(self, theta, g):
        '''
        When \alpha equals 0.5, P_RM reduces to the Henyey–Greenstein (HG)
        phase function (L. C. Henyey and J. L. Greenstein,
         Astrophys. J., 1941, 93, 70–83.)
        :param theta: scattering angle (deg)
        :param g: asymmetry factor
        :return:
        '''

        return self.P_RM(theta, g, alpha=0.5)

    def P_FF(self, theta, n, m):
        '''
        Forand–Fournier phase function (G. R. Fournier and J. L. Forand,
        Proc. SPIE, 1994, 2258, 194–201.)
        :param theta: scattering angle (deg)
        :param n: relative refractive index (real part), 1 < n < 1.35
        :param m: slope of power-law size distribution, 3.5 < m < 5.
        :return:
        '''

        theta_rad = np.radians(theta)
        u = 2 * np.sin(theta_rad / 2)
        nu = (3 - m) / 2
        delta = u ** 2 / (3 * (n - 1) ** 2)
        delta_pi = 4 / (3 * (n - 1) ** 2)
        delta_nu = delta ** nu
        delta_pi_nu = delta_pi ** nu
        pff = 1 / (4 * np.pi * (1 - delta) ** 2 * delta_nu) * \
              ((nu * (1 - delta) - (1 - delta_nu)) + 4 / u ** 2 * (delta * (1 - delta_nu) - nu * (1 - delta))) \
              - 1 / (16 * np.pi) * (1 - delta_pi_nu) * (3 * np.cos(theta_rad) ** 2 - 1) / ((1 - delta_pi) * delta_pi_nu)
        return pff

    # -------------------------------------------
    # two-term models
    # -------------------------------------------

    def P_TTRM(self, theta, gamma, g1, g2, alpha1=0.5, alpha2=0.5):
        '''
        This function has two parts with two different asymmetry
        factors, where g 1 is positive and g 2 is negative, in order to treat
        the forward and backward peaks in the phase function. The
        parameter γ gives the forward scattering portion while (1 − γ)
        is the backward scattering portion.
        :param theta: scattering angle (deg)
        :param gamma: forward/backward ratio
        :param g1: asymmetry factor >0
        :param g2: asymmetry factor <0
        :param alpha: fitting parameter
        :param alpha2: fitting parameter
        :return:
        '''
        forward = self.P_RM(theta, g1, alpha1)
        backward = self.P_RM(theta, g2, alpha2)
        return gamma * forward + (1 - gamma) * backward

    def P_TTFF(self, theta, gamma, n1, m1, n2, m2):
        '''
        Two-term Fournier-Forand phase function with two particle groups
        defined by their refractive index (real part) and power law slope of size distribution
        :param theta: scattering angle (deg)
        :param gamma: ratio of the first particle group
        :param n1: refractive index of group 1
        :param m1: size distribution slope of group 1
        :param n2: refractive index of group 2
        :param m2: size distribution slope of group 2
        :return:
        '''
        group1 = self.P_FF(theta, n1, m1)
        group2 = self.P_FF(theta, n2, m2)
        return gamma * group1 + (1 - gamma) * group2

    def P_FFRM(self, theta, gamma, n1, m1, g2, alpha2=0.5):
        '''
        This function has two parts forward with Fournier-Forand and
        backward with RM model considering g2 is negative. The
        parameter γ gives the forward scattering portion while (1 − γ)
        is the backward scattering portion.
        :param theta: scattering angle (deg)
        :param gamma: forward/backward ratio
        :param n1: refractive index of group 1
        :param m1: size distribution slope of group 1
        :param g2: asymmetry factor <0
        :param alpha2: fitting parameter
        :return:
        '''
        forward = self.P_FF(theta, n1, m1)
        backward = self.P_RM(theta, g2, alpha2)
        return gamma * forward + (1 - gamma) * backward

    def P_TTFF(self, theta, gamma, n1, m1, n2, m2):
        '''
        Two-term Fournier-Forand phase function with two particle groups
        defined by their refractive index (real part) and power law slope of size distribution
        :param theta: scattering angle (deg)
        :param gamma: ratio of the first particle group
        :param n1: refractive index of group 1
        :param m1: size distribution slope of group 1
        :param n2: refractive index of group 2
        :param m2: size distribution slope of group 2
        :return:
        '''
        group1 = self.P_FF(theta, n1, m1)
        group2 = self.P_FF(theta, n2, m2)
        return gamma * group1 + (1 - gamma) * group2


class inversion:



    def __init__(self):
        self.m = models()

    def RM_fit(self, theta, vsf):
        model = self.m.P_RM

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

    def HG_fit(self, theta, vsf):
        model = self.m.P_RM

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

    def FF_fit(self, theta, vsf):
        model = self.m.P_FF

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

    def TTRM_fit(self, theta, vsf):
        model = self.m.P_TTRM

        def objfunc(x, theta, vsf):
            '''
            Objective function to be minimized
            :param x: vector of unknowns
            :param theta: scattering angle
            :param vsf: phase function
            '''
            gamma, g1, g2, alpha1, alpha2 = np.array(list(x.valuesdict().values()))
            simu = model(theta, gamma, g1, g2, alpha1, alpha2)
            return (np.log(vsf) - np.log(simu)) #/(0.05*np.log(vsf) )

        pars = lm.Parameters()
        pars.add('gamma', value=0.99, min=0.95, max=1)
        pars.add('g1', value=0.9, min=0, max=1)
        pars.add('g2', value=-0.9, min=-.928, max=-0.05)
        pars.add('alpha1', value=0.5, min=0, max=2.5)
        pars.add('alpha2', value=0.5, min=0, max=2.5)
        return lm.Minimizer(objfunc, pars, fcn_args=(theta, vsf)), model

    def TTFF_fit(self, theta, vsf):
        model = self.m.P_TTFF

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

    def FFRM_fit(self, theta, vsf):
        model = self.m.P_FFRM

        def objfunc(x, theta, vsf):
            '''
            Objective function to be minimized
            :param x: vector of unknowns
            :param theta: scattering angle
            :param vsf: phase function
            '''
            gamma, n1, m1, g2, alpha2 = np.array(list(x.valuesdict().values()))
            simu = model(theta, gamma, n1, m1, g2, alpha2)
            return np.log(vsf) - np.log(simu)

        pars = lm.Parameters()
        pars.add('gamma', value=0.7, min=0, max=1)
        pars.add('n1', value=1.05, min=-1, max=1.35)
        pars.add('m1', value=4., min=3.5, max=5)
        pars.add('g2', value=-0.9, min=-.928, max=-0.05)
        pars.add('alpha2', value=0.5, min=0, max=2.5)

        return lm.Minimizer(objfunc, pars, fcn_args=(theta, vsf)), model
