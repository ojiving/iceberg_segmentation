# This script is used to determine the best fit powerlaw distribution to
# each survey and save the best fit parameters. It can easily be adjusted to 
# use other distributions, such as lognormal or exponential.

import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable

import datetime as dt

import glob
import os

from scipy.ndimage.filters import gaussian_filter

import multiprocessing as mp
import timeit
from scipy.stats import kstest as ks
from scipy.optimize import curve_fit

import powerlaw



#%% load and plot files
font = {'family':'Arial', 'weight' :'normal', 'size':9}

matplotlib.rc('font', **font)
matplotlib.rc('text',usetex=False)
matplotlib.rc('lines', linewidth=0.75)

# list of files to process
files = sorted(glob.glob('./iceberg_areas/*npz'))

# set up empty arrays
alpha = np.zeros(len(files))
sigma = np.zeros(len(files))
mu = np.zeros(len(files))
ice_frac = np.zeros(len(files))
frac_year = np.zeros(len(files))
totalArea = np.zeros(len(files))
totalVolume = np.zeros(len(files))

year_previous = 0

# minimum and maximum allowed iceberg areas in the powerlaw fit
xmin = 1
xmax = 50


distribution = 'powerlaw' # type of distribution to fit and plot

for j in np.arange(0,len(files)):
    
    date = dt.datetime.strptime(files[j][-12:-4],'%Y%m%d') # date of survey
    
    day_of_year = date.timetuple().tm_yday
    days_in_year = dt.datetime(date.year,12,31).timetuple().tm_yday
            
    frac_year[j] = date.year + day_of_year/days_in_year
                
    data = np.load(files[j])
    icebergArea = data['icebergArea']
    
    fit = powerlaw.Fit(icebergArea, xmin=xmin, xmax=xmax, fit_method='Likelihood')

    # this is for comparing powerlaw and lognormal fits
    # R = normalized loglikelihood ratio; if R>0 the data is more likely in the first distribution
    # p = significance value for that direction
    # other options to test include 'exponential'
    # it's clear that a power_law fit is better than exponential, indicating that the distribution is heavy-tailed
    # lognormal might be a better fit overall, though power_law is easier to work with
    R, p = fit.distribution_compare('power_law', 'lognormal', normalized_ratio=True)
    
    alpha[j] = fit.power_law.alpha
    xmin  = fit.power_law.xmin
    D = fit.power_law.D
    mu[j] = fit.lognormal.mu
    sigma[j] = fit.lognormal.sigma
    D_ln = fit.lognormal.D
    
    totalArea[j] = np.sum(icebergArea)
    totalVolume[j] = np.sum(4/3*icebergArea**(3/2)) # convert area to volume, assuming spheres
    ice_frac[j] = totalArea[j]/data['surveyArea']
 
    # determine cdf, pdf, and ccdf for plotting purposes
    if distribution == 'lognormal':
        cdf, ind = np.unique(fit.lognormal.cdf(), return_index=True)
        pdf = fit.lognormal.pdf()[ind]
        ccdf = fit.lognormal.ccdf()[ind]
        
    elif distribution == 'power_law':
        cdf, ind = np.unique(fit.power_law.cdf(), return_index=True)
        pdf = fit.power_law.pdf()[ind]
        ccdf = fit.power_law.ccdf()[ind]
    
    plt.figure(1,figsize=(6,8))
    
    #plot cdf
    ax2 = plt.subplot(312)
    x, cdf_data = powerlaw.cdf(icebergArea, xmin=xmin, xmax=xmax)
    x2, _ = fit.cdf()
    ax2.loglog(x,cdf_data,'.')
    ax2.loglog(x2,cdf)
    ax2.set_ylabel(r'P(X$\leq$x)')
    
    #plot pdf
    ax1 = plt.subplot(311)
    bins, pdf_data = fit.pdf()
    ax1.loglog((bins[:-1]+bins[1:])/2,pdf_data,'.')
    ax1.loglog(x2,pdf)
    ax1.set_ylabel('p(x)')
    
    #plot ccdf
    ax3 = plt.subplot(313)
    x, ccdf_data = powerlaw.ccdf(icebergArea, xmin=xmin, xmax=xmax)
    x2, _ = fit.ccdf()
    ax3.loglog(x,ccdf_data,'.')
    ax3.loglog(x2,ccdf)
    ax3.set_ylabel(r'P(X>x)')
    ax3.set_xlabel('Iceberg area [m$^2$]')
    
    ax1.set_xlim([0.1,1000])
    ax2.set_xlim([0.1,1000])
    ax3.set_xlim([0.1,1000])
    ax1.set_ylim([0.0000001,10])
    ax2.set_ylim([0.0000001,1])
    ax3.set_ylim([0.0000001,1])
    
    
    if distribution == 'lognormal':
        ax1.set_title(files[j][-12:-4] + r' | D stat = ' + '{:.4f}'.format(D_ln) + r' | $\mu$ = ' + '{:.2f}'.format(mu[j]) + r' | $\sigma$ = ' '{:.2f}'.format(sigma[j]) )
        plt.tight_layout()
        plt.savefig('./iceberg_areas/' + files[j][-12:-4] + '_lognormal.png', format='png', dpi=150)
    
    elif distribution == 'power_law':
        ax1.set_title(files[j][-12:-4] + r' | D stat = ' + '{:.4f}'.format(D) + r' | $\alpha$ = ' + "{:.2f}".format(alpha[j]) )
        plt.tight_layout()
        plt.savefig('./iceberg_areas/' + files[j][-12:-4] + '_power_law.png', format='png', dpi=150)
        
    plt.close()   
    
    # for each survey, also calculate the empirical ccdf using only an minimum iceberg size and save the results
    x, ccdf_data = powerlaw.ccdf(icebergArea, xmin=xmin)
    np.savez('./iceberg_areas/' + files[j][-12:-4] + '_isd.npz', x=x, ccdf=ccdf_data)
    
np.savez('totalArea_alpha_totalVolume.npz',totalArea=totalArea,alpha=alpha,totalVolume=totalVolume,fracYear=frac_year,ice_frac=ice_frac)















