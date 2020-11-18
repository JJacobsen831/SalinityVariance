# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 09:49:55 2020

@author: jjacob2
"""

import os
os.chdir('/home/jjacob2/python/Salt_Budget/SalinityVarianceBudget/Exact_Volume_Budget/')

from netCDF4 import Dataset as nc4
import numpy as np
import Exact_Budget_Terms as ebt
import obs_depth_JJ as dep
import matplotlib.pyplot as plt


#load files
FilePath = '/home/cae/runs/jasen/wc15.a01.b03.hourlywindWT.windmcurrent.diags/out/'
HistFile = FilePath + 'ocean_his_2014_0005.nc'
Hist = nc4(HistFile, 'r')
AvgFile = FilePath + 'ocean_avg_2014_0005.nc'
Avg = nc4(AvgFile, 'r')
Diag = nc4(FilePath + 'ocean_dia_2014_0005.nc', 'r')

#Monterey Bay
Vertices = np.array([[36.585, -121.976], [36.585, -121.784], [36.981, -121.784], [36.981, -121.976]])

#create masks
Masks = ebt.CreateMasks(Avg, Vertices)

#load time for steping
time = Avg.variables['ocean_time'][:]

#load dx and dy for gradients
dx = np.repeat(1/np.array(Avg.variables['pm'][:])[np.newaxis, :, :], \
               Avg.variables['salt'][0,:,0,0].size, axis = 0)
dy = np.repeat(1/np.array(Avg.variables['pn'][:])[np.newaxis, :, :], \
               Avg.variables['salt'][0,:,0,0].size, axis = 0)

EstEr = np.empty((dy[Masks['RhoMask']].size,time.size))
EstEr.fill(np.nan)


for tstep in range((time.shape[0]) -1):
    #compute variance at time step
    var = Avg.variables['salt'][tstep, :, :, :]
    
    #compute depth at averg points
    dz = np.diff(dep._set_depth(AvgFile, None, 'w',\
                                Avg.variables['h'][:],\
                                Avg.variables['zeta'][tstep, :, :]), \
                 n = 1, axis = 0)
    
    #Time derivative
    #cell thinkness on average points
    deltaA = np.diff(dep._set_depth(AvgFile, None, 'w', \
                                             Avg.variables['h'][:], \
                                             Avg.variables['zeta'][tstep,:,:]),
                              n = 1, axis = 0)
    
    #diagnostic rate
    var_rate = Diag.variables['salt_rate'][tstep, :, :, :]*Masks['RhoMask']
    
    #change in vertical cell thickness at history points
    deltaH0 = np.diff(dep._set_depth(HistFile, None, 'w', \
                                             Hist.variables['h'][:], \
                                             Hist.variables['zeta'][tstep,:,:]),
                              n = 1, axis = 0)*Masks['RhoMask']
    
    deltaH1 = np.diff(dep._set_depth(HistFile, None, 'w', \
                                             Hist.variables['h'][:], \
                                             Hist.variables['zeta'][tstep+1,:,:]),
                              n = 1, axis = 0)*Masks['RhoMask']
    
    dtH = Hist.variables['ocean_time'][tstep+1] - Hist.variables['ocean_time'][tstep]
    
    #time derivative of cell volume on avg points
    dDelta_dt = (deltaH1 - deltaH0)/dtH
    
    s = var*Masks['RhoMask']
    
    chk = (s/deltaA)*dDelta_dt
        
    dsdt = (Avg.variables['salt'][tstep+1,:,:,:]-Avg.variables['salt'][tstep,:,:,:])/dtH
        
    EstEr[:,tstep] =dsdt[Masks['RhoMask']] - var_rate[Masks['RhoMask']] + chk[Masks['RhoMask']]

#ploting
plt.contourf(EstEr)
plt.colorbar()

#checking salt_rate term
srError = np.empty((dy[Masks['RhoMask']].size,time.size))
srError.fill(np.nan)
for tstep in range((time.shape[0]) -1):
    rate = Diag.variables['salt_rate'][tstep,:,:,:][Masks['RhoMask']]
    xadv = Diag.variables['salt_xadv'][tstep, :,:,:][Masks['RhoMask']]
    yadv = Diag.variables['salt_yadv'][tstep, :,:,:][Masks['RhoMask']]
    xdiff = Diag.variables['salt_xdiff'][tstep, :,:,:][Masks['RhoMask']]
    ydiff = Diag.variables['salt_ydiff'][tstep, :,:,:][Masks['RhoMask']]
    
    srError[:,tstep] = rate - xadv - yadv - xdiff - ydiff

plt.contourf(srError)
plt.colorbar()