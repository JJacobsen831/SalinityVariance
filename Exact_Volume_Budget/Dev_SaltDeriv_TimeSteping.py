# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 14:37:16 2020

@author: Jasen
"""

import os
os.chdir('/Users/Jasen/Documents/GitHub/SalinityVarianceBudget/Subroutines/')

import obs_depth_JJ as dep
import ROMS_Tools_Mask as rt
import numpy.ma as ma
import numpy as np
from netCDF4 import Dataset as nc4
import Differential as dff

#bounds of control volume
latbounds = [37.193, 37.378]
lonbounds = [-122.7, -122.456]

#load files
FilePath = '/home/cae/runs/jasen/wc15.a01.b03.hourlywindWT.windmcurrent.diags/out/'
HistFile = FilePath + 'ocean_his_2014_0005.nc'
Hist = nc4(HistFile, 'r')

AvgFile = FilePath + 'ocean_avg_2014_0005.nc'
Avg = nc4(AvgFile, 'r')

Diag = nc4(FilePath + 'ocean_dia_2014_0005.nc', 'r')

Grd = nc4('/home/ablowe/runs/ncfiles/grids/wc15.a01.b03_grd.nc', 'r')

#create CV mask
RhoMask2D = rt.RhoMask(AvgFile, latbounds, lonbounds)
RhoMask = np.repeat(np.array(RhoMask2D)[np.newaxis, :, :], axis = 0)

#load time for steping
time = Avg.variables['ocean_time'][:]

#load dA
dA_xy = ma.array(dff.dA_top(Avg), mask = RhoMask[0,0, :, :])

#integrate time derivative of s_prime^2
SprInt = np.empty(time.shape[0])
SprInt.fill(np.nan)
for t in range(time.shape[0]) :
    #avg file variables at t
    salt = ma.array(Avg.variables['salt'][t, :, :, :], mask = RhoMask)
    s_prime = salt - salt.mean()
    
    #diff and mask are in opposite order    
    deltaA = ma.array(ma.diff(dep._set_depth(AvgFile, None, 'w', \
                                             Avg.variables['h'][:], \
                                             Avg.variables['zeta'][t,:,:]),
                              n = 1, axis = 1), \
                      mask = RhoMask)
    
    #diag file variables at t
    salt_rate = ma.array(Diag.variables['salt_rate'][t, :, :, :], \
                         mask = RhoMask)
    
    #hist variables at t and t+1
    deltaH0 = ma.array(ma.diff(dep._set_depth(HistFile, None, 'w', \
                                             Hist.variables['h'][:], \
                                             Hist.variables['zeta'][t,:,:]),
                              n = 1, axis = 1), \
                      mask = RhoMask)
    
    deltaH1 = ma.array(ma.diff(dep._set_depth(HistFile, None, 'w', \
                                             Hist.variables['h'][:], \
                                             Hist.variables['zeta'][t+1,:,:]),
                              n = 1, axis = 1), \
                      mask = RhoMask)
    
    dtH = Hist.variables['ocean_time'][t+1] - Hist.variables['ocean_time'][t]
    
    #time derivative of cell volume on avg points
    dDelta_dt = (deltaH1 - deltaH0)/dtH
    
    #compute volume
    dV = dA_xy*deltaA
    
    #integrate
    SprInt[t] = ma.sum(2*s_prime*(salt_rate - salt/deltaA*dDelta_dt)*dV)

    
    
    
    
    
    
    
