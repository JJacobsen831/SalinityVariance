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
latbounds = [35, 37]
lonbounds = [-126, -125]

#load files
FilePath = '/home/cae/runs/jasen/wc15.a01.b03.hourlywindWT.windcurrent.diags/out/'
HistFile = FilePath + 'ocean_his_2014_0006.nc'
Hist = nc4(HistFile, 'r')

AvgFile = FilePath + 'ocean_avg_2014_0006.nc'
Avg = nc4(AvgFile, 'r')

Diag = nc4(FilePath + 'ocean_dia_2014_0006.nc', 'r')
Grd = nc4('/home/ablowe/runs/ncfiles/grids/wc15.a01.b03_grd.nc', 'r')

#load time for steping
time = Avg.variables['ocean_time'][:]

#create CV mask
RhoMask = rt.RhoMask(AvgFile, latbounds, lonbounds)

for t in range(time.shape[0]) :
    #avg file variables at t
    salt = ma.array(Avg.variables['salt'][t, :, :, :], mask = RhoMask)
    
    s_prime = salt - salt.mean()
    
    deltaA = ma.diff(ma.array(dep._set_depth(AvgFile, None, 'w', \
                                             Avg.variables['h'][:], \
                                             Avg.variables['zeta'][t, :, :]), \
                              mask = RhoMask),\
                     n = 1, axis = 1)
    
    #diag file variables at t
    salt_rate = ma.array(Diag.variables['salt_rate'][t, :, :, :], \
                         mask = RhoMask)
    
    #hist variables at t and t+1
    deltaH0 = ma.diff(ma.array(dep._set_depth(HistFile, None, 'w', \
                                              Hist.variables['h'][:], \
                                              Hist.variables['zeta'][t, :, :]),\
                               mask = RhoMask), \
                      n = 1, axis = 1)
    
    deltaH1 = ma.diff(ma.array(dep._set_depth(HistFile, None, 'w', \
                                               Hist.variables['h'][:], \
                                               Hist.variables['zeta'][t+1, :, :]),\
                                mask = RhoMask),\
                      n = 1, axis = 1)
    
    dtH = Hist.variables['ocean_time'][t+1] - Hist.variables['ocean_time'][t]
    
    #time derivative of cell volume on avg points
    dDelta_dt = (deltaH1 - deltaH0)/dtH
    
    
    
    
    
    
    
