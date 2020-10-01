# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 11:14:53 2020

@author: Jasen
"""
import os
os.chdir('/Users/Jasen/Documents/GitHub/SalinityVarianceBudget/Subroutines/')

import obs_depth_JJ as dep
#import ROMS_Tools_Mask as rt
import numpy.ma as ma
import numpy as np
from netCDF4 import Dataset as nc4
import Differential as dff

#files
FilePath = '/home/cae/runs/jasen/wc15.a01.b03.hourlywindWT.windcurrent.diags/out/'
HistFile = FilePath + 'ocean_hist_2014_0006.nc'
AvgFile = FilePath + 'ocean_avg_2014_0006.nc'
DiagFile = FilePath + 'ocean_dia_2014_0006.nc'
GrdFile = '/home/ablowe/runs/ncfiles/grids/wc15.a01.b03_grd.nc'

#????
#to conserve memory should I define indices of desired CV instead of loading
#the full field, then applying mask?
#Another option would be to load the full field, apply mask, and discard full field
#issue is that a masked array stores all the data AND the mask, so for 
#large arrays it could take up more memory...
#????

def Prime(var_4dim) :
    """
    Compute anomaly of variable for each time step
    var_4dim = var_4dim[time, depth, lat, lon]
    """
    var_prime = np.empty(var_4dim.shape)
    var_prime.fill(np.nan)
    for n in range(var_4dim.shape[0]) :
        var_prime[n, :, :, :] = var_4dim[n,:,:,:] - ma.mean(var_4dim[n, :, :, :])
    
    return var_prime


#load salt_rate from diagnostic file
Diag = nc4(DiagFile, 'r')
salt_rate = Diag.variables['salt_rate'][:]

#from average file, load salt and vertical thickness
Avg = nc4(AvgFile, 'r')
salt = Avg.variables['salt'][:]

#Compute vertical thickness at w points
depth_domain = dep._set_depth_T(AvgFile, None, 'w', Avg.variables['h'][:],\
                                Avg.variables['zeta'][:])

delta = ma.diff(depth_domain, n = 1, axis = 1)

#from history files compute time derivative of thickness of cell 
Hist = nc4(HistFile, 'r')
depth_hist = dep._set_depth_T(HistFile, None, 'w', Hist.variables['h'][:], \
                              Hist.variables['zeta'][:])

delta_hist = ma.diff(depth_hist, n = 1, axis = 1)

ddelta_dt = ma.diff(delta_hist, \
                     n = 1, axis = 0)/ma.diff(Hist.variables['ocean_time'][:],\
                                    n = 1, axis = 0)

#time deriv of salt, less change in cell volume
ds_dt = salt_rate - salt/delta*ddelta_dt

#multiply by s-prime, dv, and 2
sprime2_dV = 2*Prime(salt)*dff.dV(AvgFile)

#integrate each time step
S_int = np.empty(sprime2_dV.shape[0])
S_int.fill(np.nan)
for n in range(S_int.shape[0]) :
    S_int[n] = np.sum(sprime2_dV[n, :, :, :])

