# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 15:49:58 2020

@author: Jasen
"""
import os
os.chdir('/home/jjacob2/python/Salt_Budget/SalinityVarianceBudget/Subroutines/')

from netCDF4 import Dataset as nc4
import ROMS_Tools_Mask as rt
import numpy as np

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
RhoMask = np.repeat(rt.RhoMask(Avg, latbounds, lonbounds)[np.newaxis, :, :], Avg.variables['salt'].shape[1], axis = 0)


#load time for steping
time = Avg.variables['ocean_time'][:]

##FOR Loop on time step
