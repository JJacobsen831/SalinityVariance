# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 11:38:14 2020
Load files for code testing and development within nested subroutines

@author: jjacob2
"""
import os
os.chdir('/home/jjacob2/python/Salt_Budget/SalinityVarianceBudget/Exact_Volume_Budget/')

from netCDF4 import Dataset as nc4
import numpy as np
import obs_depth_JJ as dep
import GridShift_3D as GridShift
import Manual_Mask as mt
import Differential_Tstep as dff
import Gradients_Tstep as gr

#import numpy.ma as ma
#import Exact_Budget_Terms as ebt

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
GridFile = '/home/ablowe/runs/ncfiles/grids/wc15.a01.b03_grd.nc'

#variable
salt = Avg.variables['salt']