# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 14:57:59 2020

@author: Jasen
"""
import os
os.chdir('/Users/Jasen/Documents/GitHub/ROMS_Budget/')
from netCDF4 import Dataset as nc4
import numpy.ma as ma

import obs_depth_JJ as dep


DiaFile = '/Users/Jasen/Documents/Data/ROMS_WCOF/DiagnosticFiles/ocean_dia_jasen.nc'
AvgFile = '/Users/Jasen/Documents/Data/ROMS_WCOF/DiagnosticFiles/ocean_avg_jasen.nc'
GrdFile = '/Users/Jasen/Documents/Data/ROMS_WCOF/DiagnosticFiles/grd_wcofs4_visc200_50-50-50_diff200_spherical.nc'

DiaNC = nc4(DiaFile, 'r')
AvgNC = nc4(AvgFile, 'r')
GrdNC = nc4(GrdFile, 'r')

#variable from average file
salt = AvgNC.variables['salt'][:]

    
#depth at w points
romsvars = {'h' : AvgNC.variables['h'][:], \
                'zeta' : AvgNC.variables['zeta'][:]}

# update to include Vstretching = 4 (spherical?)
depth_w = dep._set_depth_T(AvgFile, None, 'w', romsvars['h'], romsvars['zeta'])
        
delta = ma.diff(depth_w, n = 1, axis = 1)
            



#variables from 