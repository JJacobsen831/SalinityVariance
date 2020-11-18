# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 15:47:33 2020

@author: jjacob2
"""


import os
os.chdir('/home/jjacob2/python/Salt_Budget/SalinityVarianceBudget/Exact_Volume_Budget/')

from netCDF4 import Dataset as nc4
import numpy as np
import Exact_Budget_Salt_Terms as ebt
import obs_depth_JJ as dep
import matplotlib.pyplot as plt


#load files
FilePath = '/home/cae/runs/jasen/wc15.a01.b03.hourlywindWT.windmcurrent.diags/out/'
HistFile = FilePath + 'ocean_his_2014_0005.nc'
Hist = nc4(HistFile, 'r')
AvgFile = FilePath + 'ocean_avg_2014_0005.nc'
Avg = nc4(AvgFile, 'r')
Diag = nc4(FilePath + 'ocean_dia_2014_0005.nc', 'r')
#GridFile = '/home/ablowe/runs/ncfiles/grids/wc15.a01.b03_grd.nc'

#bounds of control volume (points in counter clockwise order)
#north west corner of domain for mask testing
#Vertices = np.array([[35.7, -124.215], [35.7, -124.0], [35.8, -124.0], [35.8, -124.215]])

#shelf & slope offshore of Pt. Ano Nuevo
#Vertices = np.array([[37.0, -123.0], [37.0, -122.5], [37.2, -122.5], [37.2, -123.0]])

#Carmel Bay
#Vertices = np.array([[36.522, -121.948], [36.522, -121.926], [36.567, -121.926], [36.567, -121.948]])

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

#Prealocation 
rate = np.empty(time.shape)
rate.fill(np.nan)
hadv = np.empty(time.shape)
hadv.fill(np.nan)
vadv = np.empty(time.shape)
vadv.fill(np.nan)
dif = np.empty(time.shape)
dif.fill(np.nan)

for tstep in range(time.shape[0]) :
    rate[tstep] = Diag.variables['salt_rate'][tstep,:,:,:][Masks['RhoMask']]
    
    hadv[tstep] = Diag.variables['salt_xadv'][tstep,:,:,:][Masks['RhoMask']] + \
            Diag.variables['salt_yadv'][tstep,:,:,:][Masks['RhoMask']]
    
    vadv[tstep] = Diag.variables['salt_vadv'][tstep,:,:,:][Masks['RhoMask']]
    
    dif[tstep] = Diag.variables['salt_xdiff'][tstep,:,:,:][Masks['RhoMask']] + \
            Diag.variables['salt_ydiff'][tstep,:,:,:][Masks['RhoMask']] + \
            Diag.variables['salt_vdiff'][tstep,:,:,:][Masks['RhoMask']]
            