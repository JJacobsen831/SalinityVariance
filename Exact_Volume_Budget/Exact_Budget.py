# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 15:49:58 2020

@author: Jasen
"""
import os
os.chdir('/home/jjacob2/python/Salt_Budget/SalinityVarianceBudget/Exact_Volume_Budget/')

from netCDF4 import Dataset as nc4
import numpy as np
import Exact_Budget_Terms as ebt
import obs_depth_JJ as dep
import matplotlib.pyplot as plt

#bounds of control volume (points in clockwise order)
#Vertices = np.array([[37.0, -123.0], [37.0, -122.5], [37.2, -122.5], [37.2, -123.0]])
#Vertices = np.array([[35.7, -124.215], [35.7, -124.0], [35.8, -124.0], [35.8, -124.215]])


#load files
FilePath = '/home/cae/runs/jasen/wc15.a01.b03.hourlywindWT.windmcurrent.diags/out/'
HistFile = FilePath + 'ocean_his_2014_0005.nc'
Hist = nc4(HistFile, 'r')
AvgFile = FilePath + 'ocean_avg_2014_0005.nc'
Avg = nc4(AvgFile, 'r')
Diag = nc4(FilePath + 'ocean_dia_2014_0005.nc', 'r')
#GridFile = '/home/ablowe/runs/ncfiles/grids/wc15.a01.b03_grd.nc'

#Carmel Bay
#Vertices = np.array([[36.570, -121.963], [36.570, -121.922], [36.519, -121.922], [36.519, -121.963]])

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
dsdt_dV = np.empty(time.shape)
dsdt_dV.fill(np.nan)
AdvFlux = np.empty(time.shape)
AdvFlux.fill(np.nan)
DifFlux = np.empty(time.shape)
DifFlux.fill(np.nan)
IntMix = np.empty(time.shape)
IntMix.fill(np.nan)

##Loop on time step
for tstep in range(time.shape[0]) :
    #compute variance at time step
    var = Avg.variables['salt'][tstep, :, :, :]
    
    #increase percision for multiplication
    vprime = np.array((var - var[Masks['RhoMask']].mean()), dtype = np.float64)*Masks['LandMask']
    
    #then return to float 32 -> DON'T APPLY MASK, do after calc
    vprime2 = np.array(vprime*vprime, dtype = np.float64)
        
    #compute depth at averg points
    dz = np.diff(dep._set_depth(AvgFile, None, 'w',\
                                Avg.variables['h'][:],\
                                Avg.variables['zeta'][tstep, :, :]), \
                 n = 1, axis = 0)
    
    #compute cell areas
    Areas = ebt.CellAreas(dx, dy, dz, Masks)
        
    #time derivative of salinity variance squared
    dsdt_dV[tstep] = ebt.TimeDeriv(tstep, vprime, Hist, HistFile, Avg, AvgFile, Diag, Areas['Axy'], Masks)
    
    #Advective flux
    AdvFlux[tstep] = ebt.Adv_Flux_west(tstep, vprime2, Avg, Areas, Masks)
    
    #Diffusive Flux
    DifFlux[tstep] = ebt.Diff_Flux_west(Avg, vprime2, dx, Areas, Masks)
    
    #Internal Mixing
    IntMix[tstep] = ebt.Int_Mixing(tstep, vprime, Avg, dx, dy, dz, Masks)

#plotting
Total = dsdt_dV + AdvFlux - DifFlux + IntMix

line0, = plt.plot(dsdt_dV, label = 'd/dt')
line1, = plt.plot(AdvFlux, label = 'Advective Flux')
line2, = plt.plot(DifFlux, label = 'Diffusive Flux')
line3, = plt.plot(IntMix, label = 'Internal Mixing')
line4, = plt.plot(Total, label = 'Sum of Terms')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.title('Monterey Bay Salinity Variance Budget')
#plt.savefig('ExactBudget_Coastal_03Nov2020')