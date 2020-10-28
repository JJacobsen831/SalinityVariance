# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 15:49:58 2020

@author: Jasen
"""
import os
os.chdir('/Users/Jasen/Documents/GitHub/SalinityVarianceBudget/Subroutines/')

from netCDF4 import Dataset as nc4
import numpy as np
import numpy.ma as ma
import Exact_Budget_Terms as ebt

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

#create masks
Masks = ebt.Flux_Masks(AvgFile, Avg, latbounds, lonbounds, precision = 3)

#load time for steping
time = Avg.variables['ocean_time'][:]

#Prealocation 
dsdt_dV = np.empty(time.shape)
dsdt_dV.fill(np.nan)
AdvFlux = np.empty(time.shape)
AdvFlux.fill(np.nan)
DifFlux = np.empty(time.shape)
DifFlux.fill(np.nan)
IntMix = np.empty(time.shape)
DifFlux.fill(np.nan)

tstep = 0
chk = Avg.variables['zeta'][1,:,:]
RomsFile = AvgFile


##Loop on time step
for t in range(time) :
    #compute variance at time step
    var = ma.array(Avg.variables['salt'][tstep, :, :, :], \
                   mask = Masks['RhoMask'])
    v_prime = (var - var.mean())
    v_prime2 = v_prime*v_prime
    
    #compute cell areas
    Areas = ebt.CellAreas(tstep, AvgFile, Avg, Masks)
    
    #time derivative of salinity variance squared
    dsdt_dV[tstep] = ebt.TimeDeriv(tstep, v_prime2, Hist, HistFile, Avg, AvgFile, Diag, Areas['Axy'], Masks)
    
    #Advective flux
    AdvFlux[tstep] = ebt.Adv_Flux(tstep, v_prime2, Avg, Areas, Masks)
    
    #Diffusive Flux
    DifFlux[t] = ebt.Diff_Flux(t, v_prime2, Avg, Masks, Areas)
    
    #Internal Mixing
    IntMix[t] = ebt.Int_Mixing(t, v_prime, Avg, AvgFile, GridFile, Masks)
