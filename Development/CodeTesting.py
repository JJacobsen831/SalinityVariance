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
import PolyMask as mt
import Differential_Tstep as dff
import Gradients_Tstep as gr

#import numpy.ma as ma
#import Exact_Budget_Terms

lat = Avg.variables['lat_rho'][:]
lon = Avg.variables['lon_rho'][:]

#bounds of control volume
Vertices = np.array([[37.0, -123.0], [37.0, -122.5], [37.2, -122.5], [37.2, -123.0]])

#load files
FilePath = '/home/cae/runs/jasen/wc15.a01.b03.hourlywindWT.windmcurrent.diags/out/'
AvgFile = FilePath + 'ocean_avg_2014_0005.nc'
RomsNC = nc4(AvgFile, 'r')

HistFile = FilePath + 'ocean_his_2014_0005.nc'
Hist = nc4(HistFile, 'r')
Diag = nc4(FilePath + 'ocean_dia_2014_0005.nc', 'r')
GridFile = '/home/ablowe/runs/ncfiles/grids/wc15.a01.b03_grd.nc'

#variable
salt = Avg.variables['salt'][0,:,:,:]

RomsNC = Avg

RhoMask = Masks['RhoMask']
chk = np.diff(v_prime2, n =1 , axis = 2)
Wpt_variable = dvar_w

#Masks
Masks = CreateMasks(Avg, latbounds, lonbounds)

chk = DifFlux[0]+IntMix[0]+dsdt_dV[0]
dxx = dx[:,1:,:]
chk =dxx[Masks['NFace']]
vprime2 = v_prime2


chkv0 = np.array([[1, 2, 3], [1,2,3], [1,2,3]])
chkv0 = np.repeat(chkv0[np.newaxis, :,:], 1, axis = 0)
chk = GridShift.Rho_to_Upt(chkv0)


def FaceMask(Lats, Lons, Vertices) :
    """
    Create mask on rho points based on vertices

    Parameters
    ----------
    Lats : array
        nlat by nlon array of latitudes
    Lons : array
        nlat by nlon array of longitudes.
    Vertices : array 
        N x 2 (lat, lon) of vertices of control volume.

    Returns 
    -------
    Boolian array outlining control volume on rho points

    """
    #reshape locations to N x 2 (lat x lon)
    locs = np.transpose(np.array([np.resize(Lats, Lats.size), np.resize(Lons, Lons.size)]))
    
    code = [path.Path.MOVETO, path.Path.LINETO, path.Path.LINETO]
    
    Npath = path.Path(np.array((Vertices[3], Vertices[2], Vertices[3])),code, closed = True)
    
    Pmask = np.resize(Npath.contains_points(locs, radius= 1e1), Lats.shape)
    chk = Lats[Pmask]    
    
    return Pmask