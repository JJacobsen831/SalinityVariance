# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 13:00:07 2020

Package to make face masks manually with 1s and 0s for a control volume.
Includes masks for rho points, control volume edges on rho points.
For faces on U and V points, a grid shift will be applied

@author: Jasen
"""

import numpy as np
import GridShift_3D as GridShift

def NearestIndex(RomsNC, latbounds, lonbounds) :
    """
    Finds indices of control volume given bounds on lat and lon

    Parameters
    ----------
    RomsNC : TYPE NetCDF file
        NetCDF file that contains ROMS output.
    latbounds : list
        Contains the min and max latitude (deg. N) for the control volume.
    lonbounds : list
        Contains the min and max longitdue (deg. E) for the control volume..

    Returns
    -------
    dictionary of indices of control volume.

    """
    #indices of latbounds
    idx_lat0 = np.argmin(np.abs(RomsNC.variables['lat_rho'][:, 0] - latbounds[0]))
    if idx_lat0.size > 1:
        idx_lat0 = idx_lat0[0]
    
    idx_lat1 = np.argmin(np.abs(RomsNC.variables['lat_rho'][:, 0] - latbounds[1]))
    if idx_lat1.size > 1:
        idx_lat1 = idx_lat1[0]
    
    #indices of lonbounds
    idx_lon0 = np.argmin(np.abs(RomsNC.variables['lon_rho'][0, :] - lonbounds[0]))
    if idx_lon0.size > 1:
        idx_lon0 = idx_lon0[0]
    
    idx_lon1 = np.argmin(np.abs(RomsNC.variables['lon_rho'][0, :] - lonbounds[1]))
    if idx_lon1.size > 1:
        idx_lon1 = idx_lon1[0]
    
    idx = {"lat0" : idx_lat0, "lat1" : idx_lat1, \
           "lon0" : idx_lon0, "lon1" : idx_lon1}
        
    return idx

def RhoMask(RomsNC, RhoIdx) :
    """
    Parameters
    ----------
    RomsNC : TYPE NetCDF file
        NetCDF file that contains ROMS output.
    RhoIdx : TYPE dictionary
        Contains index of vertices of control volume on rho points

    Returns
    -------
    3D array (depth, lat, lon) of ones within control volume and 0 for mask

    """
    #make array of zeros the same size as variable
    A_shape = RomsNC.variables['salt'][:].shape
    Mask_Array = np.zeros((A_shape[1], A_shape[2], A_shape[3]))
    
    Mask_Array[:, RhoIdx['lat0']:RhoIdx['lat1'], RhoIdx['lon0']:RhoIdx['lon1']] = 1
    
    return Mask_Array


def FaceMask(RomsNC, RhoIdx, MaskRho) :
    """
    Parameters
    ----------
    RomsNC : NetCDF4 file
        Contains loaded ROMS output.
    RhoIdx : TYPE dictionary
        Contains index of vertices of control volume on rho points
    MaskRho : TYPE Array
        Array of mask on rho points

    Returns
    -------
    NorthFace, WestFace, SouthFace, EastFace, ON RHO POINTS.

    """
    #empty mask 
    MaskRho[:] = 0
    
    #north mask on rho points
    NorthRho = MaskRho
    NorthRho[:, RhoIdx['lat1'], RhoIdx['lon0']:RhoIdx['lon1']] = 1
    
    #north mask on v points
    NorthV = GridShift.Rho_to_Vpt(NorthRho)
    
    #south mask on rho points
    SouthRho = MaskRho
    SouthRho[:, RhoIdx['lat0'], RhoIdx['lon0']:RhoIdx['lon1']] = 1
    
    #south mask on v points
    SouthV = GridShift.Rho_to_Vpt(SouthRho)
    
    #east mask on rho points
    EastRho = MaskRho
    EastRho[:, RhoIdx['lat0']:RhoIdx['lat1'], RhoIdx['lon1']] = 1
    
    #shift to u points
    EastU = GridShift.Rho_to_Upt(EastRho)
    
    #west mask on rho points
    WestRho = MaskRho
    WestRho[:, RhoIdx['lat0']:RhoIdx['lat1'], RhoIdx['lon0']] = 1
    
    #shift to v points
    WestU = GridShift.Rho_to_Upt(WestRho)
    
    return NorthV, WestU, SouthV, EastU
        