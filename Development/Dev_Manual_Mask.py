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

def Nearest(RomsNC, latbounds, lonbounds) :
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

def RhoMask(RomsNC, latbounds, lonbounds) :
    """
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
    3D array (depth, lat, lon) of 1s within control volume and 0 for mask

    """
    #indices of control volume
    idx = Nearest(RomsNC, latbounds, lonbounds)
    
    #make array of zeros the same size as variable
    A_shape = RomsNC.variables['salt'][:].shape
    Mask_Array = np.zeros((A_shape[1], A_shape[2], A_shape[3]))
    
    Mask_Array[:, idx['lat0']:idx['lat1'], idx['lon0']:idx['lon1']] = 1
    
    return Mask_Array

def UMask(RomsNC, latbounds, lonbounds):
    """
    Parameters
    ----------
    RomsNC : NetCDF4 file
        Contains loaded ROMS output.
    latbounds : list
        Contains the min and max latitude (deg. N) for the control volume.
    lonbounds : list
        Contains the min and max longitdue (deg. E) for the control volume.

    Returns
    -------
    Mask of 0s and 1s where control volume is True. Shifted to U points

    """
    R_mask = RhoMask(RomsNC, latbounds, lonbounds)
    
    U_mask = GridShift.Rho_to_Upt(R_mask)
    
    return U_mask

def VMask(RomsNC, latbounds, lonbounds):
    """
    Parameters
    ----------
    RomsNC : NetCDF4 file
        Contains loaded ROMS output.
    latbounds : list
        Contains the min and max latitude (deg. N) for the control volume.
    lonbounds : list
        Contains the min and max longitdue (deg. E) for the control volume.

    Returns
    -------
    Mask of 0s and 1s where control volume is True. Shifted to V points

    """
    R_mask = RhoMask(RomsNC, latbounds, lonbounds)
    
    V_mask = GridShift.Rho_to_Vpt(R_mask)
    
    return V_mask

def FaceMask(RomsNC, latbounds, lonbounds):
    """
    Parameters
    ----------
    RomsNC : NetCDF4 file
        Contains loaded ROMS output.
    latbounds : list
        Contains the min and max latitude (deg. N) for the control volume.
    lonbounds : list
        Contains the min and max longitdue (deg. E) for the control volume.

    Returns
    -------
    NorthFace, WestFace, SouthFace, EastFace, ON RHO POINTS.

    """
    #indices
    idx = Nearest(RomsNC, latbounds, lonbounds)
    
    #empty mask 
    A_shape = RomsNC.variables['salt'][:].shape
    MaskRho = np.zeros((A_shape[1], A_shape[2], A_shape[3]))
    
    #north mask on rho points
    NorthRho = MaskRho
    NorthRho[:, idx['lat1'], idx['lon0']:idx['lon1']] = 1
    
    #north mask on v points
    NorthV = GridShift.Rho_to_Vpt(NorthRho)
    
    #south mask on rho points
    SouthRho = MaskRho
    SouthRho[:, idx['lat0'], idx['lon0']:idx['lon1']] = 1
    
    #south mask on v points
    SouthV = GridShift.Rho_to_Vpt(SouthRho)
    
    #east mask on rho points
    EastRho = MaskRho
    EastRho[:, idx['lat0']:idx['lat1'], idx['lon1']] = 1
    
    #shift to u points
    EastU = GridShift.Rho_to_Upt(EastRho)
    
    #west mask on rho points
    WestRho = MaskRho
    WestRho[:, idx['lat0']:idx['lat1'], idx['lon0']] = 1
    
    #shift to v points
    WestU = GridShift.Rho_to_Upt(WestRho)
    
    return NorthV, WestU, SouthV, EastU
        