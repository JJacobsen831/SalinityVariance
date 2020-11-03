# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 11:08:55 2020

Creates mask defined as vertices of polygon
Control volume on rho points
Mask edges on rho points

@author: Jasen
"""
import numpy as np
from matplotlib import path

def RhoMask(Lats, Lons, Vertices) :
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
    
    #create path based on supplied vertices
    p = path.Path(Vertices)
    
    RhoMask = np.resize(p.contains_points(locs, radius= -1e-2), Lats.shape)
    
    return RhoMask

def FaceMask_shift3D(RhoMask) :
    """
    Shift rho mask to locate north, west, south, east faces of control volume

    Parameters
    ----------
    RhoMask : boolean
        Array that outlines control volume on rho points.

    Returns
    -------
    tuple with NorthFace, WestFace, SouthFace, EastFace.

    """
    #shift mask
    dwn = RhoMask[:, 1:, :]
    up = RhoMask[:, :-1, :]
    left = RhoMask[:, :, 1:]
    right= RhoMask[:, :, :-1]
    
    #south mask
    Smask = np.concatenate((RhoMask[:, 0:1,:], \
                            np.logical_and(dwn == True, up == False)), axis = 1)
    
    #north mask
    Nmask = np.concatenate((np.logical_and(up == True, dwn == False), \
                            RhoMask[:, -2:-1, :]), axis = 1)
    
    #east mask
    Emask = np.concatenate((np.logical_and(right == True, left == False), \
                            RhoMask[:, :, -2:-1]), axis = 2)
    
    #west mask
    Wmask = np.concatenate((RhoMask[:, :,0:1], \
                            np.logical_and(left == True, right == False)), axis = 2)
    
    return Nmask, Wmask, Smask, Emask
    
    
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