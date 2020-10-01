# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 15:31:03 2020
Masking tools and generic tools to help with processing ROMS files
@author: Jasen
"""
import numpy as np
from netCDF4 import Dataset as nc4


def RhoUV_Mask(RomsFile, latbounds, lonbounds) :
    """
    defines masks on rho, u, and v points
    """
    RomsNC = nc4(RomsFile, 'r')
        
    #define rho mask
    Rholats = (RomsNC.variables['lat_rho'][:] >= latbounds[0])*(RomsNC.variables['lat_rho'][:] <= latbounds[1])
    Rholons = (RomsNC.variables['lon_rho'][:] >= lonbounds[0])*(RomsNC.variables['lon_rho'][:] <= lonbounds[1])
    Rho_Mask = np.ma.asarray(AddDepthTime(RomsFile, np.invert(Rholats*Rholons)))
    
    #offsets for c - grid
    c_offset_v = np.round(RomsNC.variables['lat_v'][0, 0] - RomsNC.variables['lat_rho'][0,0], decimals = 2)
    c_offset_u = np.round(RomsNC.variables['lon_u'][0,0] - RomsNC.variables['lon_rho'][0,0], decimals = 2)
    
    #define u mask
    Ulats = (RomsNC.variables['lat_u'][:] >= latbounds[0])*(RomsNC.variables['lat_u'][:] <= latbounds[1])
    Ulons = (RomsNC.variables['lon_u'][:] >= lonbounds[0]-c_offset_u)*(RomsNC.variables['lon_u'][:] <= lonbounds[1]-c_offset_u)
    U_Mask = np.ma.asarray(AddDepthTime(RomsFile, np.invert(Ulats*Ulons)))
    
    #define v mask
    Vlats = (RomsNC.variables['lat_v'][:] >= latbounds[0]+c_offset_v)*(RomsNC.variables['lat_v'][:] <= latbounds[1]+c_offset_v)
    Vlons = Rholons = (RomsNC.variables['lon_v'][:] >= lonbounds[0])*(RomsNC.variables['lon_v'][:] <= lonbounds[1])
    V_Mask = np.ma.asarray(AddDepthTime(RomsFile, np.invert(Vlats*Vlons)))
    
    return Rho_Mask, U_Mask, V_Mask

def FaceMask(RomsFile, latbounds, lonbounds) :
    """
    Define masks along each face of control volume
    Returns dictionary for each face with masks on rho and u or v points
    """
    RomsNC = nc4(RomsFile, 'r')
        
    #offsets for c - grid
    c_offset_lat = np.round(RomsNC.variables['lat_v'][0, 0] - RomsNC.variables['lat_rho'][0,0], decimals = 2)
    c_offset_lon = np.round(RomsNC.variables['lon_u'][0,0] - RomsNC.variables['lon_rho'][0,0], decimals = 2)
    
    #north face v points with c-grid offset
    Vlats = (RomsNC.variables['lat_v'][:] == latbounds[0] + c_offset_lat)*(RomsNC.variables['lat_v'][:] <= latbounds[1] + c_offset_lat)
    Vlons = (RomsNC.variables['lon_v'][:] >= lonbounds[0])*(RomsNC.variables['lon_v'][:] <= lonbounds[1])
    _NFv = np.invert(Vlats*Vlons)
    NorthFace = np.ma.asarray(AddDepthTime(RomsFile, _NFv))
    
    #south face v points
    Vlats = (RomsNC.variables['lat_v'][:] >= latbounds[0]+c_offset_lat)*(RomsNC.variables['lat_v'][:] == latbounds[1]+c_offset_lat)
    Vlons = (RomsNC.variables['lon_v'][:] >= lonbounds[0])*(RomsNC.variables['lon_v'][:] <= lonbounds[1])
    _SFv = np.invert(Vlats*Vlons)
    SouthFace = np.ma.asarray(AddDepthTime(RomsFile, _SFv))
    
    
    #West face u points
    Ulats = (RomsNC.variables['lat_u'][:] >= latbounds[0])*(RomsNC.variables['lat_u'][:] <= latbounds[1])
    Ulons = (RomsNC.variables['lon_u'][:] == lonbounds[0]-c_offset_lon)*(RomsNC.variables['lon_u'][:] <= lonbounds[1]-c_offset_lon)
    _WFu = np.invert(Ulats*Ulons)
    WestFace = np.ma.asarray(AddDepthTime(RomsFile, _WFu))
        
    #East face u points
    Ulats = (RomsNC.variables['lat_u'][:] >= latbounds[0])*(RomsNC.variables['lat_u'][:] <= latbounds[1])
    Ulons = (RomsNC.variables['lon_u'][:] >= lonbounds[0]-c_offset_lon)*(RomsNC.variables['lon_u'][:] == lonbounds[1]-c_offset_lon)
    _EFu = np.invert(Ulats*Ulons)
    EastFace = np.ma.asarray(AddDepthTime(RomsFile, _EFu))
    
    return NorthFace, WestFace, SouthFace, EastFace

def ROMS_CV(varname, RomsFile, Mask):
    """
    Load variable and apply mask
    """
    RomsNC = nc4(RomsFile, 'r')
    
    Var = np.ma.array(RomsNC.variables[varname][:], mask = Mask)
    
    return Var
    
def AddDepthTime(RomsFile, var2D) :
    """
    Add depth and time dimensions for lat x lon variables
    """
    RomsNC = nc4(RomsFile, 'r')
    var = RomsNC.variables['salt'][:]
    
    #repeat over depth space
    _DepthSpace = np.repeat(np.array(var2D)[np.newaxis, :, :], \
                            var.shape[1], axis = 0)
    
    #repeat over time space
    TimeDepth = np.repeat(np.array(_DepthSpace)[np.newaxis, :, :, :],\
                           var.shape[0], axis = 0)
    return TimeDepth

def rho_dist_grd(RomsGrd) :
    """ 
    Compute distance between rho points using roms grid file
    Output on 
    """
    RomsNC = nc4(RomsGrd, 'r')
    
    #distance from grid edge
    Xx = RomsNC.variables['x_rho'][:]
    Yy = RomsNC.variables['y_rho'][:]
    
    #difference between points centered on cell edge
    dx_cell = np.diff(Xx, n = 1, axis = 1)
    dy_cell = np.diff(Yy, n = 1, axis = 0)
    
    #pad ends
    dx_pad = np.concatenate((dx_cell[:, 0:1], dx_cell, dx_cell[:, -2:-1]), axis = 1)
    dy_pad = np.concatenate((dy_cell[0:1, :], dy_cell, dy_cell[-2:-1, :]), axis = 0)
        
    #average to cell center
    dx = 0.5*(dx_pad[:, 0:dx_pad.shape[1]-1] + dx_pad[:, 1:dx_pad.shape[1]])
    dy = 0.5*(dy_pad[0:dy_pad.shape[0]-1, :] + dy_pad[1:dy_pad.shape[0]])
    
    return dx, dy

def cell_width(RomsGrd) :
    """
    compute width of cell grids
    """
    #dx, dy on rho points
    dx_rho, dy_rho = rho_dist_grd(RomsGrd)
    
    #average dx, dy at rho point to cell edges
    dy_cell = 0.5*(dy_rho[:, 0:dy_rho.shape[1]-1] + dy_rho[:, 1:dy_rho.shape[1]])
    dx_cell = 0.5*(dx_rho[0:dx_rho.shape[0]-1, :] + dx_rho[1:dx_rho.shape[0]])
    
    return dx_cell, dy_cell
  
