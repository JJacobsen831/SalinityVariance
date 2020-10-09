# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 10:36:21 2020

Package to make masks for control volume
Rho mask, face masks
All putput is 2D: lat, lon

@author: Jasen
"""
import numpy as np

def RhoMask(RomsNC, latbounds, lonbounds) :
    """
    Defines mask on rho points
    """
    Rholats = (RomsNC.variables['lat_rho'][:] >= latbounds[0])*(RomsNC.variables['lat_rho'][:] <= latbounds[1])
    Rholons = (RomsNC.variables['lon_rho'][:] >= lonbounds[0])*(RomsNC.variables['lon_rho'][:] <= lonbounds[1])
    Rho_Mask = np.ma.asarray(np.invert(Rholats*Rholons))
    
    return Rho_Mask

def UMask(RomsNC, latbounds, lonbounds, precision) :
    """
    #Defines mask on U points
    #precision = number of decimals of longitude points
    """
    #offset between rho and u points
    c_offset_u = np.round(RomsNC.variables['lon_u'][0,0] - RomsNC.variables['lon_rho'][0,0], decimals = precision)
    
    Ulats = (RomsNC.variables['lat_u'][:] >= latbounds[0])*(RomsNC.variables['lat_u'][:] <= latbounds[1])
    Ulons = (RomsNC.variables['lon_u'][:] >= lonbounds[0]-c_offset_u)*(RomsNC.variables['lon_u'][:] <= lonbounds[1]-c_offset_u)
    U_Mask = np.ma.asarray(np.invert(Ulats*Ulons))
    
    return U_Mask

def VMask(RomsNC, latbounds, lonbounds, precision) :
    
    #Defines mask on V points
    #precision = number of decimals of latitude points
    
    #offset between rho and v points
    c_offset_v = np.round(RomsNC.variables['lat_v'][0, 0] - RomsNC.variables['lat_rho'][0,0], decimals = precision)
    
    Vlats = (RomsNC.variables['lat_v'][:] >= latbounds[0]+c_offset_v)*(RomsNC.variables['lat_v'][:] <= latbounds[1]+c_offset_v)
    Vlons = (RomsNC.variables['lon_v'][:] >= lonbounds[0])*(RomsNC.variables['lon_v'][:] <= lonbounds[1])
    V_Mask = np.ma.asarray(np.invert(Vlats*Vlons))
    
    return V_Mask
    

def FaceMask(RomsNC, latbounds, lonbounds, precision) :
    """
    Defines mask on north, west, south, and east boundaries. On u or v points.
    precision = number of decimals in lat and lon points
    """
    #offsets for c - grid
    c_offset_lat = np.round(RomsNC.variables['lat_v'][0, 0] - RomsNC.variables['lat_rho'][0,0], decimals = precision)
    c_offset_lon = np.round(RomsNC.variables['lon_u'][0,0] - RomsNC.variables['lon_rho'][0,0], decimals = precision)
    
    #north face v points with c-grid offset
    Vlats = (RomsNC.variables['lat_v'][:] == latbounds[0] + c_offset_lat)*(RomsNC.variables['lat_v'][:] <= latbounds[1] + c_offset_lat)
    Vlons = (RomsNC.variables['lon_v'][:] >= lonbounds[0])*(RomsNC.variables['lon_v'][:] <= lonbounds[1])
    NorthFace = np.ma.asarray(np.invert(Vlats*Vlons))
    
    #south face v points with c-grid offset
    Vlats = (RomsNC.variables['lat_v'][:] >= latbounds[0]+c_offset_lat)*(RomsNC.variables['lat_v'][:] == latbounds[1]+c_offset_lat)
    Vlons = (RomsNC.variables['lon_v'][:] >= lonbounds[0])*(RomsNC.variables['lon_v'][:] <= lonbounds[1])
    SouthFace = np.ma.asarray(np.invert(Vlats*Vlons))
    
    #west face on u points with c-grid offset
    Ulats = (RomsNC.variables['lat_u'][:] >= latbounds[0])*(RomsNC.variables['lat_u'][:] <= latbounds[1])
    Ulons = (RomsNC.variables['lon_u'][:] == lonbounds[0]-c_offset_lon)*(RomsNC.variables['lon_u'][:] <= lonbounds[1]-c_offset_lon)
    WestFace = np.ma.asarray(np.invert(Ulats*Ulons))
    
    #east face on u points with c-grid offset
    Ulats = (RomsNC.variables['lat_u'][:] >= latbounds[0])*(RomsNC.variables['lat_u'][:] <= latbounds[1])
    Ulons = (RomsNC.variables['lon_u'][:] >= lonbounds[0]-c_offset_lon)*(RomsNC.variables['lon_u'][:] == lonbounds[1]-c_offset_lon)
    EastFace = np.ma.asarray(np.invert(Ulats*Ulons))
    
    return NorthFace, WestFace, SouthFace, EastFace