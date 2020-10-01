# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 13:26:02 2020

@author: Jasen
"""
import numpy as np
from netCDF4 import Dataset as nc4
import obs_depth_JJ as dep
import ROMS_Tools_Mask as rt



def dA_int_w(RomsFile) :
    """ Compute differential area of each cell interpolating depth at w points to u, v points """
    RomsNC = nc4(RomsFile, 'r')
      
    #depth at w points
    depth_w = dep._set_depth_T(RomsFile, None, 'w', RomsNC.variables['h'][:],\
                               RomsNC.variables['zeta'][:])
    dz_w = np.diff(depth_w, n = 1, axis = 1)
    
    # lon at rho points 
    lon_rho0 = rt.AddDepthTime(RomsFile, \
                               RomsNC.variables['lon_rho'][:, 0:dz_w.shape[3]-1])
    lon_rho1 = rt.AddDepthTime(RomsFile, \
                               RomsNC.variables['lon_rho'][:, 1:dz_w.shape[3]])
        
    # lon at u point
    lon_u = rt.AddDepthTime(RomsFile, RomsNC.variables['lon_u'][:])
    
    #dz at rho points    
    z_rho0 = dz_w[:, :, 0:dz_w.shape[2]-1] 
    z_rho1 = dz_w[:, :, 1:dz_w.shape[2]]
    
    #interpolation to u points 
    dz_u = z_rho0 + (z_rho1 - z_rho0)/(lon_rho1 - lon_rho0)*(lon_u - lon_rho0)
    
    # depth at v points
    # lats
    lat_rho0 = RomsNC.variables['lat_rho'][0:dz_w.shape[1]-1, :]
    lat_rho1 = RomsNC.variables['lat_rho'][1:dz_w.shape[1], :]
    lat_v = RomsNC.variables['lat_v'][:]
    
    #depths at w points
    z_Vrho0 = dz_w[:, 0:dz_w.shape[1]-1, :]
    z_Vrho1 = dz_w[:, 1:dz_w.shape[1], :]
    
    #interpolate between rho points, u
    dz_v = z_Vrho0 - (z_Vrho1 - z_Vrho0)/(lat_rho1 - lat_rho0)*(lat_v - lat_rho0)
    
    #sides of cells
    dx = np.repeat(1/np.array(RomsNC.variables['pm'])[np.newaxis, :, :], dz_v.shape[0], axis = 0)
    dx = dx[:, 0:dz_v.shape[1], 0:dz_v.shape[2]]
    dy = np.repeat(1/np.array(RomsNC.variables['pn'])[np.newaxis, :, :], dz_u.shape[0], axis = 0)
    dy = dy[:, 0:dz_u.shape[1], 0:dz_u.shape[2]]
   
    #compute area
    A_xz = dx*dz_v
    A_yz = dy*dz_u
    
    return A_xz, A_yz

def dAz_psi(RomsFile) :
    """Compute differential area of each cell using a trapazoid at psi points"""
    #load roms file
    RomsNC = nc4(RomsFile, 'r')
    
    #compute z
    romsvars = {'h' : RomsNC.variables['h'], \
                'zeta' : RomsNC.variables['zeta']}
    
    #compute depth at w points
    depth_domain = dep._set_depth(RomsFile, None, 'psi', romsvars['h'], romsvars['zeta'])        
    dz = np.diff(depth_domain, n = 1, axis = 0)
    
    #compute lengths of horizontal cell directions repeat over depth axis
    dx = np.repeat(1/np.array(RomsNC.variables['pm'])[np.newaxis, :, :], dz.shape[0], axis = 0)
    dy = np.repeat(1/np.array(RomsNC.variables['pn'])[np.newaxis, :, :], dz.shape[0], axis = 0)
    
    #compute differential area assuming trapizoidal shape
    A_xz = np.empty(dz.shape)
    A_xz.fill(np.nan)
    
    A_yz = np.empty(dz.shape)
    A_yz.fill(np.nan)
    
    Lm = dz.shape[1]
    Mm = dz.shape[2]
    
    #x area and y area
    for k in range(1, dz.shape[1]-1):
        A_xz[:,k-1,:] = 0.5*(dz[:, k, 0:Mm] + dz[:, k-1, 0:Mm])*dx[:,k-1,0:Mm]
    
    for k in range(1, dz.shape[2]-1):
       A_yz[:, :, k-1] = 0.5*(dz[:, 0:Lm, k]+ dz[:, 0:Lm, k-1])*dy[:, 0:Lm, k-1]
       
    return A_xz, A_yz
