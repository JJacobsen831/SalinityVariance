# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 12:27:28 2020

@author: Jasen
Tools used to create differential volumes and areas of each cell in roms output
"""
import numpy as np
from netCDF4 import Dataset as nc4
import obs_depth_JJ as dep
import ROMS_Tools_Mask as rt

def dV(RomsFile) :
    """ Load full roms grid and compute differential volume of each cell"""
    RomsNC = nc4(RomsFile, 'r')

    #compute z
    romsvars = {'h' : RomsNC.variables['h'][:], \
                'zeta' : RomsNC.variables['zeta'][:]}
    
    #compute depth at w points
    depth_domain = dep._set_depth_T(RomsFile, None, 'w', romsvars['h'], romsvars['zeta'])
        
    dz = np.diff(depth_domain, n = 1, axis = 1)
            
    #compute lengths of horizontal cell directions & repeat over depth space
    dx = rt.AddDepthTime(RomsFile, 1/np.array(RomsNC.variables['pm'][:]))
    dy = rt.AddDepthTime(RomsFile, 1/np.array(RomsNC.variables['pn'][:]))
    
    #compute differential volume of each cell
    DV = dx*dy*dz
    
    return DV
    
def dA(RomsFile, RomsGrd) :
    """
    Compute area of vertical grid cell faces 
    (Ax -> lat X depth, Ay -> lon X depth)
    """
    RomsNC = nc4(RomsFile, 'r')
    
    #depth at w points
    depth_w = dep._set_depth_T(RomsFile, None, 'w', RomsNC.variables['h'][:],\
                               RomsNC.variables['zeta'][:])
    dz_w = np.diff(depth_w, n = 1, axis = 1)
    
    #average depth at w points to u points
    dz_u = 0.5*(dz_w[:,:,:, 0:dz_w.shape[3]-1] + dz_w[:,:,:,1:dz_w.shape[3]])
    
    #average depth at w points to v points
    dz_v = 0.5*(dz_w[:,:,0:dz_w.shape[2]-1,:] + dz_w[:,:,1:dz_w.shape[2], :])
    
    #cell widths
    dx0, dy0 = rt.cell_width(RomsGrd)
    
    #expand over depth and time dimensions
    dx = rt.AddDepthTime(RomsFile, dx0)
    dy = rt.AddDepthTime(RomsFile, dy0)
    
    #Area of face with x-normal
    Ax_norm = dz_u*dy
    
    #Area of face with y-normal
    Ay_norm = dz_v*dx
    
    return Ax_norm, Ay_norm


