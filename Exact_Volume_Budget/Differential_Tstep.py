# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 11:10:28 2020

@author: Jasen
Tools used to create differential volumes and areas of each cell in roms output
takes time step into account

"""
import numpy as np
import GridShift_3D as GridShift

def dV(tstep, RomsFile, dx, dy, dz) :
    """ 
    Load roms grid at specific time step
    and computes differential volume of each cell
    """
    
    #compute differential volume of each cell
    DV = dx*dy*dz
    
    return DV
    
def dA_norm(dx, dy, dz) :
    """
    Compute area of vertical grid cell faces 
    (Ax -> lat X depth, Ay -> lon X depth)
    """
    #average depth at rho points to u points
    dz_u = GridShift.Rho_to_Upt(dz)
    
    #average depth at rho  points to v points
    dz_v = GridShift.Rho_to_Vpt(dz)
    
    #shift dy to u points and dx to v points
    dy_u = GridShift.Rho_to_Upt(dy)
    dx_v = GridShift.Rho_to_Vpt(dx)
    
    #Area of face with x-normal
    Ax_norm = dz_u*dy_u
    
    #Area of face with y-normal
    Ay_norm = dz_v*dx_v
    
    return Ax_norm, Ay_norm

def dA_top(RomsNC):
    ndepth = RomsNC.variables['salt'][0, :, 0, 0].size
    
    #dlon dlat
    dx = 1/np.array(RomsNC.variables['pm'][:])
    dy = 1/np.array(RomsNC.variables['pn'][:])
    
    #dA
    dA = dx*dy
    dA_depth = np.repeat(np.array(dA)[np.newaxis, :, :], ndepth, axis = 0)
    
    return dA_depth 
    
