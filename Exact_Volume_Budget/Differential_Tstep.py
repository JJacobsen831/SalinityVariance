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
    
def dA(tstep, RomsNC, dz_rho) :
    """
    Compute area of vertical grid cell faces 
    (Ax -> lat X depth, Ay -> lon X depth)
    """
    #average depth at rho points to u points
    dz_u = GridShift.Rho_to_Upt(dz_rho)
    
    #average depth at rho  points to v points
    dz_v = GridShift.Rho_to_Vpt(dz_rho)
    
    #cell widths
    dx_rho = np.repeat(1/np.array(RomsNC.variables['pm'][:])[np.newaxis, :, :], dz_rho.shape[0], axis = 0)
    dy_rho = np.repeat(1/np.array(RomsNC.variables['pn'][:])[np.newaxis, :, :], dz_rho.shape[0], axis = 0)
    
    #shift to u and v points
    dy = GridShift.Rho_to_Upt(dy_rho)
    dx = GridShift.Rho_to_Vpt(dx_rho)
    
    #Area of face with x-normal
    Ax_norm = dz_u*dy
    
    #Area of face with y-normal
    Ay_norm = dz_v*dx
    
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
    
