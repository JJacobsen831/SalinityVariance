# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 12:29:58 2020

@author: Jasen
"""
# ROMS data extraction for CV budget
import numpy as np
from netCDF4 import Dataset as dt
import obs_depth_JJ as dep
from Filters import godinfilt 
import seawater as sw

def RhoPsiIndex(RomsFile, latbounds, lonbounds):
    """locates indices of lat and lon within ROMS Output File assuming regular spacing"""
    #load nc file
    RomsNC = dt(RomsFile, 'r')
    
    #check if lat bounds are increasing
    if latbounds[0] < latbounds[1] :
        #locate lat rho points within bounds 
        RhoInd = {'lat_li' : np.argmin(np.array(np.abs(RomsNC.variables['lat_rho'][:, 0] - latbounds[0]))), \
                  'lat_ui' : np.argmin(np.array(np.abs(RomsNC.variables['lat_rho'][:, 0] - latbounds[1]))), \
                 }
        #locate lat psi points within bounds; add one to psi upper bound
        PsiInd = {'lat_li' : np.argmin(np.array(np.abs(RomsNC.variables['lat_psi'][:, 0] - latbounds[0]))), \
                  'lat_ui' : np.argmin(np.array(np.abs(RomsNC.variables['lat_psi'][:, 0] - latbounds[1])))+1, \
                  }
        
    else:
        RhoInd = {'lat_li' : np.argmin(np.array(np.abs(RomsNC.variables['lat_rho'][:, 0] - latbounds[1]))), \
                  'lat_ui' : np.argmin(np.array(np.abs(RomsNC.variables['lat_rho'][:, 0] - latbounds[0]))), \
                  }
        
        PsiInd = {'lat_li' : np.argmin(np.array(np.abs(RomsNC.variables['lat_psi'][:, 0] - latbounds[1]))), \
                  'lat_ui' : np.argmin(np.array(np.abs(RomsNC.variables['lat_psi'][:, 0] - latbounds[0])))+1, \
                  }
    
    #check if lon bounds are increasing
    if lonbounds[0] < lonbounds[1] :
        #locate lon rho & psi points; add 1 to psi points to get last grid cell
        RhoInd['lon_li'] = np.argmin(np.array(np.abs(RomsNC.variables['lon_rho'][0, :] - lonbounds[0])))
        RhoInd['lon_ui'] = np.argmin(np.array(np.abs(RomsNC.variables['lon_rho'][0, :] - latbounds[1])))
                  
        PsiInd['lon_li'] = np.argmin(np.array(np.abs(RomsNC.variables['lon_psi'][0, :] - lonbounds[0])))
        PsiInd['lon_ui'] = np.argmin(np.array(np.abs(RomsNC.variables['lon_psi'][0, :] - latbounds[1])))+1
        
    else :
        RhoInd['lon_li'] = np.argmin(np.array(np.abs(RomsNC.variables['lon_rho'][0, :] - lonbounds[1])))
        RhoInd['lon_ui'] = np.argmin(np.array(np.abs(RomsNC.variables['lon_rho'][0, :] - lonbounds[0])))
        
        PsiInd['lon_li'] = np.argmin(np.array(np.abs(RomsNC.variables['lon_psi'][0,:] - lonbounds[1])))
        PsiInd['lon_ui'] = np.argmin(np.array(np.abs(RomsNC.variables['lon_psi'][0,:] - lonbounds[0])))+1
    
    IndBounds = {'Rho' : RhoInd,'Psi' : PsiInd}
    
    return IndBounds


def ROMS_CV_Load(RomsFile, VarName, IndBounds) :
    """Loads ROMS variables in control volume defined by lat and lon bounds
    lat_rho/psi, lon_rho/psi, time, u, ubar, v, vbar, w, and a defined variable
    at defined latitude and longitude. Stores in dictionary"""
    #load nc file   
    RomsNC = dt(RomsFile, 'r')
    
    #subset variables in control volume and store in dictionary
    ROMS_CV = {'lat_rho' : np.array(RomsNC.variables['lat_rho'][IndBounds['Rho']['lat_li']:IndBounds['Rho']['lat_ui'], \
                                                     IndBounds['Rho']['lon_li']:IndBounds['Rho']['lon_ui']], dtype = np.float64), \
            'lon_rho' : np.array(RomsNC.variables['lon_rho'][IndBounds['Rho']['lat_li']:IndBounds['Rho']['lat_ui'],\
                                                  IndBounds['Rho']['lon_li']:IndBounds['Rho']['lon_ui']], dtype = np.float64), \
            'time' : np.array(RomsNC.variables['ocean_time'][:], dtype = np.float64), \
            #'w' : np.array(RomsNC.variables['w'][:, :, IndBounds['Rho']['lat_li']:IndBounds['Rho']['lat_ui'],IndBounds['Rho']['lon_li']:IndBounds['Rho']['lon_ui']], dtype = np.float64), \
            VarName : np.array(RomsNC.variables[VarName][:, :, IndBounds['Rho']['lat_li']:IndBounds['Rho']['lat_ui'], \
                                                IndBounds['Rho']['lon_li']:IndBounds['Rho']['lon_ui']], dtype = np.float64), \
            # psi variables (on edges)
            'lat_psi': np.array(RomsNC.variables['lat_psi'][IndBounds['Psi']['lat_li']:IndBounds['Psi']['lat_ui'], \
                                                 IndBounds['Psi']['lon_li']:IndBounds['Psi']['lon_ui']], dtype = np.float64), \
            'lon_psi': np.array(RomsNC.variables['lon_psi'][IndBounds['Psi']['lat_li']:IndBounds['Psi']['lat_ui'], \
                                                 IndBounds['Psi']['lon_li']:IndBounds['Psi']['lon_ui']], dtype = np.float64), \
            'u' : np.array(RomsNC.variables['u'][:, :, IndBounds['Psi']['lat_li']:IndBounds['Psi']['lat_ui'], \
                                                 IndBounds['Psi']['lon_li']:IndBounds['Psi']['lon_ui']], dtype = np.float64), \
            #'ubar' : np.array(RomsNC.variable['ubar'][:, IndBounds['Psi']['lat_li']:IndBounds['Psi']['lat_ui'], IndBounds['Psi']['lon_li']:IndBounds['Psi']['lon_ui']], dtype = np.float64), \
            'v' : np.array(RomsNC.variables['v'][:, :, IndBounds['Psi']['lat_li']:IndBounds['Psi']['lat_ui'], \
                                                 IndBounds['Psi']['lon_li']:IndBounds['Psi']['lon_ui']], dtype = np.float64), \
            #'vbar' : np.array(RomsNC.variables['vbar'][:, IndBounds['Psi']['lat_li']:IndBounds['Psi']['lat_ui'], IndBounds['Psi']['lon_li']:IndBounds['Psi']['lon_ui']], dtype = np.float64)
            }
        
    return ROMS_CV


def ROMS_CV_AddVar(RomsFile, ROMS_CV, VarName, IndBounds):
    """Add variable to ROMS control volume dictionary """
    RomsNC = dt(RomsFile, 'r')
    
    #update dictionary with new variable
    ROMS_CV[VarName] = np.array(RomsNC.variables[VarName][:, :, \
           IndBounds['Rho']['lat_li']:IndBounds['Rho']['lat_ui'], \
           IndBounds['Rho']['lon_li']:IndBounds['Rho']['lon_ui']], dtype = np.float64)
    
    return ROMS_CV


def Reynolds_Avg_Boxcar(Var):
    """takes temporal average at each grid cell and subtracts to get avg and prime terms """
    # temporal mean
    Var_Tmean = np.mean(Var, axis = 0)
    
    # prime value
    Var_prime = Var - Var_Tmean
    
    ReynAvg = {'avg' : Var_Tmean, 'prime' : Var_prime}
    
    return ReynAvg

def Reynolds_Avg_Godin(Var, Time):
    """applies godin filter to remove tidal signal and returns avg and prime 
    NOTE: Time should be in hourly increments with units of seconds"""
    #check time spacing
    if np.mean(np.diff(Time)) != 3600:
        raise NotImplementedError('Time not in hourly increments')
    
    #apply godin filter    
    Tmean, Filtedge = godinfilt(Var)
    
    #pad ends with nan
    Tmean[0:Filtedge] = np.nan
    Tmean[-Filtedge:-1] = np.nan
    Tmean[-1] = np.nan
    
    prime = Var - Tmean
    
    ReynAvg = {'avg' : Tmean, 'prime' : prime}
    
    return ReynAvg

def CV_VolAvg(Var) :
    """Compute volume average of control volume """
    raise NotImplementedError('Contol Volume average not developed yet')

def ModelDepth(RomsFile, point_type, IndBounds):
    """Computes ROMS depth within control volume defined by lat and lon bounds
    uses obs_depth, converted from set_depth.m"""
    #load nc file
    RomsNC = dt(RomsFile, 'r')
       
    #ROMS variables
    romsvars = {'h' : RomsNC.variables['h'], \
                'zeta' : RomsNC.variables['zeta'],\
                'N' : RomsNC.variables['Cs_r'].size}
    #compute depth
    depth_domain = dep._set_depth(RomsFile, None, point_type, romsvars['h'], romsvars['zeta'])
    
    #subset at control volume
    depth = np.array(depth_domain[:,IndBounds['Rho']['lat_li']:IndBounds['Rho']['lat_ui'],\
                                  IndBounds['Rho']['lon_li']:IndBounds['Rho']['lon_ui']])
    
    return depth

def rho_dist(RomsFile) :
    """
    Use seawater package to compute distance between rho points
    """
    RomsNC = dt(RomsFile, 'r')
    
    lat = RomsNC.variables['lat_rho'][:]
    lon = RomsNC.variables['lon_rho'][:]
    
    x_dist = np.empty((lon.shape[0], lon.shape[1]-1))
    x_dist.fill(np.nan)
    for i in range(lon.shape[0]) :
        for j in range(lon.shape[1]-1) :
            x_dist[i, j] = sw.dist([lat[i, j], lat[i, j+1]], [lon[i, j], lon[i, j +1]])[0]
            
    y_dist = np.empty((lat.shape[0]-1, lat.shape[1]))
    y_dist.fill(np.nan)
    for i in range(y_dist.shape[0]):
        for j in range(y_dist.shape[1]) :
            y_dist[i, j] = sw.dist([lat[i,j], lat[i+1, j]], [lon[i,j], lon[i+1, j]])
            
    return x_dist, y_dist