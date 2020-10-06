# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 16:07:37 2020

Exact budget subroutines compute each term at time step

@author: Jasen
"""
import numpy.ma as ma
import obs_depth_JJ as dep
import GridShift
import ROMS_Tools_Mask as rt

def TimeDeriv(tstep, Hist, HistFile, Avg, AvgFile, Diag, dA_xy, RhoMask):
    """
    Exact volume time derivative of variance squared
    """
    #compute variance
    var = ma.array(Avg.variables['salt'][tstep, :, :, :], mask = RhoMask)
    v_prime = var - var.mean()
    
    #change in vertical thickness of cell at average point
    deltaA = ma.array(ma.diff(dep._set_depth(AvgFile, None, 'w', \
                                             Avg.variables['h'][:], \
                                             Avg.variables['zeta'][tstep,:,:]),
                              n = 1, axis = 0), \
                      mask = RhoMask)
    
    #diagnostic rate
    var_rate = ma.array(Diag.variables['salt_rate'][tstep, :, :, :], \
                         mask = RhoMask)
    
    #change in vertical cell thickness at history points
    deltaH0 = ma.array(ma.diff(dep._set_depth(HistFile, None, 'w', \
                                             Hist.variables['h'][:], \
                                             Hist.variables['zeta'][tstep,:,:]),
                              n = 1, axis = 0), \
                      mask = RhoMask)
    
    deltaH1 = ma.array(ma.diff(dep._set_depth(HistFile, None, 'w', \
                                             Hist.variables['h'][:], \
                                             Hist.variables['zeta'][tstep+1,:,:]),
                              n = 1, axis = 0), \
                      mask = RhoMask)
    
    dtH = Hist.variables['ocean_time'][tstep+1] - Hist.variables['ocean_time'][tstep]
    
    #time derivative of cell volume on avg points
    dDelta_dt = (deltaH1 - deltaH0)/dtH
    
    #compute volume
    dV = dA_xy*deltaA
    
    Int_Sprime = ma.sum(2*v_prime*(var_rate - var/deltaA*dDelta_dt)*dV)
    
    return Int_Sprime
    
def AdvFlux_Masks(AvgFile, latbounds, lonbounds):
    """
    Advective Flux across boundaries
    """
    #define masks for u and v points
    _, U_Mask, V_Mask = rt.RhoUV_Mask(AvgFile, latbounds, lonbounds)
    
    #Face Masks
    NorthFace, WestFace, SouthFace, EastFace = rt.FaceMask(AvgFile,\
                                                       latbounds, lonbounds)
    
    Masks = {
            Umask : U_Mask, \
            Vmask : V_Mask, \
            Nface : NorthFace, \
            WFace : WestFace, \
            SFace : SouthFace, \
            EFace : EastFace
            }
    
    return Masks

def AdvFlux_Calc(tstep, Avg, varname, RhoMasks, Masks):
    """
    Compute Advective Fluxes
    """
    #Variance squared 
    var = ma.array(Avg.variables[varname][tstep, :, :, :], mask = RhoMask)
    v_prime2 = (var - var.mean())**2
    
    #Rethink how this is computed....
    
    #Shift from rho to u and v points
    _var_u = GridShift.Rho_to_Upt(v_prime2)
    _var_v = GridShift.Rho_to_Vpt(v_prime2)
    
    #apply mask that is shifted to u and v points to variance squared
    var_u = ma.array(_var_u, mask = U_Mask)
    var_v = ma.array(_var_v, mask = V_Mask)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    







