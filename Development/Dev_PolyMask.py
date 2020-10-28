# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 12:24:24 2020
Use method similar to inpolygon matlab funciton to create mask based on 
supplied vertices
@author: Jasen
"""
import os
os.chdir('/Users/Jasen/Documents/GitHub/SalinityVarianceBudget/Subroutines/')
from matplotlib import path 
from netCDF4 import Dataset as nc4
import numpy as np

# files
RomsFile = '/Users/Jasen/Documents/Data/ROMS_TestRuns/wc12_his_5day.nc'
RomsNC = nc4(RomsFile, 'r')

#bounds of control volume
latbounds = [35, 37]
lonbounds = [-126, -125]


lats = RomsNC.variables['lat_rho'][:]
lons = RomsNC.variables['lon_rho'][:]

#Pauls method for a rectangular mask
ilat = np.logical_and(lats >= latbounds[0], lats < latbounds[1])
ilon = np.logical_and(lons >= lonbounds[0], lons < lonbounds[1])

mask = np.logical_and(ilat, ilon) #square




#poly path
vert = np.array([[35, -126], [35, -125], [37, -125], [37, -126]])
locs = [lats, lons]


#reshape lat and lon to N by 2
locs = np.transpose([np.resize(lats, lats.size), np.resize(lons, lons.size)])

p = path.Path(vert)
inside = p.contains_points(locs)

Pmask = np.resize(inside, lats.shape)


m = path.Path.contains_points(vert)


#shift mask
dwn = Pmask[1:, :]
up = Pmask[:-1, :]

left = Pmask[:,1:]
right= Pmask[:,:-1]

#north mask
Nmask = np.concatenate((Pmask[0:1,:], np.logical_and(dwn == True, up == False)))

#south mask
Smask = np.concatenate((np.logical_and(up == True, dwn == False), Pmask[-2:-1, :]))

#west mask
Wmask = np.concatenate((np.logical_and(right == True, left == False), Pmask[:, -2:-1]), axis = 1)

imask = np.logical_and(left == True, right == False)

Emask = np.concatenate((Pmask[:,0:1], np.logical_and(left == True, right == False)), axis = 1)


lat = RomsNC.variables['lat_rho'][:, 0]
lon = RomsNC.variables['lon_rho'][0, :]
mask = np.logical_and((lat[:,None] > 35, lat[:,None] < 37), (lon[None,:] < -127, lon[None,:] > -125)) # the mask is 5-by-3

latbounds = (35, 37)
lonbounds = (-127, -125)
#load lat and lon as vectors
#lat = RomsNC.variables['lat_rho'][:, 0]
#lon = RomsNC.variables['lon_rho'][0, :] 

lats = np.linspace(30.0, 40.0, 10)
lons = np.linspace(-130, -110, 20) 

#indices of latbounds 
idx_lat0 = np.argmin(np.abs(lats - latbounds[0])) 
idx_lat1 = np.argmin(np.abs(lats - latbounds[1])) 

#indices of lonbounds
idx_lon0 = np.argmin(np.abs(lons - lonbounds[0])) 
idx_lon1 = np.argmin(np.abs(lons - lonbounds[1])) 

#create mask of zeros
mask = np.zeros((lats.size, lons.size))
#set area inside to 1
mask[idx_lat0:idx_lat1, idx_lon0:idx_lon1] = 1
