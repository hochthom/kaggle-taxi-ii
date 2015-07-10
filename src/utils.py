# -*- coding: utf-8 -*-

import numpy as np


# City center: median of trip end positions
CITY_CENTER    = np.array([[-8.615223, 41.157819]], ndmin=2)

# 99 percentil of trip length
TRIP_LENGTH_99 = 185.0

# bounding box 1 / 99 percentil
LON_RANGE = (-8.692785090000001, -8.533710000000001)
LAT_RANGE = (41.107103909999999, 41.237838000000004)

# distance in degree to km: simple linear approximation at city center 
LON_SCALE = 167.438938422
LAT_SCALE = 222.389853289


def rmsle(y_true, y_pred):
    return np.sqrt(np.mean((np.log(y_true + 1) - np.log(y_pred + 1))**2))
    
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def haversineKaggle(p1, p2):
    r = 6371
    p1 = np.array(p1, ndmin=2)
    p2 = np.array(p2, ndmin=2)
    p1 = np.radians(p1)
    p2 = np.radians(p2)
    dlon = abs(p2[:,0] - p1[:,0])
    dlat = abs(p2[:,1] - p1[:,1])
    a = np.sin(dlat)**2 + np.cos(p1[:,1])*np.cos(p2[:,1])*np.sin(dlon)**2
    c = 2 * r * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return c

def heading(p1, p2):
    p1 = np.radians(p1)
    p2 = np.radians(p2)
    lat1, lon1 = p1[1], p1[0]
    lat2, lon2 = p2[1], p2[0]
    aa = np.sin(lon2 - lon1) * np.cos(lat2)
    bb = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lon2 - lon1)
    return np.arctan2(aa, bb) + np.pi 

