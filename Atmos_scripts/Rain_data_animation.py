# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 19:20:29 2022

@author: guria
"""
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 22:15:51 2022

@author: guria
"""
import xarray as xr
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.interpolate as interpolate
import datetime
import pandas as pd
import scipy.signal as signal
import scipy.fft as fft
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

diri=r"C:\Users\guria\trmm"
valid_paths = next(os.walk(diri), (None, None, []))[2]


index_lat= lambda lat: int((lat+49.875)/0.25)
index_lon= lambda lon: int((lon+179.875)/0.25)
lat= lambda index_lat: -49.875+index_lat*0.25
lon= lambda index_lon: -179.875+index_lon*0.25
Bangalore_lat_index=[index_lat(11.375),index_lat(14.375)]
Bangalore_lon_index=[index_lon(76.375),index_lon(79.375)]


start = datetime.datetime.strptime("19980101", "%Y%m%d")
end = datetime.datetime.strptime("20191230", "%Y%m%d")
date_generated = pd.date_range(start, end)
dates=date_generated.strftime("%Y%m%d")


daily_data=[]

for date in dates:
    search_path='3B42_Daily.'+date+'.7.nc4'
    if search_path in valid_paths:
        ds=xr.open_dataset(r"C:\Users\guria\trmm"+"\\"+search_path,engine="netcdf4")
        rain=ds['precipitation'].values        
        daily_data.append(rain[Bangalore_lon_index[0]:Bangalore_lon_index[1]+1,Bangalore_lat_index[0]:Bangalore_lat_index[1]+1])
    else:
        daily_data.append("dummy")  
missed=[i for i,x in enumerate(daily_data) if x=="dummy"]
x=[i for i in range(len(daily_data)) if i not in missed]
y=list(daily_data)

try:
    while True:
        y.remove("dummy")
except ValueError:
    pass
y=np.array(y)
daily_recon=np.zeros(shape=(len(daily_data),13,13))
x_1=np.array([i for i in range(len(daily_data))])
for i in range(13):
    for j in range(13):
        f=interpolate.interp1d(x,y[:,i,j])        
        daily_recon[:,i,j]=f(x_1)

x=[lon(i) for i in range(Bangalore_lon_index[0],Bangalore_lon_index[1]+1)]
y=[lat(i) for i in range(Bangalore_lat_index[0],Bangalore_lat_index[1]+1)]
X,Y=np.meshgrid(x,y)

print("done here")

plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = False


fps = 1
frn = 12*22


zarray = daily_recon


def change_plot(frame_number, zarray):
   ax.clear()
   ax.contourf(X, Y, zarray[frame_number,:, :], 100, cmap="cool")
   ax.set_yticks([11,11.5,12,12.5,13,13.5,14], crs=ccrs.PlateCarree())
   ax.set_xticks([76,76.5,77,77.5,78,78.5,79], crs=ccrs.PlateCarree())
   lon_formatter = LongitudeFormatter(zero_direction_label=True,dateline_direction_label=False,number_format='.0f')
   ax.xaxis.set_major_formatter(lon_formatter)
   lat_formatter = LatitudeFormatter()
   ax.yaxis.set_major_formatter(lat_formatter)
   ax.coastlines(linewidth=1, alpha=0.5)
proj=ccrs.PlateCarree()
fig = plt.figure()
ax = plt.subplot(111, projection=proj)

ax.contourf(X, Y, zarray[0,:, :], 100, cmap="cool")
ax.set_yticks([11,11.5,12,12.5,13,13.5,14], crs=ccrs.PlateCarree())
ax.set_xticks([76,76.5,77,77.5,78,78.5,79], crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter(zero_direction_label=True,dateline_direction_label=False,number_format='.0f')
ax.xaxis.set_major_formatter(lon_formatter)
lat_formatter = LatitudeFormatter()
ax.yaxis.set_major_formatter(lat_formatter)
ax.coastlines(linewidth=1, alpha=0.5)

ani = animation.FuncAnimation(fig, change_plot, frn, fargs=(zarray,), interval=1000 / fps)



plt.show()