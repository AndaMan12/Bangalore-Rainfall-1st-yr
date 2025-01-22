# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 22:15:51 2022

@author: guria
"""
import xarray as xr
import os
import numpy as np
import scipy.interpolate as interpolate
import scipy.fft as fft
import scipy.signal as signal
import matplotlib.pyplot as plt
diri=r"C:\Users\guria\trmm"
valid_paths = next(os.walk(diri), (None, None, []))[2]



index_lat= lambda lat: int((lat+49.875)/0.25)
index_lon= lambda lon: int((lon+179.875)/0.25)
lat= lambda index_lat: -49.875+index_lat*0.25
lon= lambda index_lon: -179.875+index_lon*0.25
Bangalore_lat_index=[index_lat(12.375),index_lat(13.375)]
#print(Bangalore_lat_index)
Bangalore_lon_index=[index_lon(77.375),index_lon(78.375)]

monthly_avg_rain=np.zeros(shape=(22,12,Bangalore_lon_index[1]-Bangalore_lon_index[0]+1, 
                                 Bangalore_lat_index[1]-Bangalore_lat_index[0]+1))

m=0
y=0
mon='01'
yr='1998'
days=0
for path in valid_paths:
   
    if (mon!=str(path[15:17])): 
        print("month_processed:",m)
        monthly_avg_rain[y,m,:,:]/days
        m+=1
        mon=path[15:17]
        days=0
        if (yr!=str(path[11:15])): 
            print("year_processed:",y)            
            y+=1
            yr=path[11:15]
            m=0
    ds=xr.open_dataset(r"C:\Users\guria\trmm"+"\\"+path,engine="netcdf4")
    rain=ds['precipitation'].values
    monthly_avg_rain[y,m,:,:]+=rain[Bangalore_lon_index[0]:Bangalore_lon_index[1]+1,Bangalore_lat_index[0]:Bangalore_lat_index[1]+1]
    days+=1

x=[lon(i) for i in range(Bangalore_lon_index[0],Bangalore_lon_index[1]+1)]
y=[lat(i) for i in range(Bangalore_lat_index[0],Bangalore_lat_index[1]+1)]
X,Y=np.meshgrid(x,y)
net_rain_vol1=np.sum(monthly_avg_rain,(2,3))
normalised_rain=np.array([net_rain_vol1[y,:]/np.sum(net_rain_vol1[y,:]) for y in range(net_rain_vol1.shape[0])])
'''
print(net_rain_vol1)
print(normalised_rain)
normalised_chart=np.sum(normalised_rain,0)
print(normalised_chart)
N=48
spacing=1
xf=fft.fftfreq(N,spacing)[:N//2]
fft_rand=[]
FFT_RAND=[]
for i in range(normalised_chart.shape[0]-3):
    fft_rand=list(normalised_rain[i,:])
    fft_rand.extend(list(normalised_rain[i+1,:]))
    fft_rand.extend(list(normalised_rain[i+2,:]))
    fft_rand.extend(list(normalised_rain[i+3,:]))
    FFT_RAND.append(fft_rand)
yf=[(2/N)*np.abs(fft.fft(FFT_RAND[i])[:N//2]) for i in range(normalised_chart.shape[0]-3)]

YY=np.sum(np.array(yf),0)
peaks,_ =signal.find_peaks(YY)
peak_freq=[xf[i] for i in peaks]
print(peak_freq)
'''
'''
def monthly_rain(yr,month,point):
    x_ind=index_lon(point[0])-Bangalore_lon_index[0]
    y_ind=index_lat(point[1])-Bangalore_lat_index[0]
    
    return(30*monthly_avg_rain[yr,month,x_ind,y_ind])


X1 = np.linspace(min(x), max(x),num=5000)
Y1 = np.linspace(min(y), max(y),num=5000)
    
X1,Y1=np.meshgrid(X1,Y1) 
#First we interpolate the grid points. Then generate a continuous curve through them

def monthly_interpolated(year,month):
  
    mask=[~np.isnan(monthly_avg_rain[year,month,:,:])]
    
    x_mask = X[mask].reshape(-1)
    y_mask = Y[mask].reshape(-1)
    
    points_grid = np.array([x_mask,y_mask]).T
    values_grid = monthly_avg_rain[month,:,:][mask].reshape(-1) 
    
    interp_grid = interpolate.griddata(points_grid, values_grid, (X, Y), method='nearest')
    monthly_avg_rain[month,:,:]=interp_grid
   # print(monthly_avg_rain[month,:,:])
   
   points=[]
   values = []
   
   for px in x:
       for py in y:
           points.append([px,py])
           values.append(monthly_rain(year,month,[px,py])) 
           
   interp = interpolate.CloughTocher2DInterpolator(points,values )
   
   return(interp)

fig = plt.figure(figsize=(8,7))
ax=plt.axes(projection='3d')
Z=monthly_interpolated(20,6)(X1,Y1)    
im1 = ax.scatter3D(Y,X,monthly_avg_rain[20,6,:,:],color='b')
#im2 = ax.plot_surface(X1,Y1,Z,color='r')
plt.show()      
'''