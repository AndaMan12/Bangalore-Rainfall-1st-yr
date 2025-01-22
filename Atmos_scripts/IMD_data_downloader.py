# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 17:58:42 2022

@author: guria
"""

import imdlib as imd

# Downloading 8 years of rainfall data for India
start_yr = 1901
end_yr = 2018
variable = 'tmin' # other options are ('tmin'/ 'tmax')
file_dir = 'imd_base'

"""
fn_format   : str or None
        fn_format represent filename format. 
        Default vales is None.  Which means filesnames are accoding to the IMD naming convention
        If we specify fn_format = 'yearwise', filenames are renamed like <year.grd>

file_dir   : str or None
        Directory for downliading the files.
        If None, the currently working directory is used.

sub_dir : bool
		True : if you need subdirectory for each variable type;
        False: Files will be saved directly under main directory
proxies : dict
        Give details in curly bracket as shown in the example below
        e.g. proxies = { 'http' : 'http://uname:password@ip:port'}
"""

data = imd.get_data(variable, start_yr, end_yr, fn_format='yearwise', file_dir=file_dir)
