# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 10:17:22 2020

@author: nabeel
"""



import splitfolders
path = "Data_set_path"


splitfolders.ratio(path, output="Split_Covid-Pneumonia_Dataset", seed=1337,ratio=(.7, .1,.2))
