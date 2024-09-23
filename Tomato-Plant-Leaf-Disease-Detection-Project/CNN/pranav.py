import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
    
		# Read H5 file
f = h5.File(r"D:\second\GUI app for tomato leafs disease classification using CNN (1)\GUI app for tomato leafs disease classification using CNN\PlantLeafCNN_2023-04-06_final_VGG16.h5", "r")
# Get and print list of datasets within the H5 file
datasetNames = [n for n in f.keys()]
for n in datasetNames:
	print(n)
	
