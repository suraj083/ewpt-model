import numpy as np
import tensorflow as tf
import elvet
import matplotlib.pyplot as plt
from scipy import interpolate

# read data from txt files

Jb_data = np.loadtxt("JB_table.txt", dtype=float)
Jb_data = Jb_data[(-100 <= Jb_data[:,0]) & (Jb_data[:,0] <= 100)]

plt.plot(Jb_data[:,0], Jb_data[:,1])
plt.savefig("fig1.jpg")
plt.close()

# obtaining a fit using elvet

domain = elvet.box((-50, 100, 3001)) 

Jb_interpol = interpolate.splrep(Jb_data[:,0], Jb_data[:,1], k=3, s=0)
y_Jb = lambda x : tf.cast(interpolate.splev(x.numpy(), Jb_interpol, der=0), domain.dtype)

model = elvet.nn(1, 20, 20, 20, 1)

fitter_Jb = elvet.fitter(domain, y_Jb(domain), model=model, epochs=50000, lr=0.001)
fitter_Jb.save_model_weights("Jb.h5")

y_Jb = lambda x : tf.cast(interpolate.splev(x, Jb_interpol, der=0), domain.dtype)

x = np.linspace(-50,100,3001)
plt.plot(x, fitter_Jb.model(x))
plt.plot(x, y_Jb(x))
plt.savefig("fit_Jb.jpg")
plt.close()