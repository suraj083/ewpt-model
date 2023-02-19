import numpy as np
import tensorflow as tf
import elvet
import matplotlib.pyplot as plt
from scipy import interpolate

# read data from txt files

Jf_data = np.loadtxt("JF_table.txt", dtype=float)
Jf_data = Jf_data[(-100 <= Jf_data[:,0]) & (Jf_data[:,0] <= 100)]

plt.plot(Jf_data[:,0], Jf_data[:,1])
plt.savefig("fig1.jpg")
plt.close()

# obtaining a fit using elvet

domain = elvet.box((-50, 100, 3001)) 

Jf_interpol = interpolate.splrep(Jf_data[:,0], Jf_data[:,1], k=3, s=0)
y_Jf = lambda x : tf.cast(interpolate.splev(x.numpy(), Jf_interpol, der=0), domain.dtype)

model = elvet.nn(1, 20, 20, 20, 1)

fitter_Jf = elvet.fitter(domain, y_Jf(domain), model=model, epochs=50000, lr=0.001)
fitter_Jf.save_model_weights("Jf.h5")

y_Jf = lambda x : tf.cast(interpolate.splev(x, Jf_interpol, der=0), domain.dtype)

x = np.linspace(-50,100,3001)
plt.plot(x, fitter_Jf.model(x))
plt.plot(x, y_Jf(x))
plt.savefig("fit_Jf.jpg")
plt.close()