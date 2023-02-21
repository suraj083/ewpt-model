import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import elvet
import matplotlib.pyplot as plt
import pandas as pd
import keras

# =========================================================
#   Re-assembly of the NN models for Jb, Jf 
# =========================================================

# using the same domain but different models for Jb and Jf

domain0 = elvet.box((-100, 100, 4001)) 

model_Jb = keras.models.load_model("Jb_model") 
model_Jf = keras.models.load_model("Jf_model") 

plt.plot(domain0, model_Jb(domain0), label="$J_B(x)$", linestyle="dashed")
plt.plot(domain0, model_Jf(domain0), label="$J_F(x)$", linestyle="dashed")
plt.savefig("Figures/fitted_Jb_Jf_functions")
plt.close()

#==============================
#   Function definitions 
#==============================

# thermal functions
#----------------------

def JB(var):
    return model_Jb(var)

def JF(var):
    return model_Jf(var)

def dJB(var):
    var_b = tf.cast(var, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(var_b) 
        y_Jb = model_Jb(var_b)
    return tape.gradient(y_Jb, var_b)

def dJF(var):
    var_f = tf.cast(var, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(var_f) 
        y_Jf = model_Jf(var_f)
    return tape.gradient(y_Jf, var_f)


# h, T - dependent masses and their derivatives
#-----------------------------------------------

def mWsq_t(h):
    g2 = 0.6485
    return 0.25 * g2**2 * h**2

def mWsq_t_prime(h):
    g2 = 0.6485
    return 0.5 * g2**2 * h


def mWsq_l(h,T):
    g2 = 0.6485
    return 0.25 * g2**2 * h**2 + (11/6) * g2**2 * T**2

def mWsq_l_prime(h,T):
    g2 = 0.6485
    return 0.5 * g2**2 * h

def mWsq_l_prime_T(h,T):
    g2 = 0.6485
    return (11/3) * g2**2 * T

def mZsq_t(h):
    (g1, g2) = (0.4632, 0.6485)
    return 0.25 * (0.6 * g1**2 + g2**2) * h**2

def mZsq_t_prime(h):
    (g1, g2) = (0.4632, 0.6485)
    return 0.5 * (0.6 * g1**2 + g2**2) * h


def delta(h,T):
    (g1, g2) = (0.4632, 0.6485)
    return (1/60) * tf.math.sqrt(-2640 * T**2 * g1**2 * g2**2 * (11 * T**2 + 3 * h**2) + (22 * T**2 + 3 * h**2)**2 * (5 * g2**2 + 3 * g1**2)**2)

def delta_prime(h,T):              
    (g1, g2) = (0.4632, 0.6485)
    return (1/60) * (-7920 * h * T**2 * g1**2 * g2**2 + 6 * h * (5 * g2**2 + 3 * g1**2)**2 * (22 * T**2 + 3 * h**2)) / tf.math.sqrt(-2640 * T**2 * g1**2 * g2**2 * (11 * T**2 + 3 * h**2) + (22 * T**2 + 3 * h**2)**2 * (5 * g2**2 + 3 * g1**2)**2)

def delta_prime_T(h,T):              
    (g1, g2) = (0.4632, 0.6485)
    return (11/15) * T * (3 * g1**2 - 5 * g2**2)**2 * (3 * h**2 + 22 * T**2) / tf.math.sqrt(-2640 * T**2 * g1**2 * g2**2 * (11 * T**2 + 3 * h**2) + (22 * T**2 + 3 * h**2)**2 * (5 * g2**2 + 3 * g1**2)**2)


def mZsq_l(h,T):
    (g1, g2) = (0.4632, 0.6485)
    return 0.5 * (0.25 * (0.6 * g1**2 + g2**2) * h**2 + (11/6) * g2**2 * T**2 + (11/10) * g1**2 * T**2 + delta(h,T))

def mZsq_l_prime(h,T):
    (g1, g2) = (0.4632, 0.6485)
    return 0.5 * (0.5 * (0.6 * g1**2 + g2**2) * h + delta_prime(h,T))

def mZsq_l_prime_T(h,T):
    (g1, g2) = (0.4632, 0.6485)
    return 0.5 * ( (11/3) * g2**2 * T + (11/5) * g1**2 * T + delta_prime_T(h,T))


def mphsq_l(h,T):
    (g1, g2) = (0.4632, 0.6485)
    return 0.5 * (0.25 * (0.6 * g1**2 + g2**2) * h**2 + (11/6) * g2**2 * T**2 + (11/10) * g1**2 * T**2 - delta(h,T))

def mphsq_l_prime(h,T):
    (g1, g2) = (0.4632, 0.6485)
    return 0.5 * (0.5 * (0.6 * g1**2 + g2**2) * h - delta_prime(h,T))

def mphsq_l_prime_T(h,T):
    (g1, g2) = (0.4632, 0.6485)
    return 0.5 * ( (11/3) * g2**2 * T + (11/5) * g1**2 * T - delta_prime_T(h,T))


def mGsq(h,T):
    (mHsq, lmbd, g1, g2, yt, lmbd_SH, N) = (0.286270859792, -0.6888422590446, 0.4632, 0.6485, 0.92849, 10.90, 0.5)
    return mHsq + 0.5 * lmbd * h**2 + ((3/80) * g1**2 + (3/16) * g2**2 + (1/4) * yt**2 + (N*lmbd_SH / 12)) * T**2

def mGsq_prime(h,T):
    (mHsq, lmbd, g1, g2, yt, lmbd_SH, N) = (0.286270859792, -0.6888422590446, 0.4632, 0.6485, 0.92849, 10.90, 0.5)
    return lmbd * h

def mGsq_prime_T(h,T):
    (mHsq, lmbd, g1, g2, yt, lmbd_SH, N) = (0.286270859792, -0.6888422590446, 0.4632, 0.6485, 0.92849, 10.90, 0.5)
    return ((3/40) * g1**2 + (3/8) * g2**2 + (1/2) * yt**2 + (N*lmbd_SH / 6)) * T


def mhsq(h,T):
    (mHsq, lmbd, g1, g2, yt, lmbd_SH, N) = (0.286270859792, -0.6888422590446, 0.4632, 0.6485, 0.92849, 10.90, 0.5)
    return mHsq + 1.5 * lmbd * h**2 + ((3/80) * g1**2 + (3/16) * g2**2 + (1/4) * yt**2 + (N*lmbd_SH / 12)) * T**2

def mhsq_prime(h,T):
    (mHsq, lmbd, g1, g2, yt, lmbd_SH, N) = (0.286270859792, -0.6888422590446, 0.4632, 0.6485, 0.92849, 10.90, 0.5)
    return 3 * lmbd * h

def mhsq_prime_T(h,T):
    (mHsq, lmbd, g1, g2, yt, lmbd_SH, N) = (0.286270859792, -0.6888422590446, 0.4632, 0.6485, 0.92849, 10.90, 0.5)
    return ((3/40) * g1**2 + (3/8) * g2**2 + (1/2) * yt**2 + (N*lmbd_SH / 6)) * T


def mtsq(h):
    yt = 0.92849                 
    return 0.5 * yt**2 * h**2 

def mtsq_prime(h):
    yt = 0.92849                 
    return yt**2 * h

def mbsq(h):
    yb = 0.0167
    return 0.5 * yb**2 * h**2 

def mbsq_prime(h):
    yb = 0.0167
    return yb**2 * h 


def mSsq(h,T):
    (muSsq, lmbd_SH, lmbd_S, N) = (4.4819542, 10.90, 1.0, 0.5)    
    return muSsq + 0.5 * lmbd_SH * h**2 + ((lmbd_S / 12) * (N+1) + (lmbd_SH / 6)) * T**2

def mSsq_prime(h,T):
    (muSsq, lmbd_SH, lmbd_S, N) = (4.4819542, 10.90, 1.0, 0.5)    
    return lmbd_SH * h

def mSsq_prime_T(h,T):
    (muSsq, lmbd_SH, lmbd_S, N) = (4.4819542, 10.90, 1.0, 0.5)    
    return ((lmbd_S / 6) * (N+1) + (lmbd_SH / 3)) * T

# Dilaton dependent functions
#-----------------------------

def Aphi(phi):
    M = 11.8321595662
    return 1 + phi**2 / (2 * M**2)

def Aphi_prime(phi):
    M = 11.8321595662
    return phi / M**2

def Vphi(phi):
    (lmbd_phi, vphi) = (0.01, 2.5)
    Vphi_pot = (1/8) * lmbd_phi * (phi**2 - vphi**2)**2
    return Vphi_pot

def Vphi_prime(phi):
    (lmbd_phi, vphi) = (0.01, 2.5)
    Vphi_deriv = (1/2) * lmbd_phi * phi * (phi**2 - vphi**2)
    return Vphi_deriv

# The Higgs potential
#-----------------------

def Vh_oneloop(phi, h, Temp):     
    hc = tf.cast(h, tf.complex64)
    phic = tf.cast(phi, tf.complex64)
    T = Temp/80

    (mHsq, lmbd, N, mu) = (0.286270859792, -0.6888422590446, 0.5, 2.1595)

    Vh_oneloop_corrected = tf.math.real(0.5 * mHsq**2/lmbd +  0.5 * mHsq * hc**2 + 0.125 * lmbd * hc**4 + ( tf.math.xlogy(4*mWsq_t(hc)**2, mWsq_t(hc)/mu**2) - (10/3) * mWsq_t(hc)**2 + tf.math.xlogy(2*mWsq_l(hc,T/Aphi(phic))**2, mWsq_l(hc,T/Aphi(phic))/mu**2) - (5/3) * mWsq_l(hc,T/Aphi(phic))**2 +  tf.math.xlogy(2*mZsq_t(hc)**2, mZsq_t(hc)/mu**2) - (5/3) * mZsq_t(hc)**2 + tf.math.xlogy(mZsq_l(hc,T/Aphi(phic))**2, mZsq_l(hc,T/Aphi(phic))/mu**2) - (5/6) * mZsq_l(hc,T/Aphi(phic))**2 + tf.math.xlogy(mphsq_l(hc,T/Aphi(phic))**2, mphsq_l(hc,T/Aphi(phic))/mu**2) - (5/6) * mphsq_l(hc,T/Aphi(phic))**2 + tf.math.xlogy(3 * mGsq(hc,T/Aphi(phic))**2, mGsq(hc,T/Aphi(phic))/mu**2) - 4.5 * mGsq(hc,T/Aphi(phic))**2 + tf.math.xlogy(mhsq(hc,T/Aphi(phic))**2, mhsq(hc,T/Aphi(phic))/mu**2) - 1.5 * mhsq(hc,T/Aphi(phic))**2 + tf.math.xlogy(2*N * mSsq(hc,T/Aphi(phic))**2, mSsq(hc,T/Aphi(phic))/mu**2) - 3*N * mSsq(hc,T/Aphi(phic))**2 - tf.math.xlogy(12 * mtsq(hc)**2, mtsq(hc)/mu**2) + 18 * mtsq(hc)**2 - tf.math.xlogy(12 * mbsq(hc)**2, mbsq(hc)/mu**2) + 18 * mbsq(hc)**2 ) / (64 * np.math.pi**2))

    return Vh_oneloop_corrected

def Vh_zeroT_atvev():
    
    (mHsq, lmbd, N, mu, vevh) = (0.286270859792, -0.6888422590446, 0.5, 2.1595, 3.08204)
    vevhc = tf.cast(vevh, tf.complex64)

    Vh_zeroT_atvevh = tf.math.real(0.5 * mHsq**2/lmbd + 0.5 * mHsq * vevhc**2 + 0.125 * lmbd * vevhc**4 + ( tf.math.xlogy(6 * mWsq_t(vevhc)**2, mWsq_t(vevhc)/mu**2) - 5 * mWsq_t(vevhc)**2 + tf.math.xlogy(3 * mZsq_t(vevhc)**2, mZsq_t(vevhc)/mu**2) - 2.5 * mZsq_t(vevhc)**2 + tf.math.xlogy(3 * mGsq(vevhc,0)**2, mGsq(vevhc,0)/mu**2) - 4.5 * mGsq(vevhc,0)**2 + tf.math.xlogy(mhsq(vevhc,0)**2, mhsq(vevhc,0)/mu**2) - 1.5 * mhsq(vevhc,0)**2 + tf.math.xlogy(2*N * mSsq(vevhc,0)**2, mSsq(vevhc,0)/mu**2) - 3*N * mSsq(vevhc,0)**2 - tf.math.xlogy(12 * mtsq(vevhc)**2, mtsq(vevhc)/mu**2) + 18 * mtsq(vevhc)**2 - tf.math.xlogy(12 * mbsq(vevhc)**2, mbsq(vevhc)/mu**2) + 18 * mbsq(vevhc)**2 ) / (64 * np.math.pi**2) )

    return Vh_zeroT_atvevh

def Vh_finiteT(phi, h, Temp):
    hf = tf.cast(h, tf.float32)
    phif = tf.cast(phi, tf.float32)
    T = Temp/80
    N = 0.5

    Vh_finiteT = (T/Aphi(phif))**4 * ( 4*JB(mWsq_t(hf)*(Aphi(phif)/T)**2) + 2*JB(mWsq_l(hf,T/Aphi(phif))*(Aphi(phif)/T)**2) + 2*JB(mZsq_t(hf)*(Aphi(phif)/T)**2) + JB(mZsq_l(hf,T/Aphi(phif))*(Aphi(phif)/T)**2) + JB(mphsq_l(hf,T/Aphi(phif))*(Aphi(phif)/T)**2) + 3*JB(mGsq(hf,T/Aphi(phif))*(Aphi(phif)/T)**2) + JB(mhsq(hf,T/Aphi(phif))*(Aphi(phif)/T)**2) + 2*N*JB(mSsq(hf,T/Aphi(phif))*(Aphi(phif)/T)**2) - 12*JF(mtsq(hf)*(Aphi(phif)/T)**2) - 12*JF(mbsq(hf)*(Aphi(phif)/T)**2) - 156.33975 ) / (2*np.math.pi**2)

    return Vh_finiteT

def V_h(phi, h, Temp):
    VhT_potential = Vh_finiteT(phi, h, Temp) #+ Vh_oneloop(phi, h, Temp) - Vh_zeroT_atvev()

    return VhT_potential

# Derivatives of the Higgs potential w.r.t. h
#---------------------------------------------

#def dVh_dh_oneloop(phi, h, Temp):
#    hc = tf.cast(h, tf.complex64)
#    phic = tf.cast(phi, tf.complex64)
#    T = Temp/80
#
#    (mHsq, lmbd, N, mu) = (0.286270859792, -0.6888422590446, 0.5, 2.1595)
#
#    dVh_oneloop = tf.math.real( mHsq * hc + 0.5 * lmbd * hc**3 + ( tf.math.xlogy(8*mWsq_t(hc)*mWsq_t_prime(hc), mWsq_t(hc)/mu**2) - (8/3)*mWsq_t(hc)*mWsq_t_prime(hc) + tf.math.xlogy(4*mWsq_l(hc,T/Aphi(phic))*mWsq_l_prime(hc,T/Aphi(phic)), mWsq_l(hc,T/Aphi(phic))/mu**2) - (4/3)*mWsq_l(hc,T/Aphi(phic))*mWsq_l_prime(hc,T/Aphi(phic)) +  tf.math.xlogy(4*mZsq_t(hc)*mZsq_t_prime(hc), mZsq_t(hc)/mu**2) - (4/3)*mZsq_t(hc)*mZsq_t_prime(hc) + tf.math.xlogy(2*mZsq_l(hc,T/Aphi(phic))*mZsq_l_prime(hc,T/Aphi(phic)), mZsq_l(hc,T/Aphi(phic))/mu**2) - (2/3)*mZsq_l(hc,T/Aphi(phic))*mZsq_l_prime(hc,T/Aphi(phic)) + tf.math.xlogy(2*mphsq_l(hc,T/Aphi(phic))*mphsq_l_prime(hc,T/Aphi(phic)), mphsq_l(hc,T/Aphi(phic))/mu**2) - (2/3)*mphsq_l(hc,T/Aphi(phic))*mphsq_l_prime(hc,T/Aphi(phic)) + tf.math.xlogy(6*mGsq(hc,T/Aphi(phic))*mGsq_prime(hc,T/Aphi(phic)), mGsq(hc,T/Aphi(phic))/mu**2) - 6*mGsq(hc,T/Aphi(phic))*mGsq_prime(hc,T/Aphi(phic)) + tf.math.xlogy(2*mhsq(hc,T/Aphi(phic))*mhsq_prime(hc,T/Aphi(phic)), mhsq(hc,T/Aphi(phic))/mu**2) - 2*mhsq(hc,T/Aphi(phic))*mhsq_prime(hc,T/Aphi(phic)) + tf.math.xlogy(4*N*mSsq(hc,T/Aphi(phic))*mSsq_prime(hc,T/Aphi(phic)), mSsq(hc,T/Aphi(phic))/mu**2) - 4*N*mSsq(hc,T/Aphi(phic))*mSsq_prime(hc,T/Aphi(phic)) - tf.math.xlogy(24*mtsq(hc)*mtsq_prime(hc), mtsq(hc)/mu**2) + 24*mtsq(hc)*mtsq_prime(hc) - tf.math.xlogy(24*mbsq(hc)*mbsq_prime(hc), mbsq(hc)/mu**2) + 24*mbsq(hc)*mbsq_prime(hc) ) / (64 * np.math.pi**2) )
#    
#    return dVh_oneloop
    
def dVh_dh_oneloop(phi, h, Temp):
    hc = tf.cast(h, tf.complex64)
    phic = tf.cast(phi, tf.complex64)
    T = Temp/80

    (mHsq, lmbd, N, mu) = (0.286270859792, -0.6888422590446, 0.5, 2.1595)

    dVh_oneloop = tf.math.real( mHsq * hc + 0.5 * lmbd * hc**3 + ( tf.math.xlogy(8*mWsq_t(hc)*mWsq_t_prime(hc), mWsq_t(hc)/mu**2) - (8/3)*mWsq_t(hc)*mWsq_t_prime(hc) + tf.math.xlogy(4*mWsq_l(hc,T/Aphi(phic))*mWsq_l_prime(hc,T/Aphi(phic)), mWsq_l(hc,T/Aphi(phic))/mu**2) - (4/3)*mWsq_l(hc,T/Aphi(phic))*mWsq_l_prime(hc,T/Aphi(phic)) +  tf.math.xlogy(4*mZsq_t(hc)*mZsq_t_prime(hc), mZsq_t(hc)/mu**2) - (4/3)*mZsq_t(hc)*mZsq_t_prime(hc) + tf.math.xlogy(2*mZsq_l(hc,T/Aphi(phic))*mZsq_l_prime(hc,T/Aphi(phic)), mZsq_l(hc,T/Aphi(phic))/mu**2) - (2/3)*mZsq_l(hc,T/Aphi(phic))*mZsq_l_prime(hc,T/Aphi(phic)) + tf.math.xlogy(2*mphsq_l(hc,T/Aphi(phic))*mphsq_l_prime(hc,T/Aphi(phic)), mphsq_l(hc,T/Aphi(phic))/mu**2) - (2/3)*mphsq_l(hc,T/Aphi(phic))*mphsq_l_prime(hc,T/Aphi(phic)) + tf.math.xlogy(4*N*mSsq(hc,T/Aphi(phic))*mSsq_prime(hc,T/Aphi(phic)), mSsq(hc,T/Aphi(phic))/mu**2) - 4*N*mSsq(hc,T/Aphi(phic))*mSsq_prime(hc,T/Aphi(phic)) - tf.math.xlogy(24*mtsq(hc)*mtsq_prime(hc), mtsq(hc)/mu**2) + 24*mtsq(hc)*mtsq_prime(hc) - tf.math.xlogy(24*mbsq(hc)*mbsq_prime(hc), mbsq(hc)/mu**2) + 24*mbsq(hc)*mbsq_prime(hc) ) / (64 * np.math.pi**2) )
    
    return dVh_oneloop

def dVh_dh_finiteT(phi, h, Temp):
    ha = tf.cast(h, tf.float32)
    phia = tf.cast(phi, tf.float32)

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(ha) 
        VhT = Vh_finiteT(phia, ha, Temp)
    return tape.gradient(VhT, ha)

def dVh_dh(phi, h, Temp):          
    return dVh_dh_finiteT(phi, h,Temp) + dVh_dh_oneloop(phi, h, Temp)


def d2Vh_dh2(phi, h, Temp):         
    ha = tf.cast(h, tf.float32)
    phia = tf.cast(phi, tf.float32)

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(ha) 
        dVhT = dVh_dh(phia, ha, Temp)
    return tape.gradient(dVhT, ha)

# Derivatives of the Higgs potential w.r.t. phi
#-------------------------------------------------

def dVh_dphi_oneloop(phi, h, Temp):
    hc = tf.cast(h, tf.complex64)
    phic = tf.cast(phi, tf.complex64)
    T = Temp/80

    (mHsq, lmbd, N, mu) = (0.286270859792, -0.6888422590446, 0.5, 2.1595)
    
    dVdphi_oneloop = tf.math.real( (T*Aphi_prime(phic) / Aphi(phic)**2) * ( -tf.math.xlogy(4*mWsq_l(hc,T/Aphi(phic))*mWsq_l_prime_T(hc,T/Aphi(phic)), mWsq_l(hc,T/Aphi(phic))/mu**2) + (4/3)*mWsq_l(hc,T/Aphi(phic))*mWsq_l_prime_T(hc,T/Aphi(phic)) - tf.math.xlogy(2*mZsq_l(hc,T/Aphi(phic))*mZsq_l_prime_T(hc,T/Aphi(phic)), mZsq_l(hc,T/Aphi(phic))/mu**2) + (2/3)*mZsq_l(hc,T/Aphi(phic))*mZsq_l_prime_T(hc,T/Aphi(phic)) - tf.math.xlogy(2*mphsq_l(hc,T/Aphi(phic))*mphsq_l_prime_T(hc,T/Aphi(phic)), mphsq_l(hc,T/Aphi(phic))/mu**2) + (2/3)*mphsq_l(hc,T/Aphi(phic))*mphsq_l_prime_T(hc,T/Aphi(phic)) - tf.math.xlogy(6*mGsq(hc,T/Aphi(phic))*mGsq_prime_T(hc,T/Aphi(phic)), mGsq(hc,T/Aphi(phic))/mu**2) + 6*mGsq(hc,T/Aphi(phic))*mGsq_prime_T(hc,T/Aphi(phic)) - tf.math.xlogy(2*mhsq(hc,T/Aphi(phic))*mhsq_prime_T(hc,T/Aphi(phic)), mhsq(hc,T/Aphi(phic))/mu**2) + 2*mhsq(hc,T/Aphi(phic))*mhsq_prime_T(hc,T/Aphi(phic)) - tf.math.xlogy(4*N*mSsq(hc,T/Aphi(phic))*mSsq_prime_T(hc,T/Aphi(phic)), mSsq(hc,T/Aphi(phic))/mu**2) + 4*N*mSsq(hc,T/Aphi(phic))*mSsq_prime_T(hc,T/Aphi(phic)) ) / (64 * np.math.pi**2) )
    
    return dVdphi_oneloop

def dVh_dphi_finiteT(phi, h, Temp):
    ha = tf.cast(h, tf.float32)
    phia = tf.cast(phi, tf.float32)

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(phia) 
        VhT = Vh_finiteT(phia, ha, Temp)
    return tape.gradient(VhT, phia)

def dVh_dphi(phi, h, Temp):          
    return dVh_dphi_finiteT(phi, h,Temp) #+  dVh_dphi_oneloop(phi, h, Temp)

def d2Vh_dphi2(phi, h, Temp):         
    ha = tf.cast(h, tf.float32)
    phia = tf.cast(phi, tf.float32)

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(phia) 
        dVhT = dVh_dh(phia, ha, Temp)
    return tape.gradient(dVhT, phia)

#==================================================
#   Solving the differential equation using Elvet 
#==================================================

full_domain1 = elvet.box((0.01, 100, 10000))

def eqnT(r, dep_var, first_deriv, second_deriv):
    phi, h = tf.reshape(dep_var[0], (1,)), tf.reshape(dep_var[1], (1,))
    dphi_dr, dh_dr = first_deriv[0,0], first_deriv[0,1]
    d2phi_dr2, d2h_dr2 = second_deriv[0,0,0], second_deriv[0,0,1] 
    
    return (
        d2phi_dr2 + (2/r)*dphi_dr - Vphi_prime(phi) - 4*Aphi(phi)**3 * Aphi_prime(phi)*V_h(phi,h,70) - Aphi(phi)**4 * dVh_dphi(phi,h,70) - Aphi(phi)*Aphi_prime(phi)*dh_dr*dh_dr ,
        (d2h_dr2 + (2/r)*dh_dr)/Aphi(phi)**2 + 2*dphi_dr*dh_dr*Aphi_prime(phi)/Aphi(phi)**3 - dVh_dh(phi,h,70),
    )
 

# boundary condition for only the first run, with guessed values of h(0), phi(0)

bcs1 = (
    elvet.BC(0.01, lambda r, dep_var, first_deriv, second_deriv: dep_var[0] - 1.8),
    elvet.BC(0.01, lambda r, dep_var, first_deriv, second_deriv: dep_var[1] - 2.7),
    elvet.BC(0.01, lambda r, dep_var, first_deriv, second_deriv: first_deriv[0,0]),
    elvet.BC(0.01, lambda r, dep_var, first_deriv, second_deriv: first_deriv[0,1]),
    elvet.BC(100, lambda r, dep_var, first_deriv, second_deriv: dep_var[0]),
    elvet.BC(100, lambda r, dep_var, first_deriv, second_deriv: dep_var[1]),
    elvet.BC(100, lambda r, dep_var, first_deriv, second_deriv: first_deriv[0,0]),
    elvet.BC(100, lambda r, dep_var, first_deriv, second_deriv: first_deriv[0,1]),
)

# boundary condition for subsequent runs
bcs2 = (
    elvet.BC(0.01, lambda r, dep_var, first_deriv, second_deriv: first_deriv[0,0]),
    elvet.BC(0.01, lambda r, dep_var, first_deriv, second_deriv: first_deriv[0,1]),
    elvet.BC(100, lambda r, dep_var, first_deriv, second_deriv: dep_var[0]),
    elvet.BC(100, lambda r, dep_var, first_deriv, second_deriv: dep_var[1]),
    elvet.BC(100, lambda r, dep_var, first_deriv, second_deriv: first_deriv[0,0]),
    elvet.BC(100, lambda r, dep_var, first_deriv, second_deriv: first_deriv[0,1])
)

model50x7 = elvet.nn(1, 50, 50, 50, 50, 50, 50, 50, 2)

#===========
#   Run1 
#===========

result_1 = elvet.solver(eqnT, bcs1, domain=full_domain1, model=model50x7, epochs=50000, verbose=True, metrics=elvet.utils.metrics.WatchLR(True), callbacks=(elvet.utils.callbacks.EarlyStopping(min_loss=1e-6), elvet.utils.callbacks.TerminateIf(NaN=True, strictly_increasing=False) ))

# result_1 = elvet.solver(eqnT, bcs1, domain=full_domain1, model=model50x7, epochs=50000, verbose=True, metrics=elvet.utils.metrics.WatchLR(True), callbacks=(elvet.utils.LRschedulers.ReduceLROnPlateau(check_every=10000, min_lr=1e-8, scale=0.9, min_improvement_rate=0.01, store_lr=True), elvet.utils.callbacks.EarlyStopping(min_loss=1e-6), elvet.utils.callbacks.TerminateIf(NaN=True, Inf=True, strictly_increasing=False) ))

result_1.save_model_weights("Weights/run1.h5", overwrite=True)

# visualizing the change in loss with each passing epoch

plt.plot(range(len(result_1.losses)), result_1.losses)
plt.yscale("log", base=10)
plt.ylabel("loss")
plt.xlabel("epoch")
plt.savefig("Figures/run1_loss_vs_epoch.jpg")
plt.close()

# plotting the profile and its derivatives

phi_h, dphi_dh, d2phi_d2h = result_1.derivatives()

predic_phi = phi_h.numpy()[:,0]
predic_h = phi_h.numpy()[:,1]

predic_dphi = dphi_dh.numpy()[:,0,0]
predic_dh = dphi_dh.numpy()[:,0,1]

predic_d2phi = d2phi_d2h.numpy()[:,0,0,0]
predic_d2h = d2phi_d2h.numpy()[:,0,0,1]


plt.plot(result_1.domain[:,0],predic_h)
plt.plot(result_1.domain[:,0],predic_phi)
plt.savefig("Figures/run1_profiles.jpg")
plt.close()


plt.plot(result_1.domain[:,0],predic_dh)
plt.plot(result_1.domain[:,0],predic_dphi)
plt.savefig("Figures/run1_slopes.jpg")
plt.close()


#===========
#   Run2 
#===========

result_2 = elvet.solver(eqnT, bcs2, domain=full_domain1, model=model50x7, metrics=elvet.utils.metrics.WatchLR(True), callbacks=(elvet.utils.LRschedulers.ReduceLROnPlateau(check_every=10000, min_lr=1e-8, scale=0.9, min_improvement_rate=0.01, store_lr=True), elvet.utils.callbacks.EarlyStopping(min_loss=1e-7), elvet.utils.callbacks.TerminateIf(NaN=True, strictly_increasing=False) ))
result_2.load_model_weights("Weights/run1.h5")

result_2.fit(epochs=5e4, verbose=True)

result_2.save_model_weights("Weights/run2.h5", overwrite=True)

# visualizing the change in loss with each passing epoch

plt.plot(range(len(result_2.losses)), result_2.losses)
plt.yscale("log", base=10)
plt.ylabel("loss")
plt.xlabel("epoch")
plt.savefig("Figures/run2_loss_vs_epoch.jpg")
plt.close()

# plotting the profile and its derivatives

phi_h, dphi_dh, d2phi_d2h = result_2.derivatives()

predic_phi = phi_h.numpy()[:,0]
predic_h = phi_h.numpy()[:,1]

predic_dphi = dphi_dh.numpy()[:,0,0]
predic_dh = dphi_dh.numpy()[:,0,1]

predic_d2phi = d2phi_d2h.numpy()[:,0,0,0]
predic_d2h = d2phi_d2h.numpy()[:,0,0,1]


plt.plot(result_2.domain[:,0],predic_h)
plt.plot(result_2.domain[:,0],predic_phi)
plt.savefig("Figures/run2_profiles.jpg")
plt.close()


plt.plot(result_2.domain[:,0],predic_dh)
plt.plot(result_2.domain[:,0],predic_dphi)
plt.savefig("Figures/run2_slopes.jpg")
plt.close()

#===========
#   Run3 
#===========

result_3 = elvet.solver(eqnT, bcs2, domain=full_domain1, model=model50x7, metrics=elvet.utils.metrics.WatchLR(True), callbacks=(elvet.utils.LRschedulers.ReduceLROnPlateau(check_every=10000, min_lr=1e-8, scale=0.7, min_improvement_rate=0.01, store_lr=True), elvet.utils.callbacks.EarlyStopping(min_loss=7e-8), elvet.utils.callbacks.TerminateIf(NaN=True, strictly_increasing=False) ))
result_3.load_model_weights("Weights/run2.h5")

result_3.fit(epochs=5e4, verbose=True)

result_3.save_model_weights("Weights/run3.h5", overwrite=True)

# visualizing the change in loss with each passing epoch

plt.plot(range(len(result_3.losses)), result_3.losses)
plt.yscale("log", base=10)
plt.ylabel("loss")
plt.xlabel("epoch")
plt.savefig("Figures/run3_loss_vs_epoch.jpg")
plt.close()

# plotting the profile and its derivatives

phi_h, dphi_dh, d2phi_d2h = result_3.derivatives()

predic_phi = phi_h.numpy()[:,0]
predic_h = phi_h.numpy()[:,1]

predic_dphi = dphi_dh.numpy()[:,0,0]
predic_dh = dphi_dh.numpy()[:,0,1]

predic_d2phi = d2phi_d2h.numpy()[:,0,0,0]
predic_d2h = d2phi_d2h.numpy()[:,0,0,1]


plt.plot(result_3.domain[:,0],predic_h)
plt.plot(result_3.domain[:,0],predic_phi)
plt.savefig("Figures/run3_profiles.jpg")
plt.close()


plt.plot(result_3.domain[:,0],predic_dh)
plt.plot(result_3.domain[:,0],predic_dphi)
plt.savefig("Figures/run3_slopes.jpg")
plt.close()

#========================
#   Saving the output
#========================

profile_h = pd.DataFrame(data=predic_h, index=None, columns=['h(r)'])
profile_h.to_csv('Profiles/h.csv')

profile_phi = pd.DataFrame(data=predic_phi, index=None, columns=['phi(r)'])
profile_phi.to_csv('Profiles/phi.csv')

deriv_profile_h = pd.DataFrame(data=predic_dh, index=None, columns=['dh(r)/dr'])
deriv_profile_h.to_csv('Profiles/dh.csv')

deriv_profile_phi = pd.DataFrame(data=predic_dphi, index=None, columns=['dphi(r)/dr'])
deriv_profile_phi.to_csv('Profiles/dphi.csv')

second_deriv_profile_h = pd.DataFrame(data=predic_d2h, index=None, columns=['d2h(r)/dr2'])
second_deriv_profile_h.to_csv('Profiles/d2h.csv')

second_deriv_profile_phi = pd.DataFrame(data=predic_d2phi, index=None, columns=['d2phi(r)/dr2'])
second_deriv_profile_phi.to_csv('Profiles/d2phi.csv')



