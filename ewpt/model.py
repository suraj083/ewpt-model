"""Created on Monday, February 6th 2023
   Author: Suraj Prakash
"""
import sys
import tensorflow as tf
import numpy as np
from ewpt.field import field
from ewpt.auxiliary import fill_params, reg_sq_root
from ewpt.fitted_functions import fitted_JB, fitted_JF, fitted_xlogx

params_sm = ['g1', 'g2', 'yt', 'yb', 'mHsq', 'lmbd', 'vevh']
params_bsm = ['g1', 'g2', 'yt', 'yb', 'mHsq', 'mSsq', 'lmbd', 'lmbd_SH', 'lmbd_S', 'N', 'vevh']
params_eft = ['g1', 'g2', 'yt', 'yb', 'mHsq', 'lmbd', 'vevh'] # add Wilson Coefficients here

class model:
   _default_field_list = ['sm_higgs', 'goldstone', 'bsm_scalar', 'w_boson_t', 'w_boson_l', 'z_boson_t', 'z_boson_l', 'photon_l', 't_quark', 'b_quark']

   def __init__(self, param_dict: dict, field_list: list = _default_field_list, bsm: bool = True, eft: bool = False):
      
      assert len(param_dict) > 0

      try:
         if (not eft):
            if bsm:
               assert set(param_dict.keys()).issubset(set(params_bsm))
               self.params = fill_params(params_bsm, param_dict)
            else:
               assert set(param_dict.keys()).issubset(set(params_sm))
               self.params = fill_params(params_sm, param_dict)

         else:
            if bsm:
               raise ValueError
            else:
               assert set(param_dict.keys()).issubset(set(params_eft))
               self.params = fill_params(params_eft, param_dict)

      except ValueError:
         print("Model cannot be initialized with both EFT and BSM parameters simultaneously set to true.")
         sys.exit()

      except AssertionError:
         print("Wrong or missing parameter.")
         sys.exit()

      self.is_bsm = bsm
      self.is_eft = eft

      assert set(field_list).issubset(set(self._default_field_list)), "Invalid field name in field_list"
      # assert len(field_list) > 0, "Empty list of fields"
      self.field_name_list = field_list

      self.field_object_list = [field(field_name=name, param_dict=self.params, bsm=self.is_bsm, eft=self.is_eft) for name in self.field_name_list]

 
   def __str__(self) -> str:
      pass


   def cw_potential(self, h, Temp, **renorm):
      hc = tf.cast(h, tf.complex64)
      vevhc = tf.cast(self.params['vevh'], tf.complex64)
      T = Temp/80.0
      
      potV = tf.math.real(0.5 * self.params['mHsq'] * (hc**2 - vevhc**2) + 0.125 * self.params['lmbd'] * (hc**4 - vevhc**4))

      frac = (3.0 / 2.0)

      if renorm['scheme'] not in ["MS-Bar", "On-shell"]:
         print("Unknown renormalization scheme entered. Expected 'MS-Bar' or 'On-shell'")
         
      elif renorm['scheme'] == "MS-Bar":
         mu = renorm['scale']

         for field_obj in self.field_object_list:
            if field_obj.name in ['w_boson_t', 'w_boson_l', 'z_boson_t', 'z_boson_l', 'photon_l']:
               frac = (5.0 / 6.0)

            potV = potV + tf.math.real(field_obj.get_dof() * ( (tf.math.xlogy(field_obj.mass_sq(hc,T)**2, field_obj.mass_sq(hc,T)/mu**2) - tf.math.xlogy(field_obj.mass_sq(vevhc,0)**2, field_obj.mass_sq(vevhc,0)/mu**2)) - frac*(field_obj.mass_sq(hc,T)**2 - field_obj.mass_sq(vevhc,0)**2) ) / (64*np.math.pi**2))

         return potV

      elif renorm['scheme'] == "On-shell":
         for field_obj in self.field_object_list:
            potV = potV + tf.math.real( field_obj.get_dof() * ( tf.math.xlogy(field_obj.mass_sq(hc,T)**2, field_obj.mass_sq(hc,T)/field_obj.mass_sq(vevhc,T)) - frac*field_obj.mass_sq(hc,T)**2 + 2*field_obj.mass_sq(hc,T)*field_obj.mass_sq(vevhc,T) ) / (64*np.math.pi**2)) # check the cut-off reg Mathematica implementation

         return potV
         
         
   def cw_potential_deriv(self, h, Temp, **renorm):
      #hc = tf.cast(h, tf.complex64)
      vevhc = tf.cast(self.params['vevh'], tf.complex64)
      T = Temp/80.0
      
      dpotV = self.params['mHsq'] * h + 0.5 * self.params['lmbd'] * h**3 

      if renorm['scheme'] not in ["MS-Bar", "On-shell"]:
         print("Unknown renormalization scheme entered. Expected 'MS-Bar' or 'On-shell'")

      elif renorm['scheme'] == "MS-Bar":
         mu = renorm['scale']
         const_factor = np.log(mu**2) + 1.0

         for field_obj in self.field_object_list:
            if field_obj.name in ['w_boson_t', 'w_boson_l', 'z_boson_t', 'z_boson_l', 'photon_l']:
               const_factor = np.log(mu**2) + (1.0 / 3.0)
               
            dpotV = dpotV + 2 * field_obj.get_dof() * field_obj.mass_sq_field_deriv(h,T) * ( fitted_xlogx(field_obj.mass_sq(h,T)) - field_obj.mass_sq(h,T) * const_factor ) / (64 * np.math.pi**2)
               
         return dpotV
               
      elif renorm['scheme'] == 'On-shell':
         for field_obj in self.field_object_list:
            dpotV = dpotV + 2 * field_obj.get_dof() * field_obj.mass_sq_field_deriv(h,T) * ( fitted_xlogx(field_obj.mass_sq(h,T)) - field_obj.mass_sq(h,T) * (1 + tf.math.real(tf.math.log(field_obj.mass_sq(vevhc,T)))) + 2*field_obj.mass_sq(self.params['vevh'],T) )

         return dpotV


   # defining the finite-temperature potential and its derivative(s)
   def finite_T_potential(self, h, Temp, large_T_approx: bool):
      hf = tf.cast(h, tf.float32)
      hc = tf.cast(hf, tf.complex64)

      T = Temp/80.0

      total = 0.0

      if (not large_T_approx):
         for field_obj in self.field_object_list:
            if field_obj.name in ['b_quark', 't_quark']:
               total += field_obj.get_dof() * fitted_JF(field_obj.mass_sq(hf, T)/T**2)
            else:
               total += field_obj.get_dof() * fitted_JB(field_obj.mass_sq(hf, T)/T**2)

         return (T**4) * (total - 156.33975) / (2*np.math.pi**2)

      else:  # cross-check for errors
         a_b = np.math.exp(5.4076)
         a_f = np.math.exp(2.6351)

         for field_obj in self.field_object_list:
            if field_obj.name in ['b_quark', 't_quark']:
               total += field_obj.get_dof() * ( (7 * np.math.pi**4 / 360) - (np.math.pi**2 / 24)*field_obj.mass_sq(hf, T) - (1/32) * tf.math.real(tf.math.xlogy(field_obj.mass_sq(hc,T)**2, field_obj.mass_sq(hc,T)/a_f)) )
            else:
               total += field_obj.get_dof() * ( (np.math.pi**2 / 12)*field_obj.mass_sq(hf, T) - (np.math.pi / 6)*reg_sq_root(field_obj.mass_sq(hf, T))**3 - (np.math.pi**4 / 45) - (1/32) * tf.math.real(tf.math.xlogy(field_obj.mass_sq(hc,T)**2, field_obj.mass_sq(hc,T)/a_b)) )

         return (T**4) * total / (2*np.math.pi**2) # may require the regulated square root helper function

   def finite_T_potential_deriv(self, h, Temp, large_T_approx: bool):
      ha = tf.cast(h, tf.float32)
      
      if (not large_T_approx):

         with tf.GradientTape(persistent=True) as tape:
            tape.watch(ha) 
            VhT = self.finite_T_potential(ha, Temp, large_T_approx)
         
         return tape.gradient(VhT, ha)
      
      else: # cross-check for errors
         T = Temp / 80.0
         total = 0.0
         a_b = np.math.exp(5.4076)
         a_f = np.math.exp(2.6351)

         for field_obj in self.field_object_list:
            if field_obj.name in ['b_quark', 't_quark']:
               total += field_obj.get_dof() * ( -(np.math.pi**2 / 24) - (1/16) * (fitted_xlogx(field_obj.mass_sq(ha,T)) - field_obj.mass_sq(ha,T) * 2.6351) - (1/32) * field_obj.mass_sq(ha,T) )
            else:
               total += field_obj.get_dof() * ( (np.math.pi**2 / 12) - (np.math.pi / 4)*reg_sq_root(field_obj.mass_sq(ha, T)) - (1/16) * (fitted_xlogx(field_obj.mass_sq(ha,T)) - field_obj.mass_sq(ha,T) * 5.4076) - (1/32) * field_obj.mass_sq(ha,T) )

         return (T**4) * total / (2*np.math.pi**2) # may require the regulated square root helper function


   # total potential and its derivative(s)

   def total_potential(self, h, Temp, large_T_approx: bool, **renorm):
      return self.cw_potential(h, Temp, **renorm) + self.finite_T_potential(h, Temp, large_T_approx)

   def total_potential_deriv(self, h, Temp, large_T_approx: bool, **renorm):
      return self.cw_potential_deriv(h, Temp, **renorm) + self.finite_T_potential_deriv(h, Temp, large_T_approx)




      
