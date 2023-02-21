"""Created on Monday, February 6th 2023
   Author: Suraj Prakash
"""
import tensorflow as tf
import numpy as np
from field import field
from auxiliary import fill_params
from fitted_functions import fitted_JB, fitted_JF, fitted_xlogx

params_sm = ['g1', 'g2', 'yt', 'yb', 'mHsq', 'lmbd', 'vevh']
params_bsm = ['g1', 'g2', 'yt', 'yb', 'mHsq', 'mSsq', 'lmbd', 'lmbd_SH', 'lmbd_S', 'N', 'vevh']
params_eft = ['g1', 'g2', 'yt', 'yb', 'mHsq', 'lmbd', 'vevh'] # add Wilson Coefficients here

class model:
   _default_field_list = ['sm_higgs', 'goldstone', 'bsm_scalar', 'w_boson_t', 'w_boson_l', 'z_boson_t', 'z_boson_l', 'photon_l', 't_quark', 'b_quark']

   def __init__(self, param_dict: dict, field_list: list = _default_field_list, bsm: bool = True, eft: bool = False):
      
      assert len(param_dict) > 0, "Provide at least"

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

      except AssertionError:
         print("Wrong or missing parameter.")

      self.params = param_dict
      self.is_bsm = bsm
      self.is_eft = eft

      assert set(field_list).issubset(set(self._default_field_list)), "Invalid field name in field_list"
      # assert len(field_list) > 0, "Empty list of fields"
      self.field_name_list = field_list

      self.field_object_list = [field(field_name=name, param_dict=self.params, bsm=self.is_bsm, eft=self.is_eft) for name in self.field_name_list]

      # # segregating SM Higgs and goldstones from all other fields
      # self._scalar_list = ['sm_higgs', 'goldstone']
      # self._nonscalar_list = list(set(self.field_name_list).difference(set(self._scalar_list)))

      # self._scalar_object_list = [field(field_name=name, param_dict=self.params, bsm=self.is_bsm, eft=self.is_eft) for name in self._scalar_list]
      # self._nonscalar_object_list = [field(field_name=name, param_dict=self.params, bsm=self.is_bsm, eft=self.is_eft) for name in self._nonscalar_list]

      # # segregating fermions and bosons
      # self._fermion_list = ['t_quark', 'b_quark']
      # self._boson_list = list(set(self.field_name_list).difference(set(self._fermion_list)))

      # self._fermion_object_list = [field(field_name=name, param_dict=self.params, bsm=self.is_bsm, eft=self.is_eft) for name in self._fermion_list]
      # self._boson_object_list = [field(field_name=name, param_dict=self.params, bsm=self.is_bsm, eft=self.is_eft) for name in self._boson_list]

 
   def __str__(self) -> str:
      pass


   def cw_potential(self, h, Temp, **renorm):
      # return self._cw_potential_regular(h, T, scheme) + self._cw_potential_unusual(h, T, scheme)
      hc = tf.cast(h, tf.complex64)
      vevhc = tf.cast(self.params['vevh'], tf.complex64)
      T = Temp/80
      
      try:
         if renorm['scheme'] == 'MS-Bar':
            mu = renorm['scale']
            Vh_tree = tf.math.real(0.5 * self.params['mHsq'] * (hc**2 - vevhc**2) + 0.125 * self.params['lmbd'] * (hc**4 - vevhc**4))



         elif renorm['scheme'] == 'On-shell':
            pass
         
         else:
            raise ValueError

      except ValueError:
         print("Unknown renormalization scheme entered. Expected 'MS-Bar' or 'On-shell'")
         

   def cw_potential_deriv(self, h, Temp, **renorm):
      try:
         if renorm['scheme'] == 'MS-Bar':
            mu = renorm['scale']
         
         elif renorm['scheme'] == 'On-shell':
            pass

         else:
            raise ValueError

      except ValueError:
         print("Unknown renormalization scheme entered. Expected 'MS-Bar' or 'On-shell'")

   # defining the finite-temperature potential and its derivative(s)
   def finite_T_potential(self, h, Temp, large_T_approx: bool):
      hf = tf.cast(h, tf.float32)
      T = Temp/80

      total = -156.33975

      for field_object in self.field_object_list:
         if field_object.name in ['b_quark', 't_quark']:
            total += field_object.get_degrees_of_freedom() * fitted_JF(field_object.mass_sq(hf, T)/T**2)
         else:
            total += field_object.get_degrees_of_freedom() * fitted_JB(field_object.mass_sq(hf, T)/T**2)

      if (not large_T_approx):
         return (T**4) * total / (2*np.math.pi**2)

      else:
         return 0

   def finite_T_potential_deriv(self, h, Temp, large_T_approx: bool):
      if (not large_T_approx):
         ha = tf.cast(h, tf.float32)

         with tf.GradientTape(persistent=True) as tape:
            tape.watch(ha) 
            VhT = self.finite_T_potential(ha, Temp, large_T_approx)
         
         return tape.gradient(VhT, ha)
      
      else:
         return 0 # may require the regulated square root helper function


   # total potential and its derivative(s)

   def total_potential(self, h, Temp, large_T_approx: bool, **renorm):
      return self.cw_potential(h, Temp, **renorm) + self.finite_T_potential(h, Temp, large_T_approx)

   def total_potential_deriv(self, h, Temp, large_T_approx: bool, **renorm):
      return self.cw_potential_deriv(h, Temp, **renorm) + self.finite_T_potential_deriv(h, Temp, large_T_approx)




      
