"""Created on Monday, February 6th 2023
   Author: Suraj Prakash
"""
from field import field

params_sm = ['g1', 'g2', 'yt', 'yb', 'mHsq', 'lmbd']
params_bsm = ['g1', 'g2', 'yt', 'yb', 'mHsq', 'mSsq', 'lmbd', 'lmbd_SH', 'lmbd_S', 'N']
params_eft = ['g1', 'g2', 'yt', 'yb', 'mHsq', 'lmbd'] # add Wilson Coefficients here

class model:
   _default_field_list = ['sm_higgs', 'goldstone', 'bsm_scalar', 'w_boson_t', 'w_boson_l', 'z_boson_t', 'z_boson_l', 'photon_l', 't_quark', 'b_quark']

   def __init__(self, param_dict: dict, field_list: list = _default_field_list, bsm: bool = True, eft: bool = False, large_T_approx: bool = False):
      
      try:
         if (not eft):
            if bsm:
               assert set(param_dict.keys()).__eq__(set(params_bsm)) # perhaps this statement along with adding zero-valued keys to the dictionary can be combined together in a function
            else:
               assert set(param_dict.keys()).__eq__(set(params_sm))

         else:
            if bsm:
               raise ValueError
            else:
               assert set(param_dict.keys()).__eq__(set(params_eft))

      except ValueError:
         print("Model cannot be initialized with both EFT and BSM parameters simultaneously set to true.")

      except AssertionError:
         print("Wrong or missing parameter.")

      self.params = param_dict
      self.is_bsm = bsm
      self.is_eft = eft
      self.large_T_approx = large_T_approx

      assert set(field_list).issubset(set(self._default_field_list)), "Invalid field name in field_list"
      assert len(field_list) > 0, "Empty list of fields"
      self.field_name_list = field_list



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
   

   # assemling the Coleman-Weinberg one-loop potential and its derivative(s)

   # def _cw_potential_regular(self, h, T, scheme: str):
   #    try:
   #       if scheme == 'MS-Bar':
   #          pass
   #       elif scheme == 'On-shell':
   #          pass
   #       else:
   #          raise ValueError

   #    except ValueError:
   #       print("Unknown renormalization scheme entered. Expected 'MS-Bar' or 'On-shell'")

   # def _cw_potential_unusual(self, h, T, scheme: str):
   #    try:
   #       if scheme == 'MS-Bar':
   #          pass
   #       elif scheme == 'On-shell':
   #          pass
   #       else:
   #          raise ValueError

   #    except ValueError:
   #       print("Unknown renormalization scheme entered. Expected 'MS-Bar' or 'On-shell'")

   def cw_potential(self, h, T, scheme: str):
      # return self._cw_potential_regular(h, T, scheme) + self._cw_potential_unusual(h, T, scheme)
      try:
         if scheme == 'MS-Bar':
            pass
         elif scheme == 'On-shell':
            pass
         else:
            raise ValueError

      except ValueError:
         print("Unknown renormalization scheme entered. Expected 'MS-Bar' or 'On-shell'")

   
   # def _cw_potential_deriv_regular(self, h, T, scheme: str):
   #    try:
   #       if scheme == 'MS-Bar':
   #          pass
   #       elif scheme == 'On-shell':
   #          pass
   #       else:
   #          raise ValueError

   #    except ValueError:
   #       print("Unknown renormalization scheme entered. Expected 'MS-Bar' or 'On-shell'")

   # def _cw_potential_deriv_unusual(self, h, T, scheme: str):
   #    try:
   #       if scheme == 'MS-Bar':
   #          pass
   #       elif scheme == 'On-shell':
   #          pass
   #       else:
   #          raise ValueError

   #    except ValueError:
   #       print("Unknown renormalization scheme entered. Expected 'MS-Bar' or 'On-shell'")

   def cw_potential_deriv(self, h, T, scheme: str):
      #return self._cw_potential_deriv_regular(h, T, scheme) + self._cw_potential_deriv_unusual(h, T, scheme)
      try:
         if scheme == 'MS-Bar':
            pass
         elif scheme == 'On-shell':
            pass
         else:
            raise ValueError

      except ValueError:
         print("Unknown renormalization scheme entered. Expected 'MS-Bar' or 'On-shell'")

   # defining the finite-temperature potential and its derivative(s)
   def finite_T_potential(self, h, T):
      pass

   def finite_T_potential_deriv(self, h, T):
      pass

   # total potential and its derivative(s)

   def total_potential(self, h, T, scheme: str):
      return self.cw_potential(h, T, scheme) + self.finite_T_potential(h, T)

   def total_potential_deriv(self, h, T, scheme: str):
      return self.cw_potential_deriv(h, T, scheme) + self.finite_T_potential_deriv(h, T)

      
