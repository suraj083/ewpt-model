"""Created on Monday, February 6th 2023
   Author: Suraj Prakash
"""
from field import field
# Here we will create a model class whose subclasses would correspond to the various example cases - sm, bsm, sm+dilaton, bsm + dilaton,
# 
#
# sample_field_id_list = ['sm_higgs', 'bsm_scalar', 'w_boson', 'z_boson', 't_quark', 'b_quark']
# sample_param_dict = ['g1':1, 'g2':1, 'yt':1, 'yb':1, 'mHsq':1, 'mSsq':1, 'lmbd':1, 'lmbd_SH':1, 'lmbd_S':1,  ]
#  
# class model(field_list = 'list-of-field-id-strings', param_dict = 'dict-of-param-name-value-pairs', bsm: bool, eft: bool, !!! FOR LATER high-T-exp: bool):
#     self.params = param_dict
#     def set_params(self, param_dict):
#        self.params = param_dict
#--------------------------------------------------------------------  
#                       !! NO LONGER NEEDED !!
#--------------------------------------------------------------------
#    ----first construct subdicts of params corresponding to different fields----  
#    ---for now assume all these dict_keys are there---    
#
#    h_params = {'mHsq': self.params['mHsq'], 'lmbd': self.params['lmbd'], 'g1': self.params['g1'], 'g2': self.params['g2'], 'yt': self.params['yt'], 'lmbd_SH': self.params['lmbd_SH'], 'N': self.params['N'] }
#    s_params = {'mSsq': self.params['mSsq'], 'lmbd_SH': self.params['lmbd_SH'], 'lmbd_S': self.params['lmbd_S'], 'N': self.params['N']}
#    z_params = {'g1': self.params['g1'], 'g2': self.params['g2']}
#    w_params = {'g2': self.params['g2']}
#    t_params = {'yt': self.params['yt']}
#    b_params = {'yb': self.params['yb']}
# -------------------------------------------------------------------   

params_sm = ['g1', 'g2', 'yt', 'yb', 'mHsq', 'lmbd']
params_bsm = ['g1', 'g2', 'yt', 'yb', 'mHsq', 'mSsq', 'lmbd', 'lmbd_SH', 'lmbd_S', 'N']
params_eft = ['g1', 'g2', 'yt', 'yb', 'mHsq', 'lmbd'] # add Wilson Coefficients here

class model:
   _default_field_list = ['sm_higgs', 'goldstone', 'bsm_scalar', 'w_boson_t', 'w_boson_l', 'z_boson_t', 'z_boson_l', 'photon_l', 't_quark', 'b_quark']

   def __init__(self, param_dict: dict, field_list: list = _default_field_list, bsm: bool = True, eft: bool = False, large_T_approx: bool = False):
      
      try:
         if (not eft):
            if bsm:
               assert set(param_dict.keys()).__eq__(set(params_bsm))
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

      assert set(field_list).issubset(set(self._default_field_list)), "Invalid field name in field_list"
      self.field_name_list = field_list

      self.is_bsm = bsm
      self.is_eft = eft
      self.large_T_approx = large_T_approx

      self.field_object_list = [field(field_name=name, param_dict=self.params, bsm=self.is_bsm, eft=self.is_eft) for name in self.field_name_list]
 
   def __str__(self) -> str:
      pass
      
