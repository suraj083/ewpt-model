"""Created on Monday, February 6th 2023
   Author: Suraj Prakash
"""

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

class model:
   default_field_list = ['sm_higgs', 'goldstone', 'bsm_scalar', 'w_boson_t', 'w_boson_l', 'z_boson_t', 'z_boson_l', 'photon_l', 't_quark', 'b_quark']
   all_params = ['g1', 'g2', 'yt', 'yb', 'mHsq', 'mSsq', 'lmbd', 'lmbd_SH', 'lmbd_S', 'N']

   def __init__(self, param_dict: dict, field_list: list = default_field_list, bsm: bool = True, eft: bool = False, large_T_approx: bool = False):
      
      assert set(field_list).issubset(set(self.default_field_list)), "Invalid field name in field_list"
      self.fields = field_list

      assert set(param_dict.keys()).issubset(set(self.all_params))
      self.params = param_dict
      
      self.is_bsm = bsm
      self.is_eft = eft
      self.large_T_approx = large_T_approx
