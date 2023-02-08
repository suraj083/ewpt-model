"""Created on Monday, February 6th 2023
   Author: Suraj Prakash
"""


# In this case we will only define a field class and the initialization that was happening inside the model class will happen outside
# 
sample_field_id_list = ['sm_higgs', 'bsm_scalar', 'w_boson_t', 'w_boson_l', 'z_boson_t', 'z_boson_l', 'photon_l' 't_quark', 'b_quark']
sample_param_dict = {'g1':1, 'g2':1, 'yt':1, 'yb':1, 'mHsq':1, 'mSsq':1, 'lmbd':1, 'lmbd_SH':1, 'lmbd_S':1}
#
#
#

class field:
    def __init__(self, id, param_dict):
        self.name = id
        self.params = param_dict

    def mass(self, id, h, T):
        if self.name == 'sm_higgs':
            pass