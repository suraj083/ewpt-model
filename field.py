"""Created on Monday, February 6th 2023
   Author: Suraj Prakash
"""


# In this case we will only define a field class and the initialization that was happening inside the model class will happen outside
# 
sample_field_id_list = ['sm_higgs', 'goldstone', 'bsm_scalar', 'w_boson_t', 'w_boson_l', 'z_boson_t', 'z_boson_l', 'photon_l' 't_quark', 'b_quark']
sample_param_dict = {'g1':1, 'g2':1, 'yt':1, 'yb':1, 'mHsq':1, 'mSsq':1, 'lmbd':1, 'lmbd_SH':1, 'lmbd_S':1, 'N':1}
#
#
#

class field:
    def __init__(self, id: str, param_dict: dict, eft: bool, bsm: bool):
        self.name = id
        self.params = param_dict
        self.is_bsm = bsm
        self.is_eft = eft

    def mass(self, id, h, T):
        if ((not self.is_bsm) and (not self.is_eft)):
            if self.name == 'sm_higgs':
                return self.params['mHsq'] + 1.5 * self.params['lmbd'] * h**2 + ((3/80) * self.params['g1']**2 + (3/16) * self.params['g2']**2 + (1/4) * self.params['yt']**2 +(self.params['N']*self.params['lmbd_SH'] / 12)) * T**2

            


        if (self.is_bsm and (not self.is_eft)):
            if self.name == 'sm_higgs':
                return self.params['mHsq'] + 1.5 * self.params['lmbd'] * h**2 + ((3/80) * self.params['g1']**2 + (3/16) * self.params['g2']**2 + (1/4) * self.params['yt']**2 ) * T**2