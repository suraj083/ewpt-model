"""Created on Monday, February 6th 2023
   Author: Suraj Prakash
"""

import tensorflow as tf

# In this case we will only define a field class and the initialization that was happening inside the model class will happen outside
# 
sample_field_id_list = ['sm_higgs', 'goldstone', 'bsm_scalar', 'w_boson_t', 'w_boson_l', 'z_boson_t', 'z_boson_l', 'photon_l' 't_quark', 'b_quark']
sample_param_dict = {'g1':0.4632, 'g2':0.6485, 'yt':0.92849, 'yb':0.0167, 'mHsq':-1.19109, 'mSsq':0.0625, 'lmbd':0.2582, 'lmbd_SH':0.85, 'lmbd_S':0.1, 'N':0.5}
#
#
#

# In a later iteration - perhaps individual fields can be defined as members of an enum, also it may be a good idea to define different subclasses for scalar, fermions and vector bosons.
class field:
    def __init__(self, id: str, param_dict: dict, bsm: bool, eft: bool):
        """ Initialize an instance of the field class.

            Parameters
            ----------
            id : str
            param_dict :  dict
            bsm : bool
            eft : bool  
        
        """

        assert id in ['sm_higgs', 'goldstone', 'bsm_scalar', 'w_boson_t', 'w_boson_l', 'z_boson_t', 'z_boson_l', 'photon_l' 't_quark', 'b_quark'], "invalid field name"
        
        self.name = id
        self.params = param_dict
        self.is_bsm = bsm
        self.is_eft = eft

    def _delta(self, h,T):
        """ private helper function for defining longitudinal Z and photon thermal masses """
        return (1/60) * tf.math.sqrt(-2640 * T**2 * self.params['g1']**2 * self.params['g2']**2 * (11 * T**2 + 3 * h**2) + (22 * T**2 + 3 * h**2)**2 * (5 * self.params['g2']**2 + 3 * self.params['g1']**2)**2)

    def _delta_field_deriv(self, h, T):
        """ private helper function for defining derivatives of longitudinal Z and photon thermal masses """
        pass

    def _delta_temperature_deriv(self, h, T):
        """ private helper function for defining derivatives of longitudinal Z and photon thermal masses """
        pass

    def mass(self, h, T):
        """ Definition of thermal masses of individual fields
            
            Parameters
            ----------
            h : float
            T : float
            
            Returns
            -------
            float
        """

        # reorganise this further, put is_eft == false as the first condition, put is_bsm == true or false within the match cases for sm_higgs, goldstone and bsm_scalar

        if ((not self.is_bsm) and (not self.is_eft)):    # the pure SM case
            match self.name:
                case 'sm_higgs':
                    return self.params['mHsq'] + 1.5 * self.params['lmbd'] * h**2 + ((3/80) * self.params['g1']**2 + (3/16) * self.params['g2']**2 + (1/4) * self.params['yt']**2 ) * T**2
                case 'goldstone':
                    return self.params['mHsq'] + 0.5 * self.params['lmbd'] * h**2 + ((3/80) * self.params['g1']**2 + (3/16) * self.params['g2']**2 + (1/4) * self.params['yt']**2) * T**2
                case 'w_boson_t':
                    return 0.25 * self.params['g2']**2 * h**2
                case 'w_boson_l':
                    return 0.25 * self.params['g2']**2 * h**2 + (11/6) * self.params['g2']**2 * T**2
                case 'z_boson_t':
                    return 0.25 * (0.6 * self.params['g1']**2 + self.params['g2']**2) * h**2
                case 'z_boson_l':
                    return 0.5 * (0.25 * (0.6 * self.params['g1']**2 + self.params['g2']**2) * h**2 + (11/6) * self.params['g2']**2 * T**2 + (11/10) * self.params['g1']**2 * T**2 + self._delta(h,T))
                case 'photon_l':
                    return 0.5 * (0.25 * (0.6 * self.params['g1']**2 + self.params['g2']**2) * h**2 + (11/6) * self.params['g2']**2 * T**2 + (11/10) * self.params['g1']**2 * T**2 - self._delta(h,T))
                case 't_quark':
                    return 0.5 * self.params['yt']**2 * h**2 
                case 'b_quark':
                    return 0.5 * self.params['yb']**2 * h**2

        elif (self.is_bsm and (not self.is_eft)):    # the case of SM extended by a complex scalar N-plet 
            match self.name:
                case 'sm_higgs':
                    return self.params['mHsq'] + 1.5 * self.params['lmbd'] * h**2 + ((3/80) * self.params['g1']**2 + (3/16) * self.params['g2']**2 + (1/4) * self.params['yt']**2 +(self.params['N']*self.params['lmbd_SH'] / 12)) * T**2
                case 'goldstone':
                    return self.params['mHsq'] + 0.5 * self.params['lmbd'] * h**2 + ((3/80) * self.params['g1']**2 + (3/16) * self.params['g2']**2 + (1/4) * self.params['yt']**2 + (self.params['N']*self.params['lmbd_SH'] / 12)) * T**2
                case 'w_boson_t':
                    return 0.25 * self.params['g2']**2 * h**2
                case 'w_boson_l':
                    return 0.25 * self.params['g2']**2 * h**2 + (11/6) * self.params['g2']**2 * T**2
                case 'z_boson_t':
                    return 0.25 * (0.6 * self.params['g1']**2 + self.params['g2']**2) * h**2
                case 'z_boson_l':
                    return 0.5 * (0.25 * (0.6 * self.params['g1']**2 + self.params['g2']**2) * h**2 + (11/6) * self.params['g2']**2 * T**2 + (11/10) * self.params['g1']**2 * T**2 + self._delta(h,T))
                case 'photon_l':
                    return 0.5 * (0.25 * (0.6 * self.params['g1']**2 + self.params['g2']**2) * h**2 + (11/6) * self.params['g2']**2 * T**2 + (11/10) * self.params['g1']**2 * T**2 - self._delta(h,T))
                case 't_quark':
                    return 0.5 * self.params['yt']**2 * h**2 
                case 'b_quark':
                    return 0.5 * self.params['yb']**2 * h**2
                case 'bsm_scalar':
                    return self.params['mSsq'] + 0.5 * self.params['lmbd_SH'] * h**2 + ((self.params['lmbd_S'] / 12) * (self.params['N']+1) + (self.params['lmbd_SH'] / 6)) * T**2

        elif ((not self.is_bsm) and self.is_eft):    # the case of SM extended by dimension-6 SMEFT operators
            match self.name:
                case 'sm_higgs':
                    pass
                case 'goldstone':
                    pass
                case 'w_boson_t':
                    pass
                case 'w_boson_l':
                    pass
                case 'z_boson_t':
                    pass
                case 'z_boson_l':
                    pass
                case 'photon_l':
                    pass
                case 't_quark':
                    pass
                case 'b_quark':
                    pass

    def mass_field_deriv(self, h, T):
        """ Derivative of thermal masses of individual fields w.r.t. the SM Higgs as a parameter
            
            Parameters
            ----------
            h : float
            T : float
            
            Returns
            -------
            float
        """

        if ((not self.is_bsm) and (not self.is_eft)):    # the pure SM case
            match self.name:
                case 'sm_higgs':
                    pass
                case 'goldstone':
                    pass
                case 'w_boson_t':
                    pass
                case 'w_boson_l':
                    pass
                case 'z_boson_t':
                    pass
                case 'z_boson_l':
                    pass
                case 'photon_l':
                    pass
                case 't_quark':
                    pass
                case 'b_quark':
                    pass

        elif (self.is_bsm and (not self.is_eft)):    # the case of SM extended by a complex scalar N-plet 
            match self.name:
                case 'sm_higgs':
                    pass
                case 'goldstone':
                    pass
                case 'w_boson_t':
                    pass
                case 'w_boson_l':
                    pass
                case 'z_boson_t':
                    pass
                case 'z_boson_l':
                    pass
                case 'photon_l':
                    pass
                case 't_quark':
                    pass
                case 'b_quark':
                    pass
                case 'bsm_scalar':
                    pass

        elif ((not self.is_bsm) and self.is_eft):    # the case of SM extended by dimension-6 SMEFT operators
            match self.name:
                case 'sm_higgs':
                    pass
                case 'goldstone':
                    pass
                case 'w_boson_t':
                    pass
                case 'w_boson_l':
                    pass
                case 'z_boson_t':
                    pass
                case 'z_boson_l':
                    pass
                case 'photon_l':
                    pass
                case 't_quark':
                    pass
                case 'b_quark':
                    pass

    def mass_temperature_deriv(self, h, T):
        """ Derivative of thermal masses of individual fields w.r.t. the temperature parameter
            
            Parameters
            ----------
            h : float
            T : float
            
            Returns
            -------
            float
        """

        if ((not self.is_bsm) and (not self.is_eft)):    # the pure SM case
            match self.name:
                case 'sm_higgs':
                    pass
                case 'goldstone':
                    pass
                case 'w_boson_t':
                    pass
                case 'w_boson_l':
                    pass
                case 'z_boson_t':
                    pass
                case 'z_boson_l':
                    pass
                case 'photon_l':
                    pass
                case 't_quark':
                    pass
                case 'b_quark':
                    pass

        elif (self.is_bsm and (not self.is_eft)):    # the case of SM extended by a complex scalar N-plet 
            match self.name:
                case 'sm_higgs':
                    pass
                case 'goldstone':
                    pass
                case 'w_boson_t':
                    pass
                case 'w_boson_l':
                    pass
                case 'z_boson_t':
                    pass
                case 'z_boson_l':
                    pass
                case 'photon_l':
                    pass
                case 't_quark':
                    pass
                case 'b_quark':
                    pass
                case 'bsm_scalar':
                    pass

        elif ((not self.is_bsm) and self.is_eft):    # the case of SM extended by dimension-6 SMEFT operators
            match self.name:
                case 'sm_higgs':
                    pass
                case 'goldstone':
                    pass
                case 'w_boson_t':
                    pass
                case 'w_boson_l':
                    pass
                case 'z_boson_t':
                    pass
                case 'z_boson_l':
                    pass
                case 'photon_l':
                    pass
                case 't_quark':
                    pass
                case 'b_quark':
                    pass            

    def get_degrees_of_freedom(self):
        """ Returns the number of degrees of freedom for individual fields"""

        match self.name:
            case 'sm_higgs':
                return 1
            case 'goldstone':
                return 3
            case 'w_boson_t':
                return 4
            case 'w_boson_l':
                return 2
            case 'z_boson_t':
                return 2
            case 'z_boson_l':
                return 1
            case 'photon_l':
                return 1
            case 't_quark':
                return -12
            case 'b_quark':
                return -12
            case 'bsm_scalar':
                return 2*self.params['N']