"""Created on Monday, February 6th 2023
   Author: Suraj Prakash
"""

import tensorflow as tf

# In a later iteration - perhaps individual fields can be defined as members of an enum, also it may be a good idea to define different subclasses for scalar, fermions and vector bosons.
class field:
    
    def __init__(self, field_name: str, param_dict: dict, bsm: bool, eft: bool):
        """ Initialize an instance of the field class.

            Parameters
            ----------
            field_name : str
            param_dict :  dict
            bsm : bool
            eft : bool  
        
        """

        assert field_name in ['sm_higgs', 'goldstone', 'bsm_scalar', 'w_boson_t', 'w_boson_l', 'z_boson_t', 'z_boson_l', 'photon_l', 't_quark', 'b_quark'], "invalid field name"
        
        self.name = field_name

        self.params = param_dict
        self.is_bsm = bsm
        self.is_eft = eft
    
    def __str__(self) -> str:
        pass

    def _delta(self, h,T):
        """ private helper function for defining longitudinal Z and photon thermal masses """
        return (1/60) * tf.math.sqrt(-2640 * T**2 * self.params['g1']**2 * self.params['g2']**2 * (11 * T**2 + 3 * h**2) + (22 * T**2 + 3 * h**2)**2 * (5 * self.params['g2']**2 + 3 * self.params['g1']**2)**2)

    def _delta_field_deriv(self, h, T):
        """ private helper function for defining derivatives of longitudinal Z and photon thermal masses """
        return (1/60) * (-7920 * h * T**2 * self.params['g1']**2 * self.params['g2']**2 + 6 * h * (5 * self.params['g2']**2 + 3 * self.params['g1']**2)**2 * (22 * T**2 + 3 * h**2)) / tf.math.sqrt(-2640 * T**2 * self.params['g1']**2 * self.params['g2']**2 * (11 * T**2 + 3 * h**2) + (22 * T**2 + 3 * h**2)**2 * (5 * self.params['g2']**2 + 3 * self.params['g1']**2)**2)

    def _delta_temperature_deriv(self, h, T):
        """ private helper function for defining derivatives of longitudinal Z and photon thermal masses """
        return (11/15) * T * (3 * self.params['g1']**2 - 5 * self.params['g2']**2)**2 * (3 * h**2 + 22 * T**2) / tf.math.sqrt(-2640 * T**2 * self.params['g1']**2 * self.params['g2']**2 * (11 * T**2 + 3 * h**2) + (22 * T**2 + 3 * h**2)**2 * (5 * self.params['g2']**2 + 3 * self.params['g1']**2)**2)


    def mass_sq(self, h, T):
        """ Definition of thermal masses of individual fields
            
            Parameters
            ----------
            h : float
            T : float
            
            Returns
            -------
            float
        """
        
        if (not self.is_eft):                       # for the case of SM and BSM Lagrangians
            if (self.name == 'sm_higgs' and self.is_bsm == True):
                return self.params['mHsq'] + 1.5 * self.params['lmbd'] * h**2 + ((3/80) * self.params['g1']**2 + (3/16) * self.params['g2']**2 + (1/4) * self.params['yt']**2 +(self.params['N']*self.params['lmbd_SH'] / 12)) * T**2

            elif (self.name == 'sm_higgs' and self.is_bsm == False):
                return self.params['mHsq'] + 1.5 * self.params['lmbd'] * h**2 + ((3/80) * self.params['g1']**2 + (3/16) * self.params['g2']**2 + (1/4) * self.params['yt']**2 ) * T**2

            elif (self.name == 'goldstone' and self.is_bsm == True):    
                return self.params['mHsq'] + 0.5 * self.params['lmbd'] * h**2 + ((3/80) * self.params['g1']**2 + (3/16) * self.params['g2']**2 + (1/4) * self.params['yt']**2 +(self.params['N']*self.params['lmbd_SH'] / 12)) * T**2
                
            elif (self.name == 'goldstone' and self.is_bsm == False):    
                return self.params['mHsq'] + 0.5 * self.params['lmbd'] * h**2 + ((3/80) * self.params['g1']**2 + (3/16) * self.params['g2']**2 + (1/4) * self.params['yt']**2 ) * T**2
                
            elif (self.name == 'bsm_scalar' and self.is_bsm == True):
                return self.params['mSsq'] + 0.5 * self.params['lmbd_SH'] * h**2 + ((self.params['lmbd_S'] / 12) * (self.params['N']+1) + (self.params['lmbd_SH'] / 6)) * T**2 

            elif (self.name == 'bsm_scalar' and self.is_bsm == False):
                return 0.0   

            elif self.name == 'w_boson_t': 
                return 0.25 * self.params['g2']**2 * h**2

            elif self.name == 'w_boson_l':
                return 0.25 * self.params['g2']**2 * h**2 + (11/6) * self.params['g2']**2 * T**2

            elif self.name == 'z_boson_t':
                return 0.25 * (0.6 * self.params['g1']**2 + self.params['g2']**2) * h**2

            elif self.name == 'z_boson_l':
                return 0.5 * (0.25 * (0.6 * self.params['g1']**2 + self.params['g2']**2) * h**2 + (11/6) * self.params['g2']**2 * T**2 + (11/10) * self.params['g1']**2 * T**2 + self._delta(h,T))

            elif self.name == 'photon_l':
                return 0.5 * (0.25 * (0.6 * self.params['g1']**2 + self.params['g2']**2) * h**2 + (11/6) * self.params['g2']**2 * T**2 + (11/10) * self.params['g1']**2 * T**2 - self._delta(h,T))

            elif self.name == 't_quark':
                return 0.5 * self.params['yt']**2 * h**2

            elif self.name == 'b_quark':
                return 0.5 * self.params['yb']**2 * h**2

        else:                                       # for SMEFT
            if self.name == 'sm_higgs':
                pass

            elif self.name == 'goldstone':
                pass

            elif self.name == 'w_boson_t':
                pass

            elif self.name == 'w_boson_l':
                pass

            elif self.name == 'z_boson_t':
                pass

            elif self.name == 'z_boson_l':
                pass

            elif self.name == 'photon_l':
                pass

            elif self.name == 't_quark':
                pass

            elif self.name == 'b_quark':
                pass

    def mass_sq_field_deriv(self, h, T):
        """ Derivative of thermal masses of individual fields w.r.t. the SM Higgs as a parameter
            
            Parameters
            ----------
            h : float
            T : float
            
            Returns
            -------
            float
        """
        if (not self.is_eft):                       # for the case of SM and BSM Lagrangians
            if (self.name == 'bsm_scalar' and self.is_bsm == True):
                return self.params['lmbd_SH'] * h    

            elif (self.name == 'bsm_scalar' and self.is_bsm == False):
                return 0.0 
                
            elif self.name == 'sm_higgs':
                return 3 * self.params['lmbd'] * h

            elif self.name == 'goldstone':
                return self.params['lmbd'] * h
                
            elif self.name in ['w_boson_t', 'w_boson_l']:
                return 0.5 * self.params['g2']**2 * h

            elif self.name == 'z_boson_t':
                return 0.5 * (0.6 * self.params['g1']**2 + self.params['g2']**2) * h

            elif self.name == 'z_boson_l':
                return 0.5 * (0.5 * (0.6 * self.params['g1']**2 + self.params['g2']**2) * h + self._delta_field_deriv(h,T))

            elif self.name == 'photon_l':
                return 0.5 * (0.5 * (0.6 * self.params['g1']**2 + self.params['g2']**2) * h - self._delta_field_deriv(h,T))

            elif self.name == 't_quark':
                return self.params['yt']**2 * h

            elif self.name == 'b_quark':
                return self.params['yb']**2 * h 

        else:                       # for SMEFT
            if self.name == 'sm_higgs':
                pass

            elif self.name == 'goldstone':
                pass

            elif self.name == 'w_boson_t':
                pass

            elif self.name == 'w_boson_l':
                pass

            elif self.name == 'z_boson_t':
                pass

            elif self.name == 'z_boson_l':
                pass

            elif self.name == 'photon_l':
                pass

            elif self.name == 't_quark':
                pass

            elif self.name == 'b_quark':
                pass

    def mass_sq_temperature_deriv(self, h, T):
        """ Derivative of thermal masses of individual fields w.r.t. the temperature parameter
            
            Parameters
            ----------
            h : float
            T : float
            
            Returns
            -------
            float
        """
        if (not self.is_eft):                       # for the case of SM and BSM Lagrangians
            if (self.name in ['sm_higgs', 'goldstone'] and self.is_bsm == True):
                return ((3/40) * self.params['g1']**2 + (3/8) * self.params['g2']**2 + (1/2) * self.params['yt']**2 + (self.params['N']*self.params['lmbd_SH'] / 6)) * T

            elif (self.name in ['sm_higgs', 'goldstone'] and self.is_bsm == False):
                return ((3/40) * self.params['g1']**2 + (3/8) * self.params['g2']**2 + (1/2) * self.params['yt']**2 ) * T
                                
            elif (self.name == 'bsm_scalar' and self.is_bsm == True):
                return ((self.params['lmbd_S'] / 6) * (self.params['N']+1) + (self.params['lmbd_SH'] / 3)) * T     

            elif (self.name == 'bsm_scalar' and self.is_bsm == False):
                return 0.0

            elif self.name == 'w_boson_l':
                return (11/3) * self.params['g2']**2 * T

            elif self.name == 'z_boson_l':
                return 0.5 * ( (11/3) * self.params['g2']**2 * T + (11/5) * self.params['g1']**2 * T + self._delta_temperature_deriv(h,T))

            elif self.name == 'photon_l':
                return 0.5 * ( (11/3) * self.params['g2']**2 * T + (11/5) * self.params['g1']**2 * T - self._delta_temperature_deriv(h,T))
                
            elif self.name in ['w_boson_t', 'z_boson_t', 't_quark', 'b_quark']:
                    return 0.0

        else:                                       # for SMEFT
            if self.name == 'sm_higgs':
                pass

            elif self.name == 'goldstone':
                pass

            elif self.name == 'w_boson_t':
                pass

            elif self.name == 'w_boson_l':
                pass

            elif self.name == 'z_boson_t':
                pass

            elif self.name == 'z_boson_l':
                pass

            elif self.name == 'photon_l':
                pass

            elif self.name == 't_quark':
                pass

            elif self.name == 'b_quark':
                pass           

    def get_degrees_of_freedom(self) -> float:
        """ Returns the number of degrees of freedom for individual fields"""

        if self.name in ['sm_higgs', 'z_boson_l', 'photon_l']:
            return 1.0
        elif self.name == 'goldstone':
            return 3.0
        elif self.name == 'w_boson_t':
            return 4.0
        elif self.name in ['w_boson_l', 'z_boson_t']:
            return 2.0
        elif self.name in ['t_quark', 'b_quark']:
            return -12.0
        elif self.name == 'bsm_scalar':
                return 2.0*self.params['N']

