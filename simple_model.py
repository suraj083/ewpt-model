# Here we will create a model class whose subclasses would correspond to the various example cases - sm, bsm, sm+dilaton, bsm + dilaton,
# 
#
# sample_field_id_list = ['sm_higgs', 'bsm_scalar', 'w_boson', 'z_boson', 't_quark', 'b_quark']
# sample_param_dict = ['g1':1, 'g2':1, 'yt':1, 'yb':1, 'mHsq':1, 'mSsq':1, 'lmbd':1, 'lmbd_SH':1, 'lmbd_S':1,  ]
#  
# class model(fields = 'list-of-field-id-strings', params = 'dict-of-param-name-value-pairs'):
#  
#    ----first construct subdicts of params corresponding to different fields----
#    ---for now assume all these dict_keys are there---    
#
#    h_params = {'mHsq': self.params['mHsq'], 'lmbd': self.params['lmbd'], 'g1': self.params['g1'], 'g2': self.params['g2'], 'yt': self.params['yt'], 'lmbd_SH': self.params['lmbd_SH'], 'N': self.params['N'] }
#    s_params = {'mSsq': self.params['mSsq'], 'lmbd_SH': self.params['lmbd_SH'], 'lmbd_S': self.params['lmbd_S'], 'N': self.params['N']}
#    z_params = {'g1': self.params['g1'], 'g2': self.params['g2']}
#    w_params = {'g2': self.params['g2']}
#    t_params = {'yt': self.params['yt']}
#    b_params = {'yb': self.params['yb']}
#    