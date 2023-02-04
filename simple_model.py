# here we will create a model class whose subclasses would correspond to the various example cases - sm, bsm, sm+dilaton, bsm + dilaton,
# 
# 
# class model(fields = 'list-of-field-id-strings', params = 'dict-of-param-name-value-pairs'):
#   
#    ----first construct subdicts of params corresponding to different fields----
#    self.higgs_params = {'mHsq': self.params['mHsq'], 'lambda': self.params['lambda'], 'g1': self.params['g1'], 'g2': self.params['g2'], 'yt': self.params['yt'], 'lambda_SH': self.params['lambda_SH'], 'N': self.params['N'] }
#    self.w_params = {'g2': self.params['g2']}
#    self.z_params = {'g1': self.params['g1'], 'g2': self.params['g2']}
#    
#  