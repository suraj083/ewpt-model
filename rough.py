
def mWsq_t_prime(h):
    self.params['g2'] = 0.6485
    return 0.5 * self.params['g2']**2 * h


def mWsq_l_prime(h,T):
    self.params['g2'] = 0.6485
    return 0.5 * self.params['g2']**2 * h


def mZsq_t_prime(h):
    (self.params['g1'], self.params['g2']) = (0.4632, 0.6485)
    return 0.5 * (0.6 * self.params['g1']**2 + self.params['g2']**2) * h

def delta_prime(h,T):              
    (self.params['g1'], self.params['g2']) = (0.4632, 0.6485)
    return (1/60) * (-7920 * h * T**2 * self.params['g1']**2 * self.params['g2']**2 + 6 * h * (5 * self.params['g2']**2 + 3 * self.params['g1']**2)**2 * (22 * T**2 + 3 * h**2)) / (tf.math.sqrt(-2640 * T**2 * self.params['g1']**2 * self.params['g2']**2 * (11 * T**2 + 3 * h**2) + (22 * T**2 + 3 * h**2)**2 * (5 * self.params['g2']**2 + 3 * self.params['g1']**2)**2))


def mZsq_l_prime(h,T):
    (self.params['g1'], self.params['g2']) = (0.4632, 0.6485)
    return 0.5 * (0.5 * (0.6 * self.params['g1']**2 + self.params['g2']**2) * h + delta_prime(h,T))


def mphsq_l_prime(h,T):
    (self.params['g1'], self.params['g2']) = (0.4632, 0.6485)
    return 0.5 * (0.5 * (0.6 * self.params['g1']**2 + self.params['g2']**2) * h - delta_prime(h,T))


def mGsq_prime(h,T):
    (self.params['mHsq'], self.params['lmbd'], self.params['g1'], self.params['g2'], self.params['yt'], self.params['lmbd_SH'], self.params['N']) = (0.286270859792, -0.6888422590446, 0.4632, 0.6485, 0.92849, 10.90, 0.5)
    return self.params['lmbd'] * h

def self.params['mHsq']_prime(h,T):
    (self.params['mHsq'], self.params['lmbd'], self.params['g1'], self.params['g2'], self.params['yt'], self.params['lmbd_SH'], self.params['N']) = (0.286270859792, -0.6888422590446, 0.4632, 0.6485, 0.92849, 10.90, 0.5)
    return 3 * self.params['lmbd'] * h



def mtsq_prime(h):
    self.params['yt'] = 0.92849                 
    return self.params['yt']**2 * h



def mbsq_prime(h):
    self.params['yb'] = 0.0167
    return self.params['yb']**2 * h 

 

def mSsq_prime(h,T):
    (self.params['mSsq'], self.params['lmbd_SH'], self.params['lmbd_S'], self.params['N']) = (4.4819542, 10.90, 1.0, 0.5)    
    return self.params['lmbd_SH'] * h