
def mWsq_t(h):
    self.params['g2'] = 0.6485
    return 0.25 * self.params['g2']**2 * h**2

def mWsq_t_prime(h):
    self.params['g2'] = 0.6485
    return 0.5 * self.params['g2']**2 * h


def mWsq_l(h,T):
    self.params['g2'] = 0.6485
    return 0.25 * self.params['g2']**2 * h**2 + (11/6) * self.params['g2']**2 * T**2

def mWsq_l_prime(h,T):
    self.params['g2'] = 0.6485
    return 0.5 * self.params['g2']**2 * h


def mZsq_t(h):
    (self.params['g1'], self.params['g2']) = (0.4632, 0.6485)
    return 0.25 * (0.6 * self.params['g1']**2 + self.params['g2']**2) * h**2

def mZsq_t_prime(h):
    (self.params['g1'], self.params['g2']) = (0.4632, 0.6485)
    return 0.5 * (0.6 * self.params['g1']**2 + self.params['g2']**2) * h


def delta(h,T):
    (self.params['g1'], self.params['g2']) = (0.4632, 0.6485)
    return (1/60) * tf.math.sqrt(-2640 * T**2 * self.params['g1']**2 * self.params['g2']**2 * (11 * T**2 + 3 * h**2) + (22 * T**2 + 3 * h**2)**2 * (5 * self.params['g2']**2 + 3 * self.params['g1']**2)**2)

def delta_prime(h,T):              
    (self.params['g1'], self.params['g2']) = (0.4632, 0.6485)
    return (1/60) * (-7920 * h * T**2 * self.params['g1']**2 * self.params['g2']**2 + 6 * h * (5 * self.params['g2']**2 + 3 * self.params['g1']**2)**2 * (22 * T**2 + 3 * h**2)) / (tf.math.sqrt(-2640 * T**2 * self.params['g1']**2 * self.params['g2']**2 * (11 * T**2 + 3 * h**2) + (22 * T**2 + 3 * h**2)**2 * (5 * self.params['g2']**2 + 3 * self.params['g1']**2)**2))


def mZsq_l(h,T):
    (self.params['g1'], self.params['g2']) = (0.4632, 0.6485)
    return 0.5 * (0.25 * (0.6 * self.params['g1']**2 + self.params['g2']**2) * h**2 + (11/6) * self.params['g2']**2 * T**2 + (11/10) * self.params['g1']**2 * T**2 + delta(h,T))

def mZsq_l_prime(h,T):
    (self.params['g1'], self.params['g2']) = (0.4632, 0.6485)
    return 0.5 * (0.5 * (0.6 * self.params['g1']**2 + self.params['g2']**2) * h + delta_prime(h,T))


def mphsq_l(h,T):
    (self.params['g1'], self.params['g2']) = (0.4632, 0.6485)
    return 0.5 * (0.25 * (0.6 * self.params['g1']**2 + self.params['g2']**2) * h**2 + (11/6) * self.params['g2']**2 * T**2 + (11/10) * self.params['g1']**2 * T**2 - delta(h,T))

def mphsq_l_prime(h,T):
    (self.params['g1'], self.params['g2']) = (0.4632, 0.6485)
    return 0.5 * (0.5 * (0.6 * self.params['g1']**2 + self.params['g2']**2) * h - delta_prime(h,T))


def mGsq(h,T):
    (self.params['mHsq'], self.params['lmbd'], self.params['g1'], self.params['g2'], self.params['yt'], self.params['lmbd_SH'], self.params['N']) = (0.286270859792, -0.6888422590446, 0.4632, 0.6485, 0.92849, 10.90, 0.5)
    return self.params['mHsq'] + 0.5 * self.params['lmbd'] * h**2 + ((3/80) * self.params['g1']**2 + (3/16) * self.params['g2']**2 + (1/4) * self.params['yt']**2 + (self.params['N']*self.params['lmbd_SH'] / 12)) * T**2

def mGsq_prime(h,T):
    (self.params['mHsq'], self.params['lmbd'], self.params['g1'], self.params['g2'], self.params['yt'], self.params['lmbd_SH'], self.params['N']) = (0.286270859792, -0.6888422590446, 0.4632, 0.6485, 0.92849, 10.90, 0.5)
    return self.params['lmbd'] * h


def self.params['mHsq'](h,T):
    (self.params['mHsq'], self.params['lmbd'], self.params['g1'], self.params['g2'], self.params['yt'], self.params['lmbd_SH'], self.params['N']) = (0.286270859792, -0.6888422590446, 0.4632, 0.6485, 0.92849, 10.90, 0.5)
    return self.params['mHsq'] + 1.5 * self.params['lmbd'] * h**2 + ((3/80) * self.params['g1']**2 + (3/16) * self.params['g2']**2 + (1/4) * self.params['yt']**2 + (self.params['N']*self.params['lmbd_SH'] / 12)) * T**2

def self.params['mHsq']_prime(h,T):
    (self.params['mHsq'], self.params['lmbd'], self.params['g1'], self.params['g2'], self.params['yt'], self.params['lmbd_SH'], self.params['N']) = (0.286270859792, -0.6888422590446, 0.4632, 0.6485, 0.92849, 10.90, 0.5)
    return 3 * self.params['lmbd'] * h


def mtsq(h):
    self.params['yt'] = 0.92849                 
    return 0.5 * self.params['yt']**2 * h**2 

def mtsq_prime(h):
    self.params['yt'] = 0.92849                 
    return self.params['yt']**2 * h

def mbsq(h):
    self.params['yb'] = 0.0167
    return 0.5 * self.params['yb']**2 * h**2 

def mbsq_prime(h):
    self.params['yb'] = 0.0167
    return self.params['yb']**2 * h 


def mSsq(h,T):
    (self.params['mSsq'], self.params['lmbd_SH'], self.params['lmbd_S'], self.params['N']) = (4.4819542, 10.90, 1.0, 0.5)    
    return self.params['mSsq'] + 0.5 * self.params['lmbd_SH'] * h**2 + ((self.params['lmbd_S'] / 12) * (self.params['N']+1) + (self.params['lmbd_SH'] / 6)) * T**2

def mSsq_prime(h,T):
    (self.params['mSsq'], self.params['lmbd_SH'], self.params['lmbd_S'], self.params['N']) = (4.4819542, 10.90, 1.0, 0.5)    
    return self.params['lmbd_SH'] * h