4 * ( (tf.math.xlogy(mWsq_t(hc)**2, mWsq_t(hc)/mu**2) - tf.math.xlogy(mWsq_t(vevhc)**2, mWsq_t(vevhc)/mu**2)) - (5/6) * (mWsq_t(hc)**2 - mWsq_t(vevhc)**2)  )

+ 2 * ( (tf.math.xlogy(mWsq_l(hc,T)**2, mWsq_l(hc,T)/mu**2) - tf.math.xlogy(mWsq_l(vevhc,T)**2, mWsq_l(vevhc,T)/mu**2)) - (5/6) * (mWsq_l(hc,T)**2 - mWsq_l(vevhc,T)**2))

+ 2 * (tf.math.xlogy(mZsq_t(hc)**2, mZsq_t(hc)/mu**2) - (5/6) * mZsq_t(hc)**2 )

+ ( tf.math.xlogy(mZsq_l(hc,T)**2, mZsq_l(hc,T)/mu**2) - (5/6) * mZsq_l(hc,T)**2 )

#-----------------------------------------------------------------------------------------------------------------------------

+ ( tf.math.xlogy(mphsq_l(hc,T)**2, mphsq_l(hc,T)/mu**2) - (5/6) * mphsq_l(hc,T)**2 )

+ 3 * ( tf.math.xlogy(mGsq(hc,T)**2, mGsq(hc,T)/mu**2) - (3/2) * mGsq(hc,T)**2 ) 

+ ( tf.math.xlogy(mhsq(hc,T)**2, mhsq(hc,T)/mu**2) - (3/2) * mhsq(hc,T)**2 )

+ 2*N * (tf.math.xlogy(mSsq(hc,T)**2, mSsq(hc,T)/mu**2) - (3/2) * mSsq(hc,T)**2 ) 

- 12 * (tf.math.xlogy(mtsq(hc)**2, mtsq(hc)/mu**2) - (3/2) * mtsq(hc)**2 ) 

- 12 * (tf.math.xlogy(mbsq(hc)**2, mbsq(hc)/mu**2) - (3/2) * mbsq(hc)**2 )