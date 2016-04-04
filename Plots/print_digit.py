from sklearn.datasets import load_digits
digits = load_digits()

import pylab as pl 
pl.gray() 
pl.matshow(digits.images[3]) 
pl.show() 