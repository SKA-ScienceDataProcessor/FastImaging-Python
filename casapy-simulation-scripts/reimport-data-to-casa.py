import numpy as np
np.set_printoptions(threshold='nan')
tb.open('vla-resim.MS', nomodify=False)

newvis = np.loadtxt('data-vla-resim.txt').view(complex).reshape(-1)

corrected = tb.getcol('CORRECTED_DATA')
corrected[0] = newvis.transpose()
tb.putcol('CORRECTED_DATA', corrected)
tb.close()
