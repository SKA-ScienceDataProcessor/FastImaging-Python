import numpy

numpy.set_printoptions(threshold='nan')
tb.open('vla-sim.MS', nomodify=F)
corrected = tb.getcol('CORRECTED_DATA')
data_col = corrected[0].transpose()
real_col = data_col.real
imag_col = data_col.imag
numpy.savetxt('data-vla-sim.txt', numpy.hstack([real_col, imag_col]))
tb.close
