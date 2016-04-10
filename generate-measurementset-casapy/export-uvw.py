import numpy
numpy.set_printoptions(threshold='nan')
tb.open('vla-sim.MS', nomodify=F)
uvw = tb.getcol('UVW')
uvw_col = uvw.transpose()
numpy.savetxt('uvw-vla-sim.txt', uvw_col)
tb.close
