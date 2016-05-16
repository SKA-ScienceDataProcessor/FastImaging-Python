import numpy
numpy.set_printoptions(threshold='nan')

measurementset = 'vla-sim.MS'

ms.open(measurementset)
md=ms.metadata()

frequency = md.chanfreqs(0)[0]
speed_c = 3E8 #m/s
wavelength = speed_c / frequency
print "LAMBDA", wavelength

ms.close()

tb.open(measurementset, nomodify=F)
uvw_metres = tb.getcol('UVW')
uvw_lambda = uvw_metres / wavelength
uvw_lambda_col = uvw_lambda.transpose()

numpy.savetxt('uvw-lambda.txt', uvw_lambda_col)
tb.close
