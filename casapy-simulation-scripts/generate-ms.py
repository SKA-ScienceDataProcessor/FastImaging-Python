import math

import numpy as np

# -----------------------------------------
#
# define constants.
#
# -----------------------------------------

TELESCOPE = 'VLA'
FILENAME = TELESCOPE.lower()

# set phase position.
RA = "12h00m00s"
DEC = "34d00m00s"

# define flux of source.
FLUX = 1.0

# simulation properties
LOCATION = 'VLA'
OBSERVATION_LENGTH = 7200.0  # total observing time, in seconds.
DUMP_TIME = 10.0  # dump time.
FOV = 56.6  # the field of view HWHM (in arcminutes)
FREQUENCY = 2.5  # the frequency, and
FREQUENCY_UNIT = 'GHz'  # units, and
FREQUENCY_RESOLUTION = '125kHz'  # resolution.
REF_TIME = me.epoch('UTC',
                    '2014/05/01/19:55:45')  # the reference time for the simulation.
POSITION = me.observatory(LOCATION)  # observatory position
ANTENNA_LIST = './vla.c.cfg'  # name of the file containing the antennae layouts



# -----------------------------------------
#
# sm.getknownconfig() doesn't seem to work,
# so we need to extract configuration details
# from a .cfg file and use sm.getconfig() instead...
#
# -----------------------------------------

def fnGetConfig(antennalist):
    f = open(antennalist, 'r')

    x = [];
    y = [];
    z = [];
    d = []
    while True:
        line = f.readline()
        if not line: break

        items = line.split()
        if (items[0] != '#'):
            d.append(float(items[3]))
            x.append(float(items[0]))
            y.append(float(items[1]))
            z.append(float(items[2]))

    f.close()

    return x, y, z, d


# -----------------------------------------
#
# 1. make component list.
#
# -----------------------------------------

# build component list filename.
filename = 'src-' + FILENAME

# overwrite existing component list file:
os.system('rm -rf ' + filename + '.cl')

# clear any existing component list details from memory:
cl.done()

# add source 1.
cl.addcomponent(dir="J2000 " + RA + " " + DEC, flux=FLUX, fluxunit='Jy',
                freq=str(FREQUENCY) + FREQUENCY_UNIT, shape='point')

# write a component list file: 
cl.rename(filename + '.cl')

# close everything down:
cl.done()

# -----------------------------------------
#
# 2. simulate observations using the component list from above, and a Gaussian primary beam.
#
# -----------------------------------------

# build RA and declination.
fieldDir = me.direction('J2000', RA, DEC)

# build measurement set filename.
root_ms = FILENAME + '-sim'
output_ms = root_ms + '.MS'

# remove the measurement set if it already exists.
os.system('rm -rf ' + output_ms + '\n')

# create measurement set.
sm.open(output_ms)

# get antennae data if the form of several arrays.
xlist, ylist, zlist, dishes = fnGetConfig(ANTENNA_LIST)

# make a primary beam pattern:
hpbw = str(FOV / 60.) + 'deg'
vp.setpbgauss(telescope=LOCATION, dopb=True, halfwidth=hpbw,
              reffreq=str(FREQUENCY) + FREQUENCY_UNIT)

sm.setconfig(telescopename=LOCATION,
             x=xlist, y=ylist, z=zlist,
             dishdiameter=dishes,
             mount='alt-az',
             coordsystem='local',
             referencelocation=POSITION)

sm.setspwindow(spwname='IF0',
               freq=str(FREQUENCY) + FREQUENCY_UNIT,
               deltafreq='0.5MHz',
               freqresolution=FREQUENCY_RESOLUTION,
               nchannels=1,
               stokes='XX XY YX YY')

sm.setfeed(mode='perfect X Y', pol=[''])
sm.setfield(sourcename='ptcentre', sourcedirection=fieldDir)
sm.setlimits(shadowlimit=0.001, elevationlimit='15.0deg')
sm.setauto(autocorrwt=0.0)
sm.settimes(integrationtime=str(DUMP_TIME) + 's', usehourangle='T',
            referencetime=REF_TIME)

# set the start and end times, and the interval to 32 seconds.
begin = 0
end = begin + OBSERVATION_LENGTH

sm.observe('ptcentre', 'IF0', starttime=str(begin / 3600) + 'h',
           stoptime=str(end / 3600) + 'h')

# generate the predicted visibilities.
sm.setdata(fieldid=[0])
sm.predict(imagename=[''], complist='src-' + FILENAME + '.cl')

sm.setnoise(simplenoise='10Jy')
sm.corrupt()

sm.close()
sm.done()
