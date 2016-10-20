#!/usr/bin/tcsh
#
# arts-init.tcsh
# Copyright (C) 2016 Lukas Kluft <lukas.kluft@gmail.com>
#
# Distributed under terms of the MIT license.
#

# Set environment variables where to find ...
# ... the arts binary,
setenv PATH "/scratch/lehre/rtcourse/arts/build/src:$PATH"
# ... ARTS auxiliary data,
if ( $?ARTS_DATA_PATH ) then
    setenv ARTS_DATA_PATH "/scratch/lehre/rtcourse/arts-xml-data:$ARTS_DATA_PATH"
else
    setenv ARTS_DATA_PATH "/scratch/lehre/rtcourse/arts-xml-data"
endif
setenv ARTS_DATA_PATH "/scratch/lehre/rtcourse/catalogue:$ARTS_DATA_PATH"
# ... and the MATLAB package atmlab.
if ( $?MATLABPATH ) then
    setenv MATLABPATH "/scratch/lehre/rtcourse/atmlab/atmlab:$MATLABPATH"
else
    setenv MATLABPATH "/scratch/lehre/rtcourse/atmlab/atmlab"
endif

# Limit ARTS CPU usage to four cores.
setenv OMP_NUM_THREADS 4

# Load latest MATLAB version.
module unload matlab && module load matlab/2016a
