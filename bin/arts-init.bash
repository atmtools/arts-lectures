#!/bin/bash
#
# arts-init.bash
# Copyright (C) 2016 Lukas Kluft <lukas.kluft@gmail.com>
#
# Distributed under terms of the MIT license.
#

# Set environment variables where to find ...
# ... the arts binary,
PATH="/scratch/lehre/rtcourse/arts/build/src:$PATH"
# ... ARTS auxiliary data,
ARTS_DATA_PATH="/scratch/lehre/rtcourse/arts-xml-data:$ARTS_DATA_PATH"
ARTS_DATA_PATH="/scratch/lehre/rtcourse/catalogue:$ARTS_DATA_PATH"
# ... and the MATLAB package atmlab.
MATLABPATH="/scratch/lehre/rtcourse/atmlab/atmlab:$MATLABPATH"

export PATH ARTS_DATA_PATH MATLABPATH

# Limit ARTS CPU usage to four cores.
export OMP_NUM_THREADS=4

# Load latest MATLAB version.
module unload matlab && module load matlab/2016a

# Setup Python 3 environent (using Anaconda).
module unload python
export PATH="/scratch/lehre/rtcourse/anaconda3/bin:$PATH"
