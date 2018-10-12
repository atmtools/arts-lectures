#!/bin/bash
#
# arts-init.bash
# Copyright (C) 2016 Lukas Kluft <lukas.kluft@gmail.com>
#
# Distributed under terms of the MIT license.
#

# Unload CEN-IT python modules because they break the PYTHONPATH
module unload python3 python

# Set environment variables where to find ...
# ... the arts binary,
PATH="/data/share/lehre/unix/rtcourse/arts/build/src:$PATH"
PATH="$PATH:/data/share/lehre/unix/rtcourse/bin"
# ... ARTS auxiliary data,
ARTS_DATA_PATH="/data/share/lehre/unix/rtcourse/arts-xml-data:$ARTS_DATA_PATH"
ARTS_DATA_PATH="/data/share/lehre/unix/rtcourse/catalogue:$ARTS_DATA_PATH"
ARTS_INCLUDE_PATH="/data/share/lehre/unix/rtcourse/arts/controlfiles:$ARTS_INCLUDE_PATH"

export PATH ARTS_DATA_PATH ARTS_INCLUDE_PATH MATLABPATH

# Limit ARTS CPU usage to four cores.
export OMP_NUM_THREADS=2

# Load latest MATLAB version.
module unload matlab && module load matlab/2018a
MATLABPATH="/data/share/lehre/unix/rtcourse/atmlab/atmlab:$MATLABPATH"

# Setup Python 3 environent (using Anaconda).
export PATH="/data/share/lehre/unix/rtcourse/anaconda3/bin:$PATH"
