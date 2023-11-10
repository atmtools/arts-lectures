# Advanced Radiation and Remote Sensing

A practical introduction to remote sensing using the
_Atmospheric Radiative Transfer Simulator_ ([ARTS][arts]).

The exercises make use of the ARTS Python API.

Required ARTS version is pre-release 2.5.12.

ARTS requires the Miniforge3 Python environment.
Installers can be downloaded from the [Miniforge Github project][conda] page.
If you are not familiar with the installation procedure of Miniforge, [this page provides good instructions][robotology].

After you have installed Miniforge, ARTS can be installed with the following command:
```
mamba install -c rttools pyarts
```

If you use Spyder, Visual Studio Code or any other IDE  to run the Python scripts, make sure to select the correct interpreter path (`~/miniforge3/bin/python`) in your IDE.

Note that ARTS is only available for Linux and macOS. If you are on Windows or have trouble with the setup, students attending the course at Universität Hamburg can use the computers in the pool rooms or the [VDI system of CEN][vdi-cen] as a fallback solution:

After logging in to a Linux virtual machine, running this command will provide a Miniforge environment with pyarts preinstalled:

```
source /data/share/lehre/unix/rtcourse/activate.sh
```

[arts]: http://radiativetransfer.org/
[vdi-cen]: https://www.cen.uni-hamburg.de/facilities/cen-it/vdi.html
[conda]: https://github.com/conda-forge/miniforge#miniforge
[robotology]: https://github.com/robotology/robotology-superbuild/blob/master/doc/install-miniforge.md
