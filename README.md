# Advanced Radiation and Remote Sensing

A practical introduction to remote sensing using the
_Atmospheric Radiative Transfer Simulator_ ([ARTS][arts]).

The exercises make use of the ARTS Python API.

Required ARTS version is pre-release 2.5.12

Matching catalogs are [available here][cats].

The `environment.yml` file can be used to setup a [conda environment][conda]
with all necessary packages including PyARTS:

```
mamba env create -f environment.yml
mamba activate arts-lectures
jupyter-lab exercises
```

Students attending the course at Universität Hamburg use the [VDI system of CEN][vdi-cen].

[arts]: http://radiativetransfer.org/
[vdi-cen]: https://www.cen.uni-hamburg.de/facilities/cen-it/vdi.html
[typhon-github]: https://github.com/atmtools/typhon/
[cats]: https://www.radiativetransfer.org/misc/download/unstable/
[conda]: https://github.com/conda-forge/miniforge
