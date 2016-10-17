# Infrastructure for the course 'Applied Remote Sensing'.

## Recommendations
The course draws heavily on performing practical exercises using the
[Atmospheric Radiative Transer Simulator][ARTS]. You either have to
[get your own version][get-arts] or use the preconfigured environment on
`lehre3` (`squall3`).

## Setting environment variables
In order to use the precompiled ARTS version it is neccerssary to set some
environment variables (namely `PATH`, `ARTS_DATA_PATH`, `MATLABPATH` and
`OMP_NUM_THREADS`).
There are shell scripts to perform this task. Soure the appropriate script for
your shell (e.g. `bash`):

```bash
source bin/arts-init.bash
```

To check if the environment is properly set try to run

```bash
arts --version
```
## Additional software
The input and output of ARTS makes use of XML files. There are packages to
support the postprocessing of simulation results.

[typhon][] (Python 3) and [atmlab][] (MATLAB) provide reading and writing
routines for ARTS XML files as well as helpful functions related to radiation
and remote sensing.

### atmlab
A current release of atmlab is checked out and ready to use on `lehre3`.
If you have set the environment variables the atmlab directory should already
be in the `MATLABPATH`. Run the following command in MATLAB to initialize
atmlab (or add it to your `startup.m`).

```matlab
atmlab_init
```

### typhon
The recommended way to install typhon is to use [`conda`][Anaconda].

```bash
conda install -c rttools typhon
```

Alternative installation methods can be found in the [online
documentation][typhon].


[ARTS]: http://radiativetransfer.org/
[Anaconda]: https://www.continuum.io/downloads
[atmlab]: http://radiativetransfer.org/tools/#atmlab
[get-arts]: http://radiativetransfer.org/getarts/
[typhon]: http://radiativetransfer.org/tools/#typhon
