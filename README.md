# Infrastructure for the course 'Applied Remote Sensing'.

## Requirements
The course draws heavily on performing practical exercises using the
[Atmospheric Radiative Transer Simulator][ARTS]. You can check the official
website for information on how to [get ARTS][get-arts].

Students attending the course at Universität Hamburg will find a full ARTS
environment prepared on the university's server ([ARTS lecture][])

To check if the environment is properly set run:
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
Run the following command in MATLAB to initialize atmlab (or add it to
your `startup.m`):
```matlab
atmlab_init
```

### typhon
Typhon includes unittests which can be run to test the installation:
```python
pytest --pyarts typhon
```

[ARTS]: http://radiativetransfer.org/
[ARTS lecture]: https://www.mi.uni-hamburg.de/en/arbeitsgruppen/strahlung-und-fernerkundung/intern/howtos/arts-lecture.html
[Anaconda]: https://www.continuum.io/downloads
[atmlab]: http://radiativetransfer.org/tools/#atmlab
[get-arts]: http://radiativetransfer.org/getarts/
[typhon]: http://radiativetransfer.org/tools/#typhon
