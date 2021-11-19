# SeaIce_svNewton

## Usage

The firedrake virtual env needs to be activated:
```
source /path/to/firedrake/bin/activate
```

* For the time-dependent JAMES benchmark, run with following command:
```
python3 seaice.py --linearization stressvel
```

* For the one-step problem momentum-only problem, run with following command:
```
python3 seaice_momonly.py --linearization stressvel
```

Note: To switch to standard Newton linearization, use flag `--linearization stdnewton`.
