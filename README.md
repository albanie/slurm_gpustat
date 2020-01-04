## slurm_gpustat

`slurm_gpustat` is a simple command line utility that produces a summary of GPU usage on a slurm cluster. The tool can be used in two ways:
1. To query the current usage of GPUs on the cluster.
2. To launch a daemon which will log usage over time.  This log can later be queried to provide usage statistics.

### Installation

Install via `pip install slurm_gpustat`.  If you prefer to hack around with the source code, it's a [single python file](slurm_gpustat/slurm_gpustat.py).


### Usage

To print a summary of current activity:

`slurm_gpustat`

To start the logging dameon:

`slurm_gpustat --action daemon-start`

To view a summary of logged data:

`slurm_gpustat --action history`


### Example outputs

Running `slurm_gpustat` will produce something like this:

```
----------------------
Under SLURM management
----------------------
There are a total of 12 gpus [up]
3 rtx6k gpus
2 v100 gpus
3 m40 gpus
4 p40 gpus
----------------------
There are a total of 10 gpus [accessible]
1 rtx6k gpus
2 v100 gpus
3 m40 gpus
4 p40 gpus
----------------------
Usage by user:
user1     [total: 1 ] m40: 1
user2     [total: 1 ] p40: 1
user3     [total: 3 ] p40: 3
----------------------
There are 5 gpus available:
p40: 0
rtx6k: 1
v100: 2
m40: 2
```

Running `slurm_gpustat --history` (after the daemon has run for a little while) will produce something like this:

```
Historical data contains 7 samples (2020-01-03 11:51:43 to 2020-01-03 11:51:45)
GPU usage for user1:
v100  > avg: 2, max: 4
m40   > avg: 1, max: 1
total > avg: 3

GPU usage for user2:
p40m  > avg: 3, max: 4
total > avg: 3
```

### Dependencies

* `Python >= 3.6`
* `numpy`
