## slurm_gpustat

`slurm_gpustat` is a simple command line utility that produces a summary of GPU usage on a slurm cluster. The tool can be used in two ways:
1. To query the current usage of GPUs on the cluster.
2. To launch a daemon which will log usage over time.  This log can later be queried to provide usage statistics.

### Installation

Install via `pip install slurm_gpustat`.  If you prefer to hack around with the source code, it's a [single python file](slurm_gpustat/slurm_gpustat.py).


### Usage

To print a summary of current activity:

`slurm_gpustat`

To print a summary of current activity on particular partitions, e.g. `debug` & `normal`:

`slurm_gpustat -p debug,normal` or `slurm_gpustat --partition debug,normal`

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

Adding `--verbose` to this command will produce a more detailed breakdown for the section describing gpus that are still available.  Example output:
```
There are 18 gpus available:
m40: 5 available
 -> gnodeb3: 1 m40 [cpu: 38/40, gres/gpu: 3/4, mem: 68G/257669M] [user1,user2]
 -> gnodeb4: 2 m40 [cpu: 38/40, gres/gpu: 2/4, mem: 60G/193161M] [user1,user2]
 -> gnodec3: 2 m40 [cpu: 12/48, gres/gpu: 2/4, mem: 36G/257669M] [user1]
p40: 5 available
 -> gnodec4: 1 p40 [cpu: 20/48, gres/gpu: 3/4, mem: 216G/257669M] [user1]
 -> gnoded1: 1 p40 [cpu: 44/64, gres/gpu: 3/4, mem: 60G/385192M] [user4,user5]
 -> gnoded3: 1 p40 [cpu: 28/64, gres/gpu: 3/4, mem: 70G/385192M] [user2,user4,user5]
 -> gnoded4: 1 p40 [cpu: 28/64, gres/gpu: 3/4, mem: 112G/385192M] [user5,user4]
 -> gnodee3: 1 p40 [cpu: 36/56, gres/gpu: 3/4, mem: 60G/385192M] [user4,user2]
rtx6k: 3 available
 -> gnodef1: 1 rtx6k [cpu: 36/56, gres/gpu: 3/4, mem: 194G/257669M] [user2,user3]
 -> gnodeh2: 2 rtx6k [cpu: 16/24, gres/gpu: 2/4, mem: 96G/385345M] [user2,user1]
v100: 5 available
 -> gnodeg1: 2 v100 [cpu: 16/64, gres/gpu: 2/4, mem: 60G/191668M] [user5,user1]
 -> gnodeg2: 3 v100 [cpu: 8/64, gres/gpu: 1/4, mem: 40G/191668M] [user2]
```

Running `slurm_gpustat --action history` (after the daemon has run for a little while) will produce something like this:

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
* `beartype`
* `seaborn`
* `colored`

