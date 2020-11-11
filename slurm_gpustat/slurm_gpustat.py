"""A simple tool for summarising GPU statistics on a slurm cluster.

The tool can be used in two ways:
1. To simply query the current usage of GPUs on the cluster.
2. To launch a daemon which will log usage over time.  This can then later be queried
   to provide simple usage statistics.
"""
import argparse
import ast
import atexit
import os
import random
import re
import signal
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import functools

import colored
import numpy as np
import seaborn as sns

from beartype import beartype

# SLURM states which indicate that the node is not available for submitting jobs
INACCESSIBLE = {"drain*", "down*", "drng", "drain", "down"}
INTERACTIVE_CMDS = {"bash", "zsh", "sh"}

class Daemon:
    """A Generic linux daemon base class for python 3.x.

    This code is a Python3 port of Sander Marechal's Daemon module:
    http://web.archive.org/web/20131017130434/http://www.jejik.com/articles/
    2007/02/a_simple_unix_linux_daemon_in_python/

    It's a little difficult to credit the author of the Python3 port, since the code was
    published anonymously. The original can be found here:
    http://web.archive.org/web/20131101191715/http://www.jejik.com/files/examples/
    daemon3x.py
    """

    def __init__(self, pidfile):
        self.pidfile = pidfile

    def daemonize(self):
        """Deamonize class. UNIX double fork mechanism."""
        try:
            pid = os.fork()
            if pid > 0:
                # exit first parent
                sys.exit(0)
        except OSError as err:
            sys.stderr.write('fork #1 failed: {0}\n'.format(err))
            sys.exit(1)

        # decouple from parent environment
        os.chdir('/')
        os.setsid()
        os.umask(0)

        # do second fork
        try:
            pid = os.fork()
            if pid > 0:
                # exit from second parent
                sys.exit(0)
        except OSError as err:
            sys.stderr.write('fork #2 failed: {0}\n'.format(err))
            sys.exit(1)

        # redirect standard file descriptors
        sys.stdout.flush()
        sys.stderr.flush()
        si = open(os.devnull, 'r')
        so = open(os.devnull, 'a+')
        se = open(os.devnull, 'a+')

        os.dup2(si.fileno(), sys.stdin.fileno())
        os.dup2(so.fileno(), sys.stdout.fileno())
        os.dup2(se.fileno(), sys.stderr.fileno())

        # write pidfile
        atexit.register(self.delpid)

        pid = str(os.getpid())
        with open(self.pidfile, 'w+') as f:
            f.write(pid + '\n')

    def delpid(self):
        os.remove(self.pidfile)

    def start(self):
        """Start the daemon."""

        # Check for a pidfile to see if the daemon already runs
        try:
            with open(self.pidfile, 'r') as pf:

                pid = int(pf.read().strip())
        except IOError:
            pid = None

        if pid:
            message = "pidfile {0} already exists. Is the daemon already running?\n"
            sys.stderr.write(message.format(self.pidfile))
            sys.exit(1)

        # Start the daemon
        self.daemonize()
        self.run()

    def stop(self):
        """Stop the daemon."""

        # Get the pid from the pidfile
        try:
            with open(self.pidfile, 'r') as pf:
                pid = int(pf.read().strip())
        except IOError:
            pid = None

        if not pid:
            message = "pidfile {0} does not exist. Daemon not running?\n"
            sys.stderr.write(message.format(self.pidfile))
            return  # not an error in a restart

        # Try killing the daemon process
        try:
            while 1:
                os.kill(pid, signal.SIGTERM)
                time.sleep(0.1)
        except OSError as err:
            e = str(err.args)
            if e.find("No such process") > 0:
                if os.path.exists(self.pidfile):
                    os.remove(self.pidfile)
            else:
                print(str(err.args))
                sys.exit(1)

    def restart(self):
        """Restart the daemon."""
        self.stop()
        self.start()

    def run(self):
        """You should override this method when you subclass Daemon.
        It will be called after the process has been daemonized by
        start() or restart()."""
        raise NotImplementedError("Must override this class")


class GPUStatDaemon(Daemon):
    """A lightweight daemon which intermittently logs gpu usage to a text file.
    """
    timestamp_format = "%Y-%m-%d_%H:%M:%S"

    def __init__(self, pidfile, log_path, log_interval):
        """Create the daemon.

        Args:
            pidfile (str): the location of the daemon pid file.
            log_path (str): the location where the historical log will be stored.
            log_interval (int): the time interval (in seconds) at which gpu usage will
                be stored to the log.
        """
        Path(pidfile).parent.mkdir(exist_ok=True, parents=True)
        super().__init__(pidfile=pidfile)
        Path(log_path).parent.mkdir(exist_ok=True, parents=True)
        self.log_interval = log_interval
        self.log_path = log_path

    def serialize_usage(self, usage):
        """Convert data structure into an appropriate string for serialization.

        Args:
            usage (a dict-like structure): a data-structure which has the form of a
                dictionary, but may contain variants with length string representations
                (e.g. defaultdict, OrderedDict etc.)

        Returns:
            (str): a string representation of the usage data strcture.
        """
        for user, gpu_dict in usage.items():
            for key, subdict in gpu_dict.items():
                usage[user][key] = dict(subdict)
        usage = dict(usage)
        return usage.__repr__()

    @staticmethod
    def deserialize_usage(log_path):
        """Parse the `usage` data structure by reading in the contents of the text-based
        log file and deserializing.

        Args:
            log_path (str): the location of the log file.

        Returns:
            (list[dict]): a list of dicts, where each dict contains the time stamp
                associated with a set of usage statistics, together with the statistics
                themselves.
        """
        if not Path(log_path).exists():
            raise ValueError(f"No historical log found.  Did you start the daemon?")
        with open(log_path, "r") as f:
            rows = f.read().splitlines()
        data = []
        for row in rows:
            ts, usage = row.split(maxsplit=1)
            dt = datetime.strptime(ts, GPUStatDaemon.timestamp_format)
            usage = ast.literal_eval(usage)
            data.append({"timestamp": dt, "usage": usage})
        return data

    def run(self):
        """Run the daemon - will intermittently log gpu usage to disk.
        """
        while True:
            resources = parse_all_gpus()
            usage = gpu_usage(resources)
            log_row = self.serialize_usage(usage)
            timestamp = datetime.now().strftime(GPUStatDaemon.timestamp_format)
            with open(self.log_path, "a") as f:
                f.write(f"{timestamp} {log_row}\n")
            time.sleep(self.log_interval)


def historical_summary(data):
    """Print a short summary of the historical gpu usage logged by the daemon.

    Args:
        data (list): the data structure deserialized from the daemon log file (this is
            the output of the GPUStatDaemon.deserialize_usage() function.)
    """
    first_ts, last_ts = data[0]["timestamp"], data[-1]["timestamp"]
    print(f"Historical data contains {len(data)} samples ({first_ts} to {last_ts})")
    latest_usage = data[-1]["usage"]
    users, gpu_types = set(), set()
    for user, resources in latest_usage.items():
        users.add(user)
        gpu_types.update(set(resources.keys()))
    history = {}
    for row in data:
        for user, subdict in row["usage"].items():
            if user not in history:
                history[user] = {gpu_type: [] for gpu_type in gpu_types}
            type_counts = {key: sum(val.values()) for key, val in subdict.items()}
            for gpu_type in gpu_types:
                history[user][gpu_type].append(type_counts.get(gpu_type, 0))

    for user, subdict in history.items():
        print(f"GPU usage for {user}:")
        total = 0
        for gpu_type, counts in subdict.items():
            counts = np.array(counts)
            if counts.sum() == 0:
                continue
            print(f"{gpu_type:5s} > avg: {int(counts.mean())}, max: {np.max(counts)}")
            total += counts.mean()
        print(f"total > avg: {int(total)}\n")


def split_node_str(node_str):
    """Split SLURM node specifications into node_specs. Here a node_spec defines a range
    of nodes that share the same naming scheme (and are grouped together using square
    brackets).   E.g. 'node[1-3,4,6-9]' represents a single node_spec.

    Examples:
       A `node_str` of the form 'node[001-003]' will be returned as a single element
           list: ['node[001-003]']
       A `node_str` of the form 'node[001-002],node004' will be split into
           ['node[001-002]', 'node004']

    Args:
        node_str (str): a SLURM-formatted list of nodes

    Returns:
        (list[str]): SLURM node specs.
    """
    node_str = node_str.strip()
    breakpoints, stack = [0], []
    for ii, char in enumerate(node_str):
        if char == "[":
            stack.append(char)
        elif char == "]":
            stack.pop()
        elif not stack and char == ",":
            breakpoints.append(ii + 1)
    end = len(node_str) + 1
    return [node_str[i: j - 1] for i, j in zip(breakpoints, breakpoints[1:] + [end])]


def parse_node_names(node_str):
    """Parse the node list produced by the SLURM tools into separate node names.

    Examples:
       A slurm `node_str` of the form 'node[001-003]' will be split into a list of the
           form ['node001', 'node002', 'node003'].
       A `node_str` of the form 'node[001-002],node004' will be split into
           ['node001', 'node002', 'node004']

    Args:
        node_str (str): a SLURM-formatted list of nodes

    Returns:
        (list[str]): a list of separate node names.
    """
    names = []
    node_specs = split_node_str(node_str)
    for node_spec in node_specs:
        if "[" not in node_spec:
            names.append(node_spec)
        else:
            head, tail = node_spec.index("["), node_spec.index("]")
            prefix = node_spec[:head]
            subspecs = node_spec[head + 1:tail].split(",")
            for subspec in subspecs:
                if "-" not in subspec:
                    subnames = [f"{prefix}{subspec}"]
                else:
                    start, end = subspec.split("-")
                    num_digits = len(start)
                    subnames = [f"{prefix}{str(x).zfill(num_digits)}"
                                for x in range(int(start), int(end) + 1)]
                names.extend(subnames)
    return names


def parse_cmd(cmd, split=True):
    """Parse the output of a shell command...
     and if split set to true: split into a list of strings, one per line of output.

    Args:
        cmd (str): the shell command to be executed.
        split (bool): whether to split the output per line
    Returns:
        (list[str]): the strings from each output line.
    """
    output = subprocess.check_output(cmd, shell=True).decode("utf-8")
    if split:
        output = [x for x in output.split("\n") if x]
    return output


@beartype
def node_states() -> dict:
    """Query SLURM for the state of each managed node.

    Returns:
        a mapping between node names and SLURM states.
    """
    cmd = "sinfo --noheader"
    rows = parse_cmd(cmd)
    states = {}
    for row in rows:
        tokens = row.split()
        state, names = tokens[4], tokens[5]
        node_names = parse_node_names(names)
        states.update({name: state for name in node_names})
    return states


@beartype
@functools.lru_cache(maxsize=64, typed=True)
def occupancy_stats_for_node(node: str) -> dict:
    """Query SLURM for the occupancy of a given node.

    Args:
        (node): the name of the node to query

    Returns:
        a mapping between node names and occupancy stats.
    """
    cmd = f"scontrol show node {node}"
    rows = [x.strip() for x in parse_cmd(cmd)]
    keys = ("AllocTRES", "CfgTRES")
    metrics = {}
    for row in rows:
        for key in keys:
            if row.startswith(key):
                row = row.replace(f"{key}=", "")
                tokens = row.split(",")
                if tokens == [""]:
                    # SLURM sometimes omits information, so we alert the user to its
                    # its exclusion and report nothing for this node
                    print(f"Missing information for {node}: {key}, skipping....")
                    metrics[key] = {}
                else:
                    metrics[key] = {x.split("=")[0]: x.split("=")[1] for x in tokens}
    occupancy = {}
    for metric, alloc_val in metrics["AllocTRES"].items():
        cfg_val = metrics["CfgTRES"][metric]
        occupancy[metric] = f"{alloc_val}/{cfg_val}"
    return occupancy


@beartype
def parse_all_gpus(default_gpus: int = 4, default_gpu_name: str = "NONAME_GPU") -> dict:
    """Query SLURM for the number and types of GPUs under management.

    Args:
        default_gpus: The number of GPUs estimated for nodes that have incomplete SLURM
            meta data.

    Returns:
        a mapping between node names and a list of the GPUs that they have available.
    """
    cmd = "sinfo -o '%1000N|%30G' --noheader"
    rows = parse_cmd(cmd)
    resources = defaultdict(list)
    for row in rows:
        node_str, resource_strs = row.split("|")
        for resource_str in resource_strs.split(","):
            if not resource_str.startswith("gpu"):
                continue
            # Debug the regular expression below at
            # https://regex101.com/r/RHYM8Z/3
            p = re.compile(r'gpu:(?:(\w*):)?(\d*)(?:\(\S*\))?\s*')
            match = p.search(resource_str)
            gpu_type = match.group(1) if match.group(1) is not None else default_gpu_name
            # if the number of GPUs is not specified, we assume it is `default_gpus`
            gpu_count = int(match.group(2)) if match.group(2) != "" else default_gpus
            node_names = parse_node_names(node_str)
            for name in node_names:
                resources[name].append({"type": gpu_type, "count": gpu_count})
    return resources


@beartype
def resource_by_type(resources: dict) -> dict:
    """Determine the cluster capacity by gpu type

    Args:
        resources: a summary of the cluster resources, organised by node name.

    Returns:
        resources: a summary of the cluster resources, organised by gpu type
    """
    by_type = defaultdict(list)
    for node, specs in resources.items():
        for spec in specs:
            by_type[spec["type"]].append({"node": node, "count": spec["count"]})
    return by_type


@beartype
def summary_by_type(resources: dict, tag: str):
    """Print out out a summary of cluster resources, organised by gpu type.

    Args:
        resources (dict): a summary of cluster resources, organised by node name.
        tag (str): a term that will be included in the printed summary.
    """
    by_type = resource_by_type(resources)
    total = sum(x["count"] for sublist in by_type.values() for x in sublist)
    agg_str = []
    for key, val in sorted(by_type.items(), key=lambda x: sum(y["count"] for y in x[1])):
        gpu_count = sum(x["count"] for x in val)
        agg_str.append(f"{gpu_count} {key} gpus")
    print(f"There are a total of {total} gpus [{tag}]")
    print("\n".join(agg_str))


@beartype
def summary(mode: str, resources: dict = None, states: dict = None):
    """Generate a printed summary of the cluster resources.

    Args:
        mode (str): the kind of resources to query (must be one of 'accessible', 'up').
        resources (dict :: None): a summary of cluster resources, organised by node name.
        states (dict[str: str] :: None): a mapping between node names and SLURM states.
    """
    if not resources:
        resources = parse_all_gpus()
    if not states:
        states = node_states()
    if mode == "accessible":
        res = {key: val for key, val in resources.items()
               if states.get(key, "down") not in INACCESSIBLE}
    elif mode == "up":
        res = resources
    else:
        raise ValueError(f"Unknown mode: {mode}")
    summary_by_type(res, tag=mode)


@beartype
def gpu_usage(resources: dict) -> dict:
    """Build a data structure of the cluster resource usage, organised by user.

    Args:
        resources (dict :: None): a summary of cluster resources, organised by node name.

    Returns:
        (dict): a summary of resources organised by user (and also by node name).
    """
    cmd = "squeue -O tres-per-node,nodelist:100,username,jobid --noheader"
    detailed_job_cmd = "scontrol show jobid -dd %s"
    rows = parse_cmd(cmd)
    usage = defaultdict(dict)
    for row in rows:
        tokens = row.split()
        # ignore pending jobs
        if len(tokens) < 4 or not tokens[0].startswith("gpu"):
            continue
        gpu_count_str, node_str, user, jobid = tokens
        gpu_count_tokens = gpu_count_str.split(":")
        num_gpus = int(gpu_count_tokens[-1])
        if len(gpu_count_tokens) == 2:
            gpu_type = None
        elif len(gpu_count_tokens) == 3:
            gpu_type = gpu_count_tokens[1]
        # get detailed job information, to check if using bash
        detailed_output = parse_cmd(detailed_job_cmd % jobid, split=False)
        is_bash = any([f'Command={x}\n' in detailed_output for x in INTERACTIVE_CMDS])
        num_bash_gpus = num_gpus * is_bash
        node_names = parse_node_names(node_str)
        for node_name in node_names:
            # If a node still has jobs running but is draining, it will not be present
            # in the "available" resources, so we ignore it
            if node_name not in resources:
                continue
            node_gpu_types = [x["type"] for x in resources[node_name]]
            if gpu_type is None:
                if len(node_gpu_types) != 1:
                    gpu_type = random.choice(node_gpu_types)
                    msg = (f"cannot determine node gpu type for {user} on {node_name}"
                           f" (guesssing {gpu_type})")
                    print(f"WARNING >>> {msg}")
                else:
                    gpu_type = node_gpu_types[0]
            if gpu_type in usage[user]:
                usage[user][gpu_type][node_name]['n_gpu'] += num_gpus
                usage[user][gpu_type][node_name]['bash_gpu'] += num_bash_gpus

            else:
                usage[user][gpu_type] = defaultdict(lambda: {'n_gpu': 0, 'bash_gpu': 0})
                usage[user][gpu_type][node_name]['n_gpu'] += num_gpus
                usage[user][gpu_type][node_name]['bash_gpu'] += num_bash_gpus

    return usage


@beartype
def in_use(resources: dict = None):
    """Print a short summary of the resources that are currently used by each user.

    Args:
        resources: a summary of cluster resources, organised by node name.
    """
    if not resources:
        resources = parse_all_gpus()
    usage = gpu_usage(resources)
    aggregates = {}
    for user, subdict in usage.items():
        aggregates[user] = {}
        aggregates[user]['n_gpu'] = {key: sum([x['n_gpu'] for x in val.values()])
                                     for key, val in subdict.items()}
        aggregates[user]['bash_gpu'] = {key: sum([x['bash_gpu'] for x in val.values()])
                                        for key, val in subdict.items()}
    print("Usage by user:")
    for user, subdict in sorted(aggregates.items(),
                                key=lambda x: sum(x[1]['n_gpu'].values())):
        total = (f"total: {str(sum(subdict['n_gpu'].values())):2s} "
                 f"(interactive: {str(sum(subdict['bash_gpu'].values())):2s})")
        summary = ", ".join([f"{key}: {val}" for key, val in subdict['n_gpu'].items()])
        print(f"{user:10s} [{total}] {summary}")


@beartype
def available(
        resources: dict = None,
        states: dict = None,
        verbose: bool = False,
):
    """Print a short summary of resources available on the cluster.

    Args:
        resources: a summary of cluster resources, organised by node name.
        states: a mapping between node names and SLURM states.

    NOTES: Some systems allow users to share GPUs.  The logic below amounts to a
    conservative estimate of how many GPUs are available.  The algorithm is:

      For each user that requests a GPU on a node, we assume that a new GPU is allocated
      until all GPUs on the server are assigned.  If more GPUs than this are listed as
      allocated by squeue, we assume any further GPU usage occurs by sharing GPUs.
    """
    if not resources:
        resources = parse_all_gpus()
    if not states:
        states = node_states()
    res = {key: val for key, val in resources.items()
           if states.get(key, "down") not in INACCESSIBLE}
    usage = gpu_usage(resources=res)
    for subdict in usage.values():
        for gpu_type, node_dicts in subdict.items():
            for node_name, user_gpu_count in node_dicts.items():
                resource_idx = [x["type"] for x in res[node_name]].index(gpu_type)
                count = res[node_name][resource_idx]["count"]
                count = max(count - user_gpu_count['n_gpu'], 0)
                res[node_name][resource_idx]["count"] = count
    by_type = resource_by_type(res)
    total = sum(x["count"] for sublist in by_type.values() for x in sublist)
    print(f"There are {total} gpus available:")
    for key, counts_for_gpu_type in by_type.items():
        gpu_count = sum(x["count"] for x in counts_for_gpu_type)
        tail = ""
        if verbose:
            summary = []
            for x in counts_for_gpu_type:
                node, count = x["node"], x["count"]
                if count:
                    occupancy = occupancy_stats_for_node(node)
                    users = [user for user in usage if node in usage[user].get(key, [])]
                    details = [f"{key}: {val}" for key, val in sorted(occupancy.items())]
                    details = f"[{', '.join(details)}] [{','.join(users)}]"
                    summary.append(f"\n -> {node}: {count} {key} {details}")
            tail = " ".join(summary)
        print(f"{key}: {gpu_count} available {tail}")


@beartype
def all_info(color: int, verbose: bool):
    """Print a collection of summaries about SLURM gpu usage, including: all nodes
    managed by the cluster, nodes that are currently accesible and gpu usage for each
    active user.
    """
    divider, slurm_str = "---------------------------------", "SLURM"
    if color:
        colors = sns.color_palette("hls", 8).as_hex()
        divider = colored.stylize(divider, colored.fg(colors[7]))
        slurm_str = colored.stylize(slurm_str, colored.fg(colors[0]))
    print(divider)
    print(f"Under {slurm_str} management")
    print(divider)
    resources = parse_all_gpus()
    states = node_states()
    for mode in ("up", "accessible"):
        summary(mode=mode, resources=resources, states=states)
        print(divider)
    in_use(resources)
    print(divider)
    available(resources=resources, states=states, verbose=verbose)
    print(divider)


def main():
    parser = argparse.ArgumentParser(description="slurm_gpus tool")
    parser.add_argument("--action", default="current",
                        choices=["current", "history", "daemon-start", "daemon-stop"],
                        help=("The function performed by slurm_gpustat: `current` will"
                              " provide a summary of current usage, 'history' will "
                              "provide statistics from historical data (provided that the"
                              "logging daemon has been running). 'daemon-start' and"
                              "'daemon-stop' will start and stop the daemon, resp."))
    parser.add_argument("--log_path",
                        default=Path.home() / "data/daemons/logs/slurm_gpustat.log",
                        help="the location where daemon log files will be stored")
    parser.add_argument("--gpustat_pid",
                        default=Path.home() / "data/daemons/pids/slurm_gpustat.pid",
                        help="the location where the daemon PID file will be stored")
    parser.add_argument("--daemon_log_interval", type=int, default=43200,
                        help="time interval (secs) between stat logging (default 12 hrs)")
    parser.add_argument("--color", type=int, default=1, help="color output")
    parser.add_argument("--verbose", action="store_true",
                        help="provide a more detailed breakdown of resources")
    args = parser.parse_args()

    if args.action == "current":
        all_info(color=args.color, verbose=args.verbose)
    elif args.action == "history":
        data = GPUStatDaemon.deserialize_usage(args.log_path)
        historical_summary(data)
    elif args.action.startswith("daemon"):
        daemon = GPUStatDaemon(
            log_path=args.log_path,
            pidfile=args.gpustat_pid,
            log_interval=args.daemon_log_interval,
        )
        if args.action == "daemon-start":
            print("Starting daemon")
            daemon.start()
        elif args.action == "daemon-stop":
            print("Stopping daemon")
            daemon.stop()


if __name__ == "__main__":
    main()
