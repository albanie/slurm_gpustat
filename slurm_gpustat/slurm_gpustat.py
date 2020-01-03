"""A simple tool for summarising GPU statistics on a slurm cluster
"""
import argparse
import ast
import random
import subprocess
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from daemon import Daemon

INACCESSIBLE = {"drain*", "down*", "drng", "drain", "down"}


class GPUStatDaemon(Daemon):
    timestamp_format = "%Y-%m-%d_%H:%M:%S"

    def __init__(self, pidfile, log_path, log_interval):
        Path(pidfile).parent.mkdir(exist_ok=True, parents=True)
        super().__init__(pidfile=pidfile)
        Path(log_path).parent.mkdir(exist_ok=True, parents=True)
        self.log_interval = log_interval
        self.log_path = log_path

    def run(self):
        print("Running GPUStatDaemon")
        while True:
            resources = parse_all_gpus()
            usage = gpu_usage(resources)
            log_row = self.serialize_usage(usage)
            timestamp = datetime.now().strftime(GPUStatDaemon.timestamp_format)
            with open(self.log_path, "a") as f:
                f.write(f"{timestamp} {log_row}\n")
            time.sleep(self.log_interval)

    def serialize_usage(self, usage):
        """Convert data structure into an appropriate string for serialization"""
        for user, gpu_dict in usage.items():
            for key, subdict in gpu_dict.items():
                usage[user][key] = dict(subdict)
        usage = dict(usage)
        return usage.__repr__()

    @staticmethod
    def deserialize_usage(log_path):
        """Convert csv -> list of dicts"""
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


def historical_summary(data):
    first_ts, last_ts = data[0]["timestamp"], data[1]["timestamp"]
    print(f"Historical data contains {len(data)} samples ({first_ts} to {last_ts})")
    latest_usage = data[-1]["usage"]
    users, gpu_types = set(), set()
    for user, resources in latest_usage.items():
        users.add(user)
        gpu_types.update(set(resources.keys()))
    history = {user: {gpu_type: [] for gpu_type in gpu_types} for user in users}
    for row in data:
        for user, subdict in row["usage"].items():
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


def parse_cmd(cmd):
    output = subprocess.check_output(cmd, shell=True).decode("utf-8")
    rows = [x for x in output.split("\n") if x]
    return rows


def node_states():
    cmd = "sinfo --noheader"
    rows = parse_cmd(cmd)
    states = {}
    for row in rows:
        tokens = row.split()
        state, names = tokens[4], tokens[5]
        node_names = parse_node_names(names)
        states.update({name: state for name in node_names})
    return states


def parse_all_gpus(default_gpus=4):
    cmd = "sinfo -o '%50N|%30G' --noheader"
    rows = parse_cmd(cmd)
    resources = defaultdict(list)
    for row in rows:
        node_str, resource_strs = row.split("|")
        for resource_str in resource_strs.split(","):
            if not resource_str.startswith("gpu"):
                continue
            tokens = resource_str.strip().split(":")
            # if the number of GPUs is not specified, we assume it is `default_gpus`
            if tokens[2] == "":
                tokens[2] = default_gpus
            gpu_type, gpu_count = tokens[1], int(tokens[2])
            node_names = parse_node_names(node_str)
            for name in node_names:
                resources[name].append({"type": gpu_type, "count": gpu_count})
    return resources


def resource_by_type(resources):
    by_type = defaultdict(lambda: 0)
    for specs in resources.values():
        for spec in specs:
            by_type[spec["type"]] += spec["count"]
    return by_type


def summary_by_type(resources, tag):
    by_type = resource_by_type(resources)
    total = sum(by_type.values())
    agg_str = []
    for key, val in sorted(by_type.items(), key=lambda x: x[1]):
        agg_str.append(f"{val} {key} gpus")
    print(f"There are a total of {total} gpus [{tag}]")
    print("\n".join(agg_str))


def summary(tag, resources=None, states=None):
    if not resources:
        resources = parse_all_gpus()
    if not states:
        states = node_states()
    if tag == "accessible":
        res = {key: val for key, val in resources.items()
               if states.get(key, "down") not in INACCESSIBLE}
    elif tag == "up":
        res = resources
    else:
        raise ValueError(f"Unknown tag: {tag}")
    summary_by_type(res, tag=tag)


def gpu_usage(resources):
    cmd = "squeue -O tres-per-node,nodelist,username --noheader"
    rows = parse_cmd(cmd)
    usage = defaultdict(dict)
    for row in rows:
        tokens = row.split()
        # ignore pending jobs
        if len(tokens) < 3 or not tokens[0].startswith("gpu"):
            continue
        gpu_count_str, node_str, user = tokens
        gpu_count_tokens = gpu_count_str.split(":")
        num_gpus = int(gpu_count_tokens[-1])
        if len(gpu_count_tokens) == 2:
            gpu_type = None
        elif len(gpu_count_tokens) == 3:
            gpu_type = gpu_count_tokens[1]
        node_names = parse_node_names(node_str)
        for node_name in node_names:
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
                usage[user][gpu_type][node_name] += num_gpus
            else:
                usage[user][gpu_type] = defaultdict(lambda: 0)
                usage[user][gpu_type][node_name] += num_gpus
    return usage


def in_use(resources=None):
    if not resources:
        resources = parse_all_gpus()
    usage = gpu_usage(resources)
    aggregates = {}
    for user, subdict in usage.items():
        aggregates[user] = {key: sum(val.values()) for key, val in subdict.items()}
    print("Usage by user:")
    for user, subdict in sorted(aggregates.items(), key=lambda x: sum(x[1].values())):
        total = f"total: {str(sum(subdict.values())):2s}"
        summary_str = ", ".join([f"{key}: {val}" for key, val in subdict.items()])
        print(f"{user:10s} [{total}] {summary_str}")


def available(resources=None, states=None):
    """Some systems allow users to share GPUs.  The logic below amounts to a conservative
    estimate of how many GPUs are available.  The logic is essentially:

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
                count = max(count - user_gpu_count, 0)
                res[node_name][resource_idx]["count"] = count
    by_type = resource_by_type(res)
    total = sum(by_type.values())
    print(f"There are {total} gpus available:")
    for key, val in by_type.items():
        print(f"{key}: {val}")


def all_info():
    divider = "----------------------"
    print(divider)
    print("Under SLURM management")
    print(divider)
    resources = parse_all_gpus()
    states = node_states()
    for tag in ("up", "accessible"):
        summary(tag=tag, resources=resources, states=states)
        print(divider)
    in_use(resources)
    print(divider)
    available(resources=resources, states=states)


def main():
    parser = argparse.ArgumentParser(description="slurm_gpus tool")
    parser.add_argument("--action", default="current",
                        choices=["current", "history", "daemon-start", "daemon-stop"])
    parser.add_argument("--log_path",
                        default=Path.home() / "data/daemons/logs/slurm_gpustat.log")
    parser.add_argument("--gpustat_pid",
                        default=Path.home() / "data/daemons/pids/slurm_gpustat.pid")
    parser.add_argument("--daemon_log_interval", type=int, default=43200,
                        help="time interval (secs) between stat logging (default 12 hrs)")
    args = parser.parse_args()

    if args.action == "current":
        all_info()
    elif args.action == "history":
        data = GPUStatDaemon.deserialize_usage(args.log_path)
        historical_summary(data)
    elif args.action.startswith("daemon"):
        daemon = GPUStatDaemon(
            log_path=args.log_path,
            log_interval=args.daemon_log_interval,
            pidfile=args.gpustat_pid,
        )
        if args.action == "daemon-start":
            print("Starting daemon")
            daemon.start()
        elif args.action == "daemon-stop":
            print("Stopping daemon")
            daemon.stop()


if __name__ == "__main__":
    main()
