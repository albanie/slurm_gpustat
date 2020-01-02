"""A simple tool for summarising GPU statistics on a slurm cluster
"""
import os
import argparse
from collections import defaultdict
import subprocess

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
                    subnames = [f"{prefix}{x}" for x in range(int(start), int(end) + 1)]
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
    cmd = "sinfo -o '%50N|%10G'"
    rows = parse_cmd(cmd)
    resources = {}
    for row in rows:
        node_str, resource_str = row.split("|")
        if not resource_str.startswith("gpu"):
            continue
        tokens = resource_str.strip().split(":")
        # if the number of GPUs is not specified, we assume it is `default_gpus`
        if tokens[2] == "":
            tokens[2] = default_gpus
        gpu_type, gpu_count = tokens[1], int(tokens[2])
        node_names = parse_node_names(node_str)
        res = {name: {"type": gpu_type, "count": gpu_count} for name in node_names}
        resources.update(res)
    return resources


def resource_by_type(resources):
    by_type = defaultdict(lambda: 0)
    for spec in resources.values():
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
               if states[key] not in {"drain*", "down*", "drng"}}
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
        num_gpus = int(gpu_count_str.split(":")[1])
        node_names = parse_node_names(node_str)
        for node_name in node_names:
            gpu_type = resources[node_name]["type"]
            if gpu_type in usage[user]:
                usage[user][gpu_type] += num_gpus
            else:
                usage[user][gpu_type] = num_gpus
    return usage


def in_use(resources=None):
    if not resources:
        resources = parse_all_gpus()
    usage = gpu_usage(resources)
    print("Usage by user:")
    for user, subdict in sorted(usage.items(), key=lambda x: sum(x[1].values())):
        total = f"total: {str(sum(subdict.values())):2s}"
        summary_str = ", ".join([f"{key}: {val}" for key, val in subdict.items()])
        print(f"{user:10s} [{total}] {summary_str}")


def available(resources=None, states=None):
    if not resources:
        resources = parse_all_gpus()
    if not states:
        states = node_states()
    usage = gpu_usage(resources)
    res = {key: val for key, val in resources.items()
           if states[key] not in {"drain*", "down*", "drng"}}
    by_type = resource_by_type(res)
    for subdict in usage.values():
        for gpu_type, count in subdict.items():
            by_type[gpu_type] -= count
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
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--in_use", action="store_true")
    parser.add_argument("--available", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.all:
        all_info()
    if args.in_use:
        in_use()
    if args.available:
        available()


if __name__ == "__main__":
    main()
