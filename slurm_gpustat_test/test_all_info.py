"""Test suite for slurm_gpustat/slurm_gpustat.py
"""

from slurm_gpustat.slurm_gpustat import all_info


def test_expected_current_output_no_color():
    all_info(color=0, verbose=False, partition=None)


def test_expected_current_output_color():
    all_info(color=1, verbose=False, partition=None)


def test_expected_current_output_no_color_verbose():
    all_info(color=0, verbose=True, partition=None)


def test_expected_current_output_color_verbose():
    all_info(color=1, verbose=True, partition=None)


if __name__ == "__main__":
    test_expected_current_output_color()
    test_expected_current_output_no_color()
    test_expected_current_output_color_verbose()
    test_expected_current_output_no_color_verbose()
