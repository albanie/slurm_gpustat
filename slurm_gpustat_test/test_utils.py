import unittest
from slurm_gpustat.slurm_gpustat import parse_gpu_type_and_count_via_regex


class Tests(unittest.TestCase):

    def test_regex_for_parsing_gpu_types(self):
        DEFAULT_GPUS = 4
        DEFAULT_GPU_NAME = "NONAME_GPU"
        test_cases = [
            {
                "test_str": "gpu:Tesla_V100:4(S:0-1)",
                "expected_gpu_type": "Tesla_V100",
                "expected_num_gpus": 4,
             },
            {
                "test_str": "gpu:Tesla_V100:4",
                "expected_gpu_type": "Tesla_V100",
                "expected_num_gpus": 4,
            },
            {
                "test_str": "gpu:Tesla_V100:1",
                "expected_gpu_type": "Tesla_V100",
                "expected_num_gpus": 1,
            },
            {
                "test_str": "gpu:Tesla_V100:",
                "expected_gpu_type": "Tesla_V100",
                "expected_num_gpus": DEFAULT_GPUS,
            },
            {
                "test_str": "gpu:3(S:0-1)",
                "expected_gpu_type": DEFAULT_GPU_NAME,
                "expected_num_gpus": 3,
            },
            {
                "test_str": "gpu:nvidia_a100:4(S:0-1)",
                "expected_gpu_type": "nvidia_a100",
                "expected_num_gpus": 4,
            },
        ]
        for test_case in test_cases:
            gpu_type, gpu_count = parse_gpu_type_and_count_via_regex(test_case["test_str"])
            self.assertEqual(test_case["expected_gpu_type"], gpu_type)
            self.assertEqual(test_case["expected_num_gpus"], gpu_count)


if __name__ == "__main__":
    unittest.main()
