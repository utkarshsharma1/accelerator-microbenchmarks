"""This script runs microbenchmarks and collects metrics.

Sample usage (on TPU vm):
  $ python src/run_benchmark.py --config=configs/benchmark_collectives.yaml
"""

import argparse
import csv
import datetime
import importlib
import inspect
import itertools
import random
import string
from typing import Any, Callable, Dict, List, Tuple
from benchmark_utils import maybe_write_metrics_file, rename_xla_dump
import jax
import yaml
import ray
from concurrent.futures import ThreadPoolExecutor
import os
import copy
import sys
import shutil

COLLECTIVE_BENCHMARK_MAP = {
    "all_gather": "benchmark_collectives.all_gather_benchmark",
    "psum": "benchmark_collectives.psum_benchmark",
    "psum_scatter": "benchmark_collectives.psum_scatter_benchmark",
    "all_to_all": "benchmark_collectives.all_to_all_benchmark",
    "ppermute": "benchmark_collectives.ppermute_benchmark",
}

MATMUL_BENCHMARK_MAP = {
    "naive_matmul": "benchmark_matmul.naive_matmul",
    "single_host_naive_matmul": "benchmark_matmul.single_host_naive_matmul",
    "multilayer_collective_matmul": ("benchmark_matmul.multilayer_collective_matmul"),
    "collective_matmul_one_direction": (
        "benchmark_matmul.collective_matmul_one_direction"
    ),
    "collective_matmul_two_directions": (
        "benchmark_matmul.collective_matmul_two_directions"
    ),
}
CONVOLUTION_BENCHMARK_MAP = {
    "numpy_convolve": "benchmark_convolution.numpy_convolve",
    "scipy_signal_convolve": "benchmark_convolution.scipy_signal_convolve",
    "scipy_signal_convolve2d": "benchmark_convolution.scipy_signal_convolve2d",
    "lax_conv_general_dilated": ("benchmark_convolution.lax_conv_general_dilated"),
}
ATTENTION_BENCHMARK_MAP = {
    "naive_attention": "benchmark_attention.naive_attention_benchmark",
    "pallas_flash_attention": ("benchmark_attention.pallas_flash_attention_benchmark"),
    "splash_attention": "benchmark_attention.splash_attention_benchmark",
    "flax_nnx_attention": "benchmark_attention.flax_nnx_attention_benchmark",
    "flax_linen_attention": ("benchmark_attention.flax_linen_attention_benchmark"),
    "keras_attention": "benchmark_attention.keras_attention_benchmark",
}
HBM_BENCHMARK_MAP = {
    "single_chip_hbm_copy": "benchmark_hbm.single_chip_hbm_copy",
}
BENCHMARK_MAP = {}
BENCHMARK_MAP.update(COLLECTIVE_BENCHMARK_MAP)
BENCHMARK_MAP.update(MATMUL_BENCHMARK_MAP)
BENCHMARK_MAP.update(CONVOLUTION_BENCHMARK_MAP)
BENCHMARK_MAP.update(ATTENTION_BENCHMARK_MAP)
BENCHMARK_MAP.update(HBM_BENCHMARK_MAP)


# Mapping from dtype string to actual dtype object
dtype_mapping = {
    "bfloat16": jax.numpy.bfloat16,
    "float32": jax.numpy.float32,
    "int32": jax.numpy.int32,
    # Add other dtypes as needed
}

# Always dump HLOs
TMP_XLA_DUMP_DIR = "/tmp/microbenchmarks/hlo_graphs"

# Fixed local directory for collecting metrics
LOCAL_METRICS_OUTPUT_DIR = "/tmp/microbenchmarks/outputs"

def get_benchmark_config(config_path: str) -> Dict[str, Any]:
    """Load benchmark configuration from a YAML file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


# Dynamically load the benchmark functions.
def get_benchmark_functions(
    benchmark_name: str,
) -> Tuple[Callable[..., Any], Callable[..., Any]]:
    """Dynamically load the benchmark function and its calculate_metrics function from the predefined map."""
    if benchmark_name not in BENCHMARK_MAP:
        raise ValueError(f"Benchmark {benchmark_name} is not defined in the map.")

    module_path, func_name = BENCHMARK_MAP[benchmark_name].rsplit(".", 1)

    try:
        module = importlib.import_module(f"{module_path}")
        benchmark_func = getattr(module, func_name)
    except ModuleNotFoundError as e:
        raise ValueError(
            f"Unable to import {module_path}.{func_name}. ModuleNotFoundError {e}."
        ) from e
    except AttributeError as e:
        raise ValueError(
            f"Unable to import {module_path}.{func_name}. AttributeError {e}."
        ) from e

    try:
        calculate_metrics_func = getattr(module, f"{func_name}_calculate_metrics")
    except AttributeError:
        raise ValueError(
            f"Calculate metrics function for {benchmark_name} not found."
        ) from None

    return benchmark_func, calculate_metrics_func


def preprocess_benchmark_param(
    benchmark_param: Dict[str, Any], trace_dir: str = None
) -> Dict[str, Any]:
    """Preprocess the benchmark parameter before running the benchmark."""
    if "dtype" in benchmark_param:
        dtype_str = benchmark_param["dtype"]
        if dtype_str in dtype_mapping:
            benchmark_param["dtype"] = dtype_mapping[dtype_str]
        else:
            raise ValueError(f"Unsupported dtype: {dtype_str}")

    for key, value in benchmark_param.items():
        if isinstance(value, str) and value.startswith("SAME_AS_"):
            same_as_key = value.split("SAME_AS_")[1]
            if same_as_key not in benchmark_param:
                raise ValueError(
                    f"Parameter {same_as_key} not found in the benchmark_param."
                )
            benchmark_param[key] = benchmark_param[same_as_key]

    benchmark_param["trace_dir"] = trace_dir
    return benchmark_param


def generate_benchmark_params_sweeping(
    benchmark_sweep_params: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Generate benchmark parameters by sweeping through the specified ranges."""
    generated_params = []
    for sweep_params in benchmark_sweep_params:
        param_sets = {}
        for key, value in sweep_params.items():
            if key.endswith("_range"):
                key = key[:-6]

            if isinstance(value, dict):
                start = value.get("start")
                end = value.get("end")
                multiplier = value.get("multiplier", None)
                increase_by = value.get("increase_by", None)
                param_values = []
                current_value = start
                while current_value <= end:
                    param_values.append(current_value)
                    if multiplier:
                        current_value *= multiplier
                    elif increase_by:
                        current_value += increase_by
                    else:
                        raise ValueError("Must provide either multiplier or increase_by.")
                param_sets[key] = param_values
            else:
                param_sets[key] = [value]

        param_names = list(param_sets.keys())
        combinations = [
            dict(zip(param_names, values))
            for values in itertools.product(*(param_sets[name] for name in param_names))
        ]
        generated_params += combinations
    return generated_params


def run_single_benchmark(benchmark_config: Dict[str, Any], common_metrics_dir: str):
    """Run a single benchmark with one or more configurations."""
    benchmark_name = benchmark_config.get("benchmark_name")
    benchmark_params = benchmark_config.get("benchmark_params", [])
    benchmark_sweep_params = benchmark_config.get("benchmark_sweep_params", {})
    if benchmark_sweep_params:
        benchmark_params += generate_benchmark_params_sweeping(benchmark_sweep_params)
    xla_dump_dir = benchmark_config.get("xla_dump_dir")

    if not benchmark_name:
        raise ValueError("Each benchmark must have a 'benchmark_name'.")

    benchmark_func, calculate_metrics_func = get_benchmark_functions(benchmark_name)
    print(f"\n{'=' * 30}Starting benchmark '{benchmark_name}'{'=' * 30}\n")

    for benchmark_param in benchmark_params:
        original_benchmark_param = copy.deepcopy(benchmark_param)
        benchmark_param = preprocess_benchmark_param(benchmark_param, trace_dir=None)
        print(f"Running benchmark: {benchmark_name} with params: {benchmark_param}")
        test_start_time = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()
        benchmark_results = benchmark_func(**benchmark_param)
        test_end_time = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()

        calculate_metrics_params = inspect.signature(calculate_metrics_func).parameters
        filtered_benchmark_results = {
            key: value
            for key, value in benchmark_results.items()
            if key in calculate_metrics_params
        }
        benchmark_params_to_filter = ["num_runs", "trace_dir"]
        filtered_benchmark_param = {
            key: value
            for key, value in benchmark_param.items()
            if key not in benchmark_params_to_filter
        }
        metadata, metrics = calculate_metrics_func(
            **filtered_benchmark_param, **filtered_benchmark_results
        )

        maybe_write_metrics_file(
            common_metrics_dir,
            metrics,
            metadata,
            benchmark_name,
            test_start_time,
            test_end_time,
        )
        if xla_dump_dir:
            rename_xla_dump(
                tmp_xla_dump_dir=TMP_XLA_DUMP_DIR,
                dest_xla_dump_dir=os.path.join(LOCAL_METRICS_OUTPUT_DIR, xla_dump_dir),
                benchmark_name=benchmark_name,
                benchmark_param=original_benchmark_param,
            )

def main(config_path: str, multithreaded: bool):
    """Main function."""
    config = get_benchmark_config(config_path)
    benchmarks = config.get("benchmarks")
    if not benchmarks or not isinstance(benchmarks, list):
        raise ValueError("Configuration must contain a 'benchmarks' list.")

    if os.path.exists(LOCAL_METRICS_OUTPUT_DIR):
      shutil.rmtree(LOCAL_METRICS_OUTPUT_DIR)
    os.makedirs(LOCAL_METRICS_OUTPUT_DIR)

    if os.path.exists(TMP_XLA_DUMP_DIR):
      shutil.rmtree(TMP_XLA_DUMP_DIR)
    os.makedirs(TMP_XLA_DUMP_DIR)
    os.environ["XLA_FLAGS"] = f"--xla_dump_to={TMP_XLA_DUMP_DIR}"

    if multithreaded:
        print("Multithreaded mode not fully supported in this version.")
    else:
        for benchmark_config in benchmarks:
            run_single_benchmark(benchmark_config, LOCAL_METRICS_OUTPUT_DIR)

    print(f"Benchmark outputs are in {LOCAL_METRICS_OUTPUT_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run microbenchmarks and collect metrics.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--multithreaded", action='store_true', help="Run benchmarks in multithreaded mode using Ray.")
    args = parser.parse_args()
    main(args.config, args.multithreaded)
