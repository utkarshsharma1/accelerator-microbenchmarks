"""A script to run the microbenchmarks in Jax over DCN and ICI collectives."""

# pylint: disable=g-importing-member
import argparse
import datetime
from functools import partial
import random
import string

from benchmark_utils import maybe_write_metrics_file
from benchmark_utils import simple_timeit
import jax
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
import numpy as np
# pylint: disable=g-importing-member

TRACE_BASE_DIR = None
METRICS_JSONL_DIR = None
MATRIX_STEP_SIZE = 1024
MATRIX_START_SIZE = 1024
MATRIX_MAX_SIZE = 30000


def create_mesh(dcn_size=2, ici_size=4):
  """Creates a hybrid mesh with the given DCN and ICI sizes."""
  dcn_parallelism = [dcn_size, 1]
  ici_parallelism = [1, ici_size]

  total_devices = jax.device_count()
  if total_devices != (dcn_size * ici_size):
    raise ValueError(
        f"Need {dcn_size * ici_size} devices, but found {total_devices}"
    )
  if dcn_size > 1:
    mesh_devices = mesh_utils.create_hybrid_device_mesh(
        ici_parallelism, dcn_parallelism, devices=jax.devices()
    )
    mesh = Mesh(mesh_devices, ("dcn", "ici"))
  else:
    mesh_devices = mesh_utils.create_device_mesh(
        [ici_size], devices=jax.devices()
    )
    mesh = Mesh(mesh_devices, ("ici"))
  return mesh, dcn_parallelism, ici_parallelism



def psum_benchmark(matrix_dim, dcn_size=2, ici_size=4):
  """Benchmark the psum collective operation."""
  dcn_bandwidths = {}
  ici_bandwidths = {}
  dtype = jax.numpy.bfloat16
  mesh, _, _ = create_mesh(dcn_size, ici_size)
  matrix = jnp.ones((matrix_dim, matrix_dim), dtype=dtype)
  matrix_size_gbyte = matrix.size * dtype.dtype.itemsize / 1e9

  # DCN benchmark
  if dcn_size > 1:

    @partial(shard_map, mesh=mesh, in_specs=P("dcn", None), out_specs=P(None))
    def psum_dcn_op(x):
      return jax.lax.psum(x, "dcn")

    sharded_matrix = jax.device_put(
        matrix, jax.sharding.NamedSharding(mesh, P("dcn", None))
    )
    jitted_op = jax.jit(psum_dcn_op)
    average_time_ms = simple_timeit(
        jitted_op, sharded_matrix, task="psum_dcn_op"
    )

    # bandwidth is claculated as psum can be done via reduce_scatter +
    # all_gather so bandwidth is the sum of the two (formulas below)
    achieved_bandwidth_gbyte_s = (
        matrix_size_gbyte
        * (dcn_size - 1)
        * 2
        / dcn_size
        / dcn_size
        / (average_time_ms / 1e3)
    )
    dcn_bandwidths[matrix_size_gbyte] = achieved_bandwidth_gbyte_s
    print(
        f"psum_dcn: Matrix size: {matrix_dim}x{matrix_dim}, {dtype=}, "
        f"{matrix_size_gbyte=}, {achieved_bandwidth_gbyte_s=}"
    )

  # ICI benchmark
  if ici_size > 1:

    @partial(shard_map, mesh=mesh, in_specs=P(None, "ici"), out_specs=P(None))
    def psum_ici_op(x):
      return jax.lax.psum(x, "ici")

    sharded_matrix = jax.device_put(
        matrix, jax.sharding.NamedSharding(mesh, P(None, "ici"))
    )
    jitted_op = jax.jit(psum_ici_op)
    average_time_ms = simple_timeit(
        jitted_op, sharded_matrix, task="psum_ici_op"
    )

    # bandwidth is claculated as psum can be done via reduce_scatter +
    # all_gather so bandwidth is the sum of the two (formulas below)
    achieved_bandwidth_gbyte_s = (
        matrix_size_gbyte
        * (ici_size - 1)
        * 2
        / ici_size
        / ici_size
        / (average_time_ms / 1e3)
    )
    ici_bandwidths[matrix_size_gbyte] = achieved_bandwidth_gbyte_s
    print(
        f"psum_ici: Matrix size: {matrix_dim}x{matrix_dim}, {dtype=}, "
        f"{matrix_size_gbyte=}, {achieved_bandwidth_gbyte_s=}"
    )

  return dcn_bandwidths, ici_bandwidths


def psum_scatter_benchmark(matrix_dim, dcn_size=2, ici_size=4):
  """Benchmark the psum_scatter collective operation."""
  dcn_bandwidths = {}
  ici_bandwidths = {}
  dtype = jax.numpy.bfloat16
  mesh, _, _ = create_mesh(dcn_size, ici_size)
  matrix = jnp.ones((matrix_dim, matrix_dim), dtype=dtype)
  matrix_size_gbyte = matrix.size * dtype.dtype.itemsize / 1e9

  # DCN benchmark
  if dcn_size > 1:

    @partial(
        shard_map, mesh=mesh, in_specs=P("dcn", None), out_specs=P("dcn", None)
    )
    def psum_scatter_dcn_op(x):
      return jax.lax.psum_scatter(x, "dcn", tiled=True)

    sharded_matrix = jax.device_put(
        matrix, jax.sharding.NamedSharding(mesh, P("dcn", None))
    )
    jitted_op = jax.jit(psum_scatter_dcn_op)
    average_time_ms = simple_timeit(
        jitted_op, sharded_matrix, task="psum_scatter_dcn_op"
    )

    # each sharded matrix size is matrix_size_gbyte / dcn_size and then it needs
    # to use (dcn_size - 1) steps in a ring algorithm
    achieved_bandwidth_gbyte_s = (
        matrix_size_gbyte
        * (dcn_size - 1)
        / dcn_size
        / dcn_size
        / (average_time_ms / 1e3)
    )
    dcn_bandwidths[matrix_size_gbyte] = achieved_bandwidth_gbyte_s
    print(
        f"psum_scatter_dcn: Matrix size: {matrix_dim}x{matrix_dim}, {dtype=}, "
        f"{matrix_size_gbyte=}, {achieved_bandwidth_gbyte_s=}"
    )

  # ICI benchmark
  if ici_size > 1:

    @partial(
        shard_map, mesh=mesh, in_specs=P(None, "ici"), out_specs=P(None, "ici")
    )
    def psum_scatter_ici_op(x):
      return jax.lax.psum_scatter(x, "ici", tiled=True)

    sharded_matrix = jax.device_put(
        matrix, jax.sharding.NamedSharding(mesh, P(None, "ici"))
    )
    jitted_op = jax.jit(psum_scatter_ici_op)
    average_time_ms = simple_timeit(
        jitted_op, sharded_matrix, task="psum_scatter_ici_op"
    )

    # each sharded matrix size is matrix_size_gbyte / ici_size and then it needs
    # to use (ici_size - 1) steps in a ring algorithm
    achieved_bandwidth_gbyte_s = (
        matrix_size_gbyte
        * (ici_size - 1)
        / ici_size
        / ici_size
        / (average_time_ms / 1e3)
    )
    ici_bandwidths[matrix_size_gbyte] = achieved_bandwidth_gbyte_s
    print(
        f"psum_scatter_ici: Matrix size: {matrix_dim}x{matrix_dim}, {dtype=}, "
        f"{matrix_size_gbyte=}, {achieved_bandwidth_gbyte_s=}"
    )

  return dcn_bandwidths, ici_bandwidths


def all_gather_benchmark(matrix_dim, dcn_size=2, ici_size=4):
  """Benchmark the all_gather collective operation."""
  dcn_bandwidths = {}
  ici_bandwidths = {}
  dtype = jax.numpy.bfloat16
  mesh, _, _ = create_mesh(dcn_size, ici_size)
  matrix = jnp.ones((matrix_dim, matrix_dim), dtype=dtype)
  matrix_size_gbyte = matrix.size * dtype.dtype.itemsize / 1e9

  # DCN benchmark
  if dcn_size > 1:

    @partial(
        shard_map, mesh=mesh, in_specs=P("dcn", None), out_specs=P("dcn", None)
    )
    def all_gather_dcn_op(x):
      return jax.lax.all_gather(x, "dcn", tiled=True)

    sharded_matrix = jax.device_put(
        matrix, jax.sharding.NamedSharding(mesh, P("dcn", None))
    )
    jitted_op = jax.jit(all_gather_dcn_op)
    average_time_ms = simple_timeit(
        jitted_op, sharded_matrix, task="all_gather_dcn_op"
    )

    # each sharded matrix size is matrix_size_gbyte / dcn_size and then it needs
    # to use (dcn_size - 1) steps in a ring algorithm
    achieved_bandwidth_gbyte_s = (
        matrix_size_gbyte * (dcn_size - 1) / dcn_size / (average_time_ms / 1e3)
    )
    dcn_bandwidths[matrix_size_gbyte] = achieved_bandwidth_gbyte_s
    print(
        f"all_gather_dcn: Matrix size: {matrix_dim}x{matrix_dim}, {dtype=}, "
        f"{matrix_size_gbyte=}, {achieved_bandwidth_gbyte_s=}"
    )

  # ICI benchmark
  if ici_size > 1:

    @partial(
        shard_map, mesh=mesh, in_specs=P(None, "ici"), out_specs=P(None, "ici")
    )
    def all_gather_ici_op(x):
      return jax.lax.all_gather(x, "ici", tiled=True)

    sharded_matrix = jax.device_put(
        matrix, jax.sharding.NamedSharding(mesh, P(None, "ici"))
    )
    jitted_op = jax.jit(all_gather_ici_op)
    average_time_ms = simple_timeit(
        jitted_op, sharded_matrix, task="all_gather_ici_op"
    )

    # each sharded matrix size is matrix_size_gbyte / ici_size and then it needs
    # to use (ici_size - 1) steps in a ring algorithm
    achieved_bandwidth_gbyte_s = (
        matrix_size_gbyte * (ici_size - 1) / ici_size / (average_time_ms / 1e3)
    )
    ici_bandwidths[matrix_size_gbyte] = achieved_bandwidth_gbyte_s
    print(
        f"all_gather_ici: Matrix size: {matrix_dim}x{matrix_dim}, {dtype=}, "
        f"{matrix_size_gbyte=}, {achieved_bandwidth_gbyte_s=}"
    )

  return dcn_bandwidths, ici_bandwidths


def ppermute_benchmark(matrix_dim, dcn_size=2, ici_size=4):
  """Benchmark the ppermute collective operation."""
  dcn_bandwidths = {}
  ici_bandwidths = {}
  dtype = jax.numpy.bfloat16
  mesh, _, _ = create_mesh(dcn_size, ici_size)
  matrix = jnp.ones((matrix_dim, matrix_dim), dtype=dtype)
  matrix_size_gbyte = matrix.size * dtype.dtype.itemsize / 1e9

  # DCN benchmark
  if dcn_size > 1:

    @partial(
        shard_map, mesh=mesh, in_specs=P("dcn", None), out_specs=P("dcn", None)
    )
    def ppermute_dcn_op(x):
      perm = [(i, (i + 1) % dcn_size) for i in range(dcn_size)]
      return jax.lax.ppermute(x, "dcn", perm)

    sharded_matrix = jax.device_put(
        matrix, jax.sharding.NamedSharding(mesh, P("dcn", None))
    )
    jitted_op = jax.jit(ppermute_dcn_op)
    average_time_ms = simple_timeit(
        jitted_op, sharded_matrix, task="ppermute_dcn_op"
    )

    # each sharded matrix size is matrix_size_gbyte / dcn_size and then it needs
    # to use 1 step
    achieved_bandwidth_gbyte_s = (
        matrix_size_gbyte / dcn_size / (average_time_ms / 1e3)
    )
    dcn_bandwidths[matrix_size_gbyte] = achieved_bandwidth_gbyte_s
    print(
        f"ppermute_dcn: Matrix size: {matrix_dim}x{matrix_dim}, {dtype=}, "
        f"{matrix_size_gbyte=}, {achieved_bandwidth_gbyte_s=}"
    )

  # ICI benchmark
  if ici_size > 1:

    @partial(
        shard_map, mesh=mesh, in_specs=P(None, "ici"), out_specs=P(None, "ici")
    )
    def ppermute_ici_op(x):
      perm = [(i, (i + 1) % ici_size) for i in range(ici_size)]
      return jax.lax.ppermute(x, "ici", perm)

    sharded_matrix = jax.device_put(
        matrix, jax.sharding.NamedSharding(mesh, P(None, "ici"))
    )
    jitted_op = jax.jit(ppermute_ici_op)
    average_time_ms = simple_timeit(
        jitted_op, sharded_matrix, task="ppermute_ici_op"
    )

    # each sharded matrix size is matrix_size_gbyte / ici_size and then it needs
    # to use 1 step
    achieved_bandwidth_gbyte_s = (
        matrix_size_gbyte / ici_size / (average_time_ms / 1e3)
    )
    ici_bandwidths[matrix_size_gbyte] = achieved_bandwidth_gbyte_s
    print(
        f"ppermute_ici: Matrix size: {matrix_dim}x{matrix_dim}, {dtype=}, "
        f"{matrix_size_gbyte=}, {achieved_bandwidth_gbyte_s=}"
    )

  return dcn_bandwidths, ici_bandwidths


def all_to_all_benchmark(matrix_dim, dcn_size=2, ici_size=4):
  """Benchmark the all_to_all collective operation."""
  dcn_bandwidths = {}
  ici_bandwidths = {}
  dtype = jax.numpy.bfloat16
  mesh, _, _ = create_mesh(dcn_size, ici_size)
  matrix = jnp.ones((matrix_dim, matrix_dim), dtype=dtype)
  matrix_size_gbyte = matrix.size * dtype.dtype.itemsize / 1e9

  # DCN benchmark
  if dcn_size > 1:

    @partial(
        shard_map, mesh=mesh, in_specs=P("dcn", None), out_specs=P("dcn", None)
    )
    def all_to_all_dcn_op(x):
      return jax.lax.all_to_all(
          x, "dcn", split_axis=0, concat_axis=0, tiled=True
      )

    sharded_matrix = jax.device_put(
        matrix, jax.sharding.NamedSharding(mesh, P("dcn", None))
    )
    jitted_op = jax.jit(all_to_all_dcn_op)
    average_time_ms = simple_timeit(
        jitted_op, sharded_matrix, task="all_to_all_dcn_op"
    )

    # TODO: figure out the correct bandwidth calculation for all_to_all
    achieved_bandwidth_gbyte_s = (
        matrix_size_gbyte
        * (dcn_size - 1)
        / dcn_size
        / dcn_size
        / (average_time_ms / 1e3)
    )
    dcn_bandwidths[matrix_size_gbyte] = achieved_bandwidth_gbyte_s
    print(
        f"all_to_all_dcn: Matrix size: {matrix_dim}x{matrix_dim}, {dtype=}, "
        f"{matrix_size_gbyte=}, {achieved_bandwidth_gbyte_s=}"
    )

  # ICI benchmark
  if ici_size > 1:

    @partial(
        shard_map, mesh=mesh, in_specs=P(None, "ici"), out_specs=P(None, "ici")
    )
    def all_to_all_ici_op(x):
      return jax.lax.all_to_all(
          x, "ici", split_axis=0, concat_axis=0, tiled=True
      )

    sharded_matrix = jax.device_put(
        matrix, jax.sharding.NamedSharding(mesh, P(None, "ici"))
    )
    jitted_op = jax.jit(all_to_all_ici_op)
    average_time_ms = simple_timeit(
        jitted_op, sharded_matrix, task="all_to_all_ici_op"
    )

    # TODO: figure out the correct bandwidth calculation for all_to_all
    achieved_bandwidth_gbyte_s = (
        matrix_size_gbyte
        * (ici_size - 1)
        / ici_size
        / ici_size
        / (average_time_ms / 1e3)
    )
    ici_bandwidths[matrix_size_gbyte] = achieved_bandwidth_gbyte_s
    print(
        f"all_to_all_ici: Matrix size: {matrix_dim}x{matrix_dim}, {dtype=}, "
        f"{matrix_size_gbyte=}, {achieved_bandwidth_gbyte_s=}"
    )

  return dcn_bandwidths, ici_bandwidths


def run_benchmark(benchmark_fn, name):
  """Runs the benchmark and saves traces."""
  test_start_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
  trace_name = f"t_{name}_" + "".join(
      random.choice(string.ascii_uppercase + string.digits) for _ in range(10)
  )
  trace_dir = None
  if TRACE_BASE_DIR:
    trace_dir = f"{TRACE_BASE_DIR}/{trace_name}"
    jax.profiler.start_trace(trace_dir)

  dcn_bandwidths = {}
  ici_bandwidths = {}
  matrix_size = MATRIX_START_SIZE
  while True:
    try:
      dcn_bw, ici_bw = benchmark_fn(matrix_size)
      dcn_bandwidths.update(dcn_bw)
      ici_bandwidths.update(ici_bw)
      matrix_size += MATRIX_STEP_SIZE
      if matrix_size > MATRIX_MAX_SIZE:
        break
    except MemoryError:
      print(
          "MemoryError: Failed to create or process matrix of size"
          f" {matrix_size} x {matrix_size}.\n"
      )
      break
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(f"Exception: {e} occurred at size {matrix_size} x {matrix_size}.\n")
      break

  if TRACE_BASE_DIR:
    jax.profiler.stop_trace()
    print(f"Trace saved to {trace_dir}")

  test_end_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

  metrics = {}
  if dcn_bandwidths:
    metrics.update({
        "dcn_max_bandwidth_gbyte_s": max(dcn_bandwidths.values()),
        "dcn_median_bandwidth_gbyte_s": np.percentile(
            list(dcn_bandwidths.values()), 50
        ),
        "dcn_p90_bandwidth_gbyte_s": np.percentile(
            list(dcn_bandwidths.values()), 90
        ),
    })
  if ici_bandwidths:
    metrics.update({
        "ici_max_bandwidth_gbyte_s": max(ici_bandwidths.values()),
        "ici_median_bandwidth_gbyte_s": np.percentile(
            list(ici_bandwidths.values()), 50
        ),
        "ici_p90_bandwidth_gbyte_s": np.percentile(
            list(ici_bandwidths.values()), 90
        ),
    })

  if METRICS_JSONL_DIR:
    maybe_write_metrics_file(
        METRICS_JSONL_DIR,
        metrics,
        name,
        test_start_time,
        test_end_time,
    )

  return dcn_bandwidths, ici_bandwidths


def main():
  parser = argparse.ArgumentParser(
      description=(
          "A script to run collective operation benchmarks using shard_map and"
          " dump the results to a JSONL file."
      ),
      formatter_class=argparse.RawTextHelpFormatter,
  )

  parser.add_argument(
      "--trace_dir",
      type=str,
      help=(
          "Set the output directory, such as"
          " `--trace_dir=/tmp/microbenchmark/outputs`"
      ),
  )
  parser.add_argument(
      "--metrics_jsonl_dir",
      type=str,
      help=(
          "The directory to generate the metrics JSONL file, such as"
          " `--metrics_jsonl_dir=/tmp/microbenchmark/outputs/`"
      ),
  )
  parser.add_argument(
      "--benchmarks",
      nargs="+",
      choices=[
          "psum",
          "psum_scatter",
          "all_gather",
          "ppermute",
          "all_to_all",
          "all",
      ],
      default=["all"],
      help="Specify which benchmarks to run",
  )
  parser.add_argument(
      "--dcn_size", type=int, default=2, help="Number of DCN slices"
  )
  parser.add_argument(
      "--ici_size",
      type=int,
      default=4,
      help="Number of devices per slice (ICI)",
  )

  args = parser.parse_args()

  global TRACE_BASE_DIR, METRICS_JSONL_DIR
  if args.trace_dir:
    TRACE_BASE_DIR = args.trace_dir
  if args.metrics_jsonl_dir:
    METRICS_JSONL_DIR = args.metrics_jsonl_dir

  benchmark_map = {
      "psum": lambda x: psum_benchmark(x, args.dcn_size, args.ici_size),
      "psum_scatter": lambda x: psum_scatter_benchmark(
          x, args.dcn_size, args.ici_size
      ),
      "all_gather": lambda x: all_gather_benchmark(
          x, args.dcn_size, args.ici_size
      ),
      "ppermute": lambda x: ppermute_benchmark(x, args.dcn_size, args.ici_size),
      "all_to_all": lambda x: all_to_all_benchmark(
          x, args.dcn_size, args.ici_size
      ),
  }

  benchmarks_to_run = (
      benchmark_map
      if "all" in args.benchmarks
      else {b: benchmark_map[b] for b in args.benchmarks}
  )

  for name, benchmark in benchmarks_to_run.items():
    print(f"Running benchmark {name}")
    _, _ = run_benchmark(benchmark, name)


if __name__ == "__main__":
  main()