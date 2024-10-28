# RankML
Microbenchmarks that assess the performance of individual operations and components on accelerators with JAX.

## Setup

Setup the cloud TPU enviroment. 

## Running the code

After setting up the environment, you can run the benchmark script using the following command:

```bash
python3 benchmark_collective.py --ici_size=<ici_size> --dcn_size=<dcn_size>
```