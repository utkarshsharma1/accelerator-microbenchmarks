export CLUSTER_NAME="mlperf-v5p"
export REGION="europe-west4"
export ZONE="europe-west4-b"
export PROJECT_ID="cloud-tpu-multipod-dev"
# TODO change it to the TPU type you want to run the workload on.
# For example, v5p-128 or v5p-256.
export TPU_TYPE="v5p-256"
export NUM_SLICES=1

# TODO change it
# This is the name of the workload that will be created in the xpk command.
export WORKLOAD_NAME="utksharma-tpu-mb-v5p-256"

gcloud config set project ${PROJECT_ID}
gcloud config set compute/zone ${ZONE}
gcloud container clusters get-credentials ${CLUSTER_NAME} \
  --region ${REGION} --project ${PROJECT_ID}

# TODO change the gsutil path for every run, this is the path where the results will be stored.
# bucket remains the same, but the folder name changes.
xpk workload create \
  --cluster=${CLUSTER_NAME} \
  --device-type=${TPU_TYPE} \
  --command="git clone https://github.com/utkarshsharma1/accelerator-microbenchmarks.git \
  && cd accelerator-microbenchmarks && pip install -r requirements.txt && \
  python src/run_benchmark.py --config=configs/xlml_v5p_256_utksharma.yaml \
  && gsutil -m cp -r /tmp/microbenchmarks gs://v5p-microbenchmarks/v5p-256-$(date +%Y-%m-%d-%H-%M-%S)/" \
  --num-slices=${NUM_SLICES} \
  --docker-image=us-docker.pkg.dev/cloud-tpu-images/jax-stable-stack/tpu:jax0.5.2-rev1 \
  --workload=${WORKLOAD_NAME}

# Wait for the workload to finish, then delete at the end.
xpk workload delete --cluster=${CLUSTER_NAME} --workload=${WORKLOAD_NAME}
