CLUSTER_NAME="mlperf-v5p"
REGION="europe-west4"
ZONE="europe-west4-b"
PROJECT_ID="cloud-tpu-multipod-dev"
DOCKER_IMAGE="us-docker.pkg.dev/cloud-tpu-images/jax-stable-stack/tpu:jax0.5.2-rev1"
RUN_ID=$(date +%Y-%m-%d-%H-%M-%S)
TPU_TYPE=v5p-128
GCS_BASE_PATH="gs://v5p-microbenchmarks/report_data_${RUN_ID}"

# gcloud setup
gcloud config set project ${PROJECT_ID}
gcloud config set compute/zone ${ZONE}
gcloud container clusters get-credentials ${CLUSTER_NAME} --region ${REGION} --project ${PROJECT_ID}

# Benchmark variables
CONFIG_FILE="configs/xlml_v5p_128_utksharma.yaml"
WORKLOAD_NAME="prisha-mb-${TPU_TYPE}"
GCS_JSONL_PATH="${GCS_BASE_PATH}/${TPU_TYPE}/metrics_report.jsonl"
GCS_EXCEL_PATH="${GCS_BASE_PATH}/${TPU_TYPE}_benchmark_report.xlsx"
XPK_COMMAND="git clone https://github.com/utkarshsharma1/accelerator-microbenchmarks.git && \
cd accelerator-microbenchmarks && \
pip install -r requirements.txt && \
python src/run_benchmark.py \
  --config=${CONFIG_FILE} \
  --generate_report \
  --gcs_jsonl_path=\"${GCS_JSONL_PATH}\" \
  --tpu_type=\"${TPU_TYPE}\" \
  --gcs_excel_path=\"${GCS_EXCEL_PATH}\""
xpk workload create --cluster=${CLUSTER_NAME} --device-type=${TPU_TYPE} --command="${XPK_COMMAND}" --num-slices=1 --docker-image=${DOCKER_IMAGE} --workload=${WORKLOAD_NAME}

# Delete the workload after it finishes
xpk workload delete --cluster=${CLUSTER_NAME} --workload=${WORKLOAD_NAME}

