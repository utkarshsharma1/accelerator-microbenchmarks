export CLUSTER_NAME=#<cluster_name>
export REGION=#<region>
export ZONE=#<zone>
export PROJECT_ID=#<project_id>
export DOCKER_IMAGE="us-docker.pkg.dev/cloud-tpu-images/jax-stable-stack/tpu:jax0.5.2-rev1"
export RUN_ID=$(date +%Y-%m-%d-%H-%M-%S)

# Change as per need (v5p-128/v5p-256)
export TPU_TYPE=v5p-128
export GCS_BASE_PATH=#<gcs_base_path>
export GCS_PATH="${GCS_BASE_PATH}/report_data_${RUN_ID}"

# gcloud setup
gcloud config set project ${PROJECT_ID}
gcloud config set compute/zone ${ZONE}
gcloud container clusters get-credentials ${CLUSTER_NAME} --region ${REGION} --project ${PROJECT_ID}

# Benchmark variables, starts with configs/<file_name>
CONFIG_FILE=#<config_file>

# TODO: change workload name
WORKLOAD_NAME="prisha-mb-${TPU_TYPE}"

GCS_JSONL_PATH="${GCS_PATH}/${TPU_TYPE}/metrics_report.jsonl"
GCS_EXCEL_PATH="${GCS_PATH}/${TPU_TYPE}_benchmark_report.xlsx"
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
