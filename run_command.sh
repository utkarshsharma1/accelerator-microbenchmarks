export CLUSTER_NAME="mlperf-v5p"
export REGION="europe-west4"
export ZONE="europe-west4-b"
export PROJECT_ID="cloud-tpu-multipod-dev"
gcloud config set project ${PROJECT_ID}
gcloud config set compute/zone ${ZONE}
gcloud container clusters get-credentials ${CLUSTER_NAME} --region ${REGION} --project ${PROJECT_ID}

export GITHUB_USER="prishajain1"
export BRANCH_NAME="combined_report"
export DOCKER_IMAGE="us-docker.pkg.dev/cloud-tpu-images/jax-stable-stack/tpu:jax0.5.2-rev1"
export GCS_BASE_PATH="gs://v5p-microbenchmarks/report_data"

# Run v5p-128 Benchmark
TPU_TYPE_128="v5p-128"
CONFIG_128="configs/xlml_v5p_128_utksharma.yaml"
WORKLOAD_128="prisha-mb-128"
GCS_PATH_128="${GCS_BASE_PATH}/${TPU_TYPE_128}/metrics_report.jsonl"
XPK_COMMAND_128="set -ex && rm -rf accelerator-microbenchmarks && git clone -b ${BRANCH_NAME} https://github.com/${GITHUB_USER}/accelerator-microbenchmarks.git && cd accelerator-microbenchmarks && pip install --no-cache-dir -r requirements.txt && python src/run_benchmark.py --config=${CONFIG_128} && gsutil -m cp /tmp/microbenchmarks/outputs/metrics_report.jsonl ${GCS_PATH_128}"
xpk workload create --cluster=${CLUSTER_NAME} --device-type=${TPU_TYPE_128} --command="${XPK_COMMAND_128}" --num-slices=1 --docker-image=${DOCKER_IMAGE} --workload=${WORKLOAD_128}
# WAIT FOR THIS WORKLOAD TO COMPLETE SUCCESSFULLY, then delete workload.
xpk workload delete --cluster=${CLUSTER_NAME} --workload=${WORKLOAD_128}

# Run v5p-256 Benchmark
TPU_TYPE_256="v5p-256"
CONFIG_256="configs/xlml_v5p_256_utksharma.yaml"
WORKLOAD_256="prisha-mb-256"
GCS_PATH_256="${GCS_BASE_PATH}/${TPU_TYPE_256}/metrics_report.jsonl"
XPK_COMMAND_256="set -ex && rm -rf accelerator-microbenchmarks && git clone -b ${BRANCH_NAME} https://github.com/${GITHUB_USER}/accelerator-microbenchmarks.git && cd accelerator-microbenchmarks && pip install --no-cache-dir -r requirements.txt && python src/run_benchmark.py --config=${CONFIG_256} && gsutil -m cp /tmp/microbenchmarks/outputs/metrics_report.jsonl ${GCS_PATH_256}"
xpk workload create --cluster=${CLUSTER_NAME} --device-type=${TPU_TYPE_256} --command="${XPK_COMMAND_256}" --num-slices=1 --docker-image=${DOCKER_IMAGE} --workload=${WORKLOAD_256}
# WAIT FOR THIS WORKLOAD TO COMPLETE SUCCESSFULLY, then delete workload.
xpk workload delete --cluster=${CLUSTER_NAME} --workload=${WORKLOAD_256}


# Generate report
JSONL_128="${GCS_BASE_PATH}/v5p-128/metrics_report.jsonl"
JSONL_256="${GCS_BASE_PATH}/v5p-256/metrics_report.jsonl"
LOCAL_EXCEL="/tmp/combined_benchmark_report.xlsx"
GCS_EXCEL_PATH="${GCS_BASE_PATH}/combined_report_$(date +%Y-%m-%d-%H-%M-%S).xlsx"

python -m pip install --upgrade google-cloud-storage openpyxl
python src/generate_combined_report.py --gcs_path_128 "${JSONL_128}" --gcs_path_256 "${JSONL_256}" --local_output "${LOCAL_EXCEL}"
if [[ -f "${LOCAL_EXCEL}" ]]; then gsutil cp "${LOCAL_EXCEL}" "${GCS_EXCEL_PATH}" && echo "Report uploaded to ${GCS_EXCEL_PATH}"; else echo "Report generation failed"; fi

