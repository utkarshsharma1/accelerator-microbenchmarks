import json
import os
import argparse
from collections import defaultdict
from openpyxl import Workbook
from google.cloud import storage
import logging
import io

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Map ici_size to number of cores
CORE_MAP = {
    64: 128,  # ici_size 64 -> 128 cores
    128: 256   # ici_size 128 -> 256 cores
}
# The fixed column headers for core counts in the report
CORE_COLUMNS = sorted(CORE_MAP.values()) # Results in [128, 256]

def download_gcs_blob_as_text(gcs_path):
    try:
        if not gcs_path.startswith("gs://"):
            raise ValueError("GCS path must start with gs://")
        parts = gcs_path[5:].split("/", 1)
        bucket_name = parts[0]
        blob_name = parts[1]
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        logging.info(f"Downloading {gcs_path}")
        return blob.download_as_text()
    except Exception as e:
        logging.error(f"Failed to download {gcs_path}: {e}")
        raise

def upload_to_gcs(bucket_name, blob_name, data):
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_file(data, content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        logging.info(f"Uploaded to gs://{bucket_name}/{blob_name}")
    except Exception as e:
        logging.error(f"Failed to upload to gs://{bucket_name}/{blob_name}: {e}")
        raise

def generate_combined_excel_report(jsonl_gcs_path_128, jsonl_gcs_path_256, excel_gcs_path):
    logging.info(f"Generating combined report from {jsonl_gcs_path_128} and {jsonl_gcs_path_256}")

    data_by_test = defaultdict(lambda: defaultdict(dict))

    def process_jsonl(jsonl_content, expected_ici_size):
        if not jsonl_content: return
        core_count = CORE_MAP.get(expected_ici_size)
        if core_count is None: return

        for line in jsonl_content.strip().split('\n'):
            if not line: continue
            try:
                record = json.loads(line)
                if 'metrics' in record and 'dimensions' in record:
                    metrics = record['metrics']
                    dims = record['dimensions']
                    test_name = dims.get('test_name')
                    matrix_dim = dims.get('matrix_dim')
                    ici_size = dims.get('ici_size')

                    if test_name and matrix_dim is not None and ici_size is not None and int(ici_size) == expected_ici_size:
                        try:
                            dim = int(matrix_dim)
                            data_by_test[test_name][dim][core_count] = metrics
                        except ValueError:
                            logging.warning(f"Skipping record, non-int matrix_dim: {matrix_dim}")
            except json.JSONDecodeError:
                logging.warning(f"Skipping line, JSON decode error: {line}")
            except Exception as e:
                logging.error(f"Error processing line: {e} - Line: {line}")

    try:
        content_128 = download_gcs_blob_as_text(jsonl_gcs_path_128)
        process_jsonl(content_128, 64)
    except Exception as e:
        logging.error(f"Could not process 128-core file: {e}")

    try:
        content_256 = download_gcs_blob_as_text(jsonl_gcs_path_256)
        process_jsonl(content_256, 128)
    except Exception as e:
        logging.error(f"Could not process 256-core file: {e}")

    if not data_by_test:
        logging.info("No valid data found to generate report.")
        return

    wb = Workbook()
    if "Sheet" in wb.sheetnames: wb.remove(wb["Sheet"])

    metrics_keys = ["ici_bandwidth_gbyte_s_p50", "ici_bandwidth_gbyte_s_p90", "ici_bandwidth_gbyte_s_p95", "ici_bandwidth_gbyte_s_p99", "ici_bandwidth_gbyte_s_avg"]

    for test_name in sorted(data_by_test.keys()):
        matrix_data = data_by_test[test_name]
        safe_test_name = "".join(c for c in test_name if c.isalnum() or c in (' ', '_')).rstrip()[:31]
        ws = wb.create_sheet(title=safe_test_name)
        matrix_dims = sorted(matrix_data.keys())

        current_col = 1
        for metric in metrics_keys:
            ws.cell(row=1, column=current_col, value=metric)
            ws.cell(row=2, column=current_col, value="dimensions\\TPUs")
            ws.cell(row=2, column=current_col + 1, value=CORE_COLUMNS[0])  # 128
            ws.cell(row=2, column=current_col + 2, value=CORE_COLUMNS[1])  # 256

            for row_idx, dim in enumerate(matrix_dims):
                ws.cell(row=3 + row_idx, column=current_col, value=dim)
                for i, core_count in enumerate(CORE_COLUMNS):
                    core_metrics = matrix_data[dim].get(core_count)
                    metric_val = core_metrics.get(metric) if core_metrics else ""
                    ws.cell(row=3 + row_idx, column=current_col + 1 + i, value=metric_val)

            current_col += 5  # 1 dim col + 2 data cols + 2 spacer cols

    try:
        excel_buffer = io.BytesIO()
        wb.save(excel_buffer)
        excel_buffer.seek(0)

        if not excel_gcs_path.startswith("gs://"):
            raise ValueError("GCS output path must start with gs://")
        parts = excel_gcs_path[5:].split("/", 1)
        bucket_name = parts[0]
        blob_name = parts[1]
        upload_to_gcs(bucket_name, blob_name, excel_buffer)
        logging.info(f"Combined Excel report uploaded to {excel_gcs_path}")

    except Exception as e:
        logging.error(f"Failed to save or upload Excel file: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate combined benchmark report and upload to GCS.")
    parser.add_argument("--gcs_path_128", required=True, help="GCS path to metrics_report.jsonl for 128 cores")
    parser.add_argument("--gcs_path_256", required=True, help="GCS path to metrics_report.jsonl for 256 cores")
    parser.add_argument("--gcs_output_path", required=True, help="GCS path to save the generated Excel file")
    args = parser.parse_args()
    generate_combined_excel_report(args.gcs_path_128, args.gcs_path_256, args.gcs_output_path)
