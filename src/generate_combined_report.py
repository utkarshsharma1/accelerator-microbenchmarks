import json
import os
import argparse
from collections import defaultdict
from openpyxl import Workbook
from google.cloud import storage
import logging
import io
import re

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def parse_tpu_type(tpu_type):
    """
    Parses the TPU type string to extract chip count and determine
    expected ici_size and core count for reporting.

    Args:
        tpu_type (str): The TPU type string, e.g., "v5p-128", "v5p-256".

    Returns:
        dict: Containing "chips", "expected_ici_size", and "cores".

    Raises:
        ValueError: If the chip count cannot be extracted.
    """
    match = re.search(r'(\d+)$', tpu_type)
    if not match:
        raise ValueError(f"Cannot extract chip count from tpu_type: {tpu_type}")

    chips = int(match.group(1))

    # Based on the original script's logic for v5p:
    # - The number of cores reported is equal to the number of chips.
    # - The 'ici_size' field in the JSON data seems to be half the number of chips.
    cores_to_report = chips
    expected_ici_size = chips // 2

    logging.info(f"Parsed tpu_type '{tpu_type}': chips={chips}, expected_ici_size={expected_ici_size}, cores_to_report={cores_to_report}")
    return {
        "chips": chips,
        "expected_ici_size": expected_ici_size,
        "cores": cores_to_report
    }

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

def generate_excel_report(jsonl_gcs_path, tpu_type, excel_gcs_path):
    logging.info(f"Generating report for TPU type {tpu_type} from {jsonl_gcs_path}")

    try:
        config = parse_tpu_type(tpu_type)
    except ValueError as e:
        logging.error(f"Error parsing TPU type: {e}")
        return

    expected_ici_size = config["expected_ici_size"]
    core_count = config["cores"]

    data_by_test = defaultdict(lambda: defaultdict(dict))

    def process_jsonl(jsonl_content):
        if not jsonl_content: return

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

                    if test_name and matrix_dim is not None and ici_size is not None:
                        try:
                            if int(ici_size) == expected_ici_size:
                                dim = int(matrix_dim)
                                data_by_test[test_name][dim][core_count] = metrics
                        except ValueError:
                            logging.warning(f"Skipping record, non-int matrix_dim or ici_size: {matrix_dim}, {ici_size}")
            except json.JSONDecodeError:
                logging.warning(f"Skipping line, JSON decode error: {line}")
            except Exception as e:
                logging.error(f"Error processing line: {e} - Line: {line}")

    try:
        jsonl_content = download_gcs_blob_as_text(jsonl_gcs_path)
        process_jsonl(jsonl_content)
    except Exception as e:
        logging.error(f"Could not process {tpu_type} file: {e}")
        return

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
            ws.cell(row=2, column=current_col + 1, value=core_count)

            for row_idx, dim in enumerate(matrix_dims):
                ws.cell(row=3 + row_idx, column=current_col, value=dim)
                core_metrics = matrix_data[dim].get(core_count)
                metric_val = core_metrics.get(metric) if core_metrics else ""
                ws.cell(row=3 + row_idx, column=current_col + 1, value=metric_val)

            current_col += 3  # 1 dim col + 1 data col + 1 spacer col

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
        logging.info(f"Excel report uploaded to {excel_gcs_path}")

    except Exception as e:
        logging.error(f"Failed to save or upload Excel file: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate benchmark report for a single TPU type and upload to GCS.")
    parser.add_argument("--gcs_path", required=True, help="GCS path to metrics_report.jsonl file")
    parser.add_argument("--tpu_type", required=True, help="TPU type (e.g., v5p-128, v5p-256)")
    parser.add_argument("--gcs_output_path", required=True, help="GCS path to save the generated Excel file")
    args = parser.parse_args()
    generate_excel_report(args.gcs_path, args.tpu_type, args.gcs_output_path)
