import yaml
import os
import glob
import re
import shutil # For potential safer renaming with move

def rename_hlo_dumps(config_path, hlo_dump_dir):
    """
    Renames various HLO dump files with their corresponding configuration strings,
    handling arbitrary parameters from benchmark_sweep_params.

    Args:
        config_path (str): Path to the YAML configuration file (e.g., simple_matmul.yaml).
        hlo_dump_dir (str): Directory where HLO dumps are stored (e.g., /tmp/hlo_graphs/).
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    benchmark_params = []
    for benchmark in config.get('benchmarks', []):
        for params in benchmark.get('benchmark_sweep_params', []):
            benchmark_params.append(params)

    if not benchmark_params:
        print("No benchmark sweep parameters found in the config file. Exiting.")
        return

    print(f"Found {len(benchmark_params)} benchmark configurations in '{config_path}'.")

    # Define the patterns for the files you want to rename
    file_patterns = [
        '*jit_f*before_optimizations*.txt',
        '*jit_f*.tpu_comp_env.txt',
        '*jit_f*.execution_options.txt'
    ]

    # Process each type of file
    for pattern in file_patterns:
        full_pattern = os.path.join(hlo_dump_dir, pattern)
        hlo_files = sorted(
            glob.glob(full_pattern),
            key=os.path.getmtime
        )

        if not hlo_files:
            print(f"\nNo files found matching pattern: '{full_pattern}'. Skipping this type.")
            continue

        print(f"\nProcessing {len(hlo_files)} files matching pattern: '{full_pattern}'")

        if len(hlo_files) != len(benchmark_params):
            print(f"  Warning: Number of files ({len(hlo_files)}) does not match number of configurations ({len(benchmark_params)}).")
            print("  This script assumes a one-to-one, ordered correspondence. Please verify the output carefully.")

        for i, hlo_file_path in enumerate(hlo_files):
            if i < len(benchmark_params):
                params = benchmark_params[i]

                # Dynamically create the config string from all parameters
                # Example: {'a': 10, 'b': 20} -> "a10_b20"
                config_parts = []
                for key, value in params.items():
                    config_parts.append(f"{key}{value}")
                config_string = "_".join(config_parts)

                if not config_string: # Handle cases where a param set might be empty
                    print(f"  Warning: Empty parameter set at index {i}. Skipping file '{os.path.basename(hlo_file_path)}'.")
                    continue

                directory, base_filename = os.path.split(hlo_file_path)

                # Prepend the config string to the original filename
                new_filename = f"{config_string}_{base_filename}"
                new_file_path = os.path.join(directory, new_filename)

                try:
                    shutil.move(hlo_file_path, new_file_path)
                    print(f"  Renamed: '{base_filename}' -> '{new_filename}'")
                except Exception as e:
                    print(f"  Error renaming '{hlo_file_path}': {e}")
            else:
                print(f"  Warning: No matching configuration for HLO file: '{os.path.basename(hlo_file_path)}'. Skipping.")



def rename_xla_dump(xla_dump_dir, benchmark_name, benchmark_param):
    """
    Finds the latest XLA dump file matching '*jit_f*before_optimizations*.txt',
    then identifies all other files that share the same 'jit_f.[unique_id]' identifier
    and renames them to 'benchmark_name_serialized_params.original_suffix_with_extension'.
    """

    serialized_benchmark_param = str(benchmark_param)
    anchor_pattern = os.path.join(xla_dump_dir, '*jit_f*before_optimizations*.txt')
    matching_anchor_files = glob.glob(anchor_pattern)

    if not matching_anchor_files:
        print(f"No files found for anchor pattern: '{anchor_pattern}'. No files will be renamed.")
        return

    # Sort anchor files by modification time (latest first)
    matching_anchor_files.sort(key=os.path.getmtime, reverse=True)
    latest_anchor_file = matching_anchor_files[0]
    print(f"Latest anchor file found: '{latest_anchor_file}' (Modified: {datetime.fromtimestamp(os.path.getmtime(latest_anchor_file))})")

    # Extract the common 'jit_f.[unique_id]' part from the anchor file.
    # This regex captures from 'jit_f.' up to the next '.' (before the specific suffix like '.before_optimizations')
    # Example: 'module_0080.jit_f.cl_747713181.before_optimizations.txt'
    # This will extract 'jit_f.cl_747713181'
    filename_base = os.path.basename(latest_anchor_file)
    jit_id_match = re.search(r'(jit_f\.[^.]+)', filename_base)

    if not jit_id_match:
        print(f"Could not extract 'jit_f.[unique_id]' from '{filename_base}'. Cannot proceed with renaming.")
        return

    common_jit_id_prefix = jit_id_match.group(1) # e.g., 'jit_f.cl_747713181'
    print(f"Extracted common JIT ID prefix for family: '{common_jit_id_prefix}'")

    # Find all files in the directory that contain this specific common_jit_id_prefix
    # We are looking for files like 'module_XXX.jit_f.ID.suffix.txt'
    all_related_files_pattern = os.path.join(xla_dump_dir, f'*{common_jit_id_prefix}*')
    all_related_files = glob.glob(all_related_files_pattern)

    if not all_related_files:
        print(f"No files found containing '{common_jit_id_prefix}'. This is unexpected if an anchor was found.")
        return

    new_base_name = f"{benchmark_name}_{serialized_benchmark_param}"

    print(f"\n--- Renaming files belonging to the '{common_jit_id_prefix}' family ---")
    for original_filepath in all_related_files:
        original_filename = os.path.basename(original_filepath)
        
        # Find the specific suffix part *after* the common_jit_id_prefix.
        # This regex looks for the common_jit_id_prefix, then captures everything after it,
        # ensuring it starts with a dot if there's more.
        # Example: if original_filename is 'module_0080.jit_f.cl_747713181.after_codegen.txt'
        # and common_jit_id_prefix is 'jit_f.cl_747713181'
        # we want to capture '.after_codegen.txt'
        suffix_match = re.search(re.escape(common_jit_id_prefix) + r'(\..*)', original_filename)
        
        if suffix_match:
            original_suffix_with_extension = suffix_match.group(1) # e.g., '.after_codegen.txt'
        else:
            print("shouldn't get here")

        new_filename = f"{new_base_name}{original_suffix_with_extension}"
        new_filepath = os.path.join(xla_dump_dir, new_filename)

        if original_filepath == new_filepath:
            print(f"Skipping: '{original_filename}' already has the desired name or path.")
            continue

        try:
            os.rename(original_filepath, new_filepath)
            print(f"Renamed '{original_filename}' to '{new_filename}'")
        except OSError as e:
            print(f"Error renaming file '{original_filepath}' to '{new_filepath}': {e}")
        except Exception as e:
            print(f"An unexpected error occurred while renaming '{original_filepath}': {e}")
