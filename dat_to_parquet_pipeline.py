!pip install pandas
!pip install pyarrow

import pandas as pd
import json
import os
import re
import numpy as np
from datetime import datetime
import pyarrow as pa
import pyarrow.parquet as pq
from collections import defaultdict, Counter
import mmap
import zipfile
import glob

def parse_sas_input_statement(input_text: str) -> dict:
    """Parse SAS INPUT statement - simplified version"""
    columns = []

    # Pattern for ranges: VARNAME $ start-end or VARNAME start-end
    range_pattern = r'(\w+)\s+(\$\s+)?(\d+)-(\d+)'
    range_matches = re.findall(range_pattern, input_text)

    for var_name, dollar_sign, start_str, end_str in range_matches:
        start_pos = int(start_str) - 1  # Convert to 0-based indexing
        end_pos = int(end_str)  # SAS end is inclusive, Python slice is exclusive

        columns.append({
            'name': var_name,
            'start': start_pos,
            'end': end_pos,
            'width': end_pos - start_pos,
            'type': 'string' if dollar_sign else 'numeric'
        })

    # Pattern for single positions
    single_pattern = r'(\w+)\s+(\$\s+)?(\d+)(?!\d|-)'
    single_matches = re.findall(single_pattern, input_text)

    existing_vars = {col['name'] for col in columns}
    date_pattern = r'(\w+)\s+DATE\d+'
    date_vars = {match[0] for match in re.findall(date_pattern, input_text)}

    for var_name, dollar_sign, pos_str in single_matches:
        if var_name not in existing_vars and var_name not in date_vars:
            start_pos = int(pos_str) - 1
            end_pos = start_pos + 1

            columns.append({
                'name': var_name,
                'start': start_pos,
                'end': end_pos,
                'width': 1,
                'type': 'string' if dollar_sign else 'numeric'
            })

    columns.sort(key=lambda x: x['start'])
    return {'columns': columns}

def parse_sas_file(file_content: str) -> dict:
    """Extract INPUT statement from SAS file"""
    input_match = re.search(r'INPUT\s+(.*?);', file_content, re.DOTALL | re.IGNORECASE)
    if not input_match:
        raise ValueError("No INPUT statement found in SAS file")
    input_text = input_match.group(1)
    return parse_sas_input_statement(input_text)

def detect_file_encoding(file_path, sample_size=50000):
    """
    Optimized encoding detection using memory mapping
    """
    encodings_to_try = ['latin-1', 'cp1252', 'iso-8859-1', 'utf-8', 'ascii']  # Put latin-1 first for data files

    print(f"🔍 Detecting encoding for: {os.path.basename(file_path)}")

    try:
        # Use memory mapping for faster file access
        with open(file_path, 'rb') as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mmapped_file:
                sample = mmapped_file[:sample_size]

                for encoding in encodings_to_try:
                    try:
                        decoded = sample.decode(encoding)
                        # Additional validation - check if it looks like reasonable text
                        if encoding == 'utf-8':
                            # For UTF-8, be more strict - check for reasonable characters
                            printable_ratio = sum(1 for c in decoded[:1000] if c.isprintable() or c.isspace()) / min(1000, len(decoded))
                            if printable_ratio < 0.8:  # If less than 80% printable, skip UTF-8
                                continue
                        print(f"✓ Detected encoding: {encoding}")
                        return encoding
                    except UnicodeDecodeError:
                        continue
    except Exception as e:
        print(f"⚠️  Error in encoding detection: {e}")

    print("⚠️  No perfect encoding found, using latin-1 with error replacement")
    return 'latin-1'

def get_file_line_count(file_path):
    """
    Fast line counting using memory mapping
    """
    try:
        with open(file_path, 'rb') as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mmapped_file:
                lines = mmapped_file.count(b'\n')
                return lines
    except Exception:
        # Fallback to slower method
        with open(file_path, 'r', encoding='latin-1') as f:
            return sum(1 for _ in f)

def optimized_parse_fixed_width(file_path, colspecs, names, encoding='utf-8', chunk_size=50000):
    """
    Optimized fixed-width parsing using vectorized operations
    """
    print(f"📦 Parsing fixed-width file with optimized method (encoding: {encoding})")

    # Pre-calculate column positions for slicing
    col_slices = [(start, end) for start, end in colspecs]

    all_data = []
    chunk_count = 0
    total_rows = 0

    try:
        # Use larger buffer for better I/O performance with error handling
        with open(file_path, 'r', encoding=encoding, buffering=8192*8, errors='replace') as f:
            lines = []

            for line_num, line in enumerate(f, 1):
                lines.append(line.rstrip('\n\r'))

                if len(lines) >= chunk_size:
                    chunk_count += 1
                    chunk_rows = len(lines)
                    total_rows += chunk_rows

                    print(f"Processing chunk {chunk_count}: {chunk_rows:,} rows (Total: {total_rows:,})")

                    # Vectorized parsing of chunk
                    chunk_data = parse_lines_vectorized(lines, col_slices, names)
                    all_data.append(chunk_data)

                    lines = []  # Reset for next chunk

            # Process remaining lines
            if lines:
                chunk_count += 1
                chunk_rows = len(lines)
                total_rows += chunk_rows

                print(f"Processing final chunk {chunk_count}: {chunk_rows:,} rows (Total: {total_rows:,})")
                chunk_data = parse_lines_vectorized(lines, col_slices, names)
                all_data.append(chunk_data)

        # Combine all chunks efficiently
        print("🔗 Combining chunks...")
        if len(all_data) == 1:
            final_df = all_data[0]
        else:
            final_df = pd.concat(all_data, ignore_index=True, copy=False)

        print(f"✓ Parsed {total_rows:,} rows successfully")
        return final_df

    except UnicodeDecodeError as e:
        print(f"❌ Encoding error with {encoding}: {e}")
        print("🔄 Retrying with latin-1 encoding...")

        # Retry with latin-1 if the detected encoding fails
        if encoding != 'latin-1':
            return optimized_parse_fixed_width(file_path, colspecs, names, 'latin-1', chunk_size)
        else:
            raise

    except Exception as e:
        print(f"❌ Error in optimized parsing: {e}")
        raise

def parse_lines_vectorized(lines, col_slices, names):
    """
    Vectorized parsing of lines using numpy operations
    """
    # Convert lines to numpy array for vectorized operations
    data = {}

    for i, (start, end) in enumerate(col_slices):
        col_name = names[i]

        # Extract column data using list comprehension (faster than pandas apply)
        col_values = [line[start:end].strip() if len(line) > start else '' for line in lines]

        # Convert empty strings to None for better handling
        col_values = [val if val else None for val in col_values]

        data[col_name] = col_values

    return pd.DataFrame(data)

def optimized_quality_analysis(df, column_metadata, sample_size=10000):
    """
    Optimized quality analysis using vectorized operations and sampling
    """
    print(f"\n=== Running Optimized Quality Analysis ===")

    total_rows = len(df)
    quality_report = []

    # Use sampling for large datasets to speed up analysis
    if total_rows > sample_size:
        print(f"Using stratified sample of {sample_size:,} rows for quality analysis")
        sample_df = df.sample(n=sample_size, random_state=42)
        sample_multiplier = total_rows / sample_size
    else:
        sample_df = df
        sample_multiplier = 1

    for col_name in df.columns:
        col_data = sample_df[col_name]
        col_info = column_metadata.get(col_name, {})

        # Vectorized null/blank analysis
        is_null = col_data.isna()
        is_blank = (col_data.astype(str).str.strip() == '') | is_null

        null_count = int(is_null.sum() * sample_multiplier)
        blank_count = int(is_blank.sum() * sample_multiplier)
        non_blank_count = total_rows - blank_count

        # Get non-blank values for further analysis
        non_blank_values = col_data[~is_blank]

        # Numeric analysis
        numeric_count = 0
        if len(non_blank_values) > 0:
            try:
                pd.to_numeric(non_blank_values, errors='raise')
                numeric_count = int(len(non_blank_values) * sample_multiplier)
            except:
                pass

        # Unique values analysis (on sample)
        unique_values = set(non_blank_values.astype(str).head(1000))
        unique_count = len(unique_values)
        unique_ratio = (unique_count / len(non_blank_values)) if len(non_blank_values) > 0 else 0

        # Length analysis
        if len(non_blank_values) > 0:
            lengths = non_blank_values.astype(str).str.len()
            min_length = int(lengths.min())
            max_length = int(lengths.max())
            avg_length = float(lengths.mean())
        else:
            min_length = max_length = avg_length = 0

        # Sample values
        sample_values = [str(val)[:20] for val in list(unique_values)[:5]]

        # Pattern analysis (simplified and faster)
        patterns = analyze_patterns_fast(non_blank_values.head(500))
        pattern_summary = [f"{pat}({cnt})" for pat, cnt in list(patterns.items())[:3]]

        # Quality flags
        flags = []
        pct_non_blank = (non_blank_count / total_rows) * 100 if total_rows > 0 else 0

        if pct_non_blank < 50:
            flags.append("HIGH_BLANK_RATE")
        if unique_count == 1 and len(non_blank_values) > 10:
            flags.append("ALL_SAME_VALUE")
        if col_info.get('type') == 'numeric' and numeric_count == 0 and len(non_blank_values) > 0:
            flags.append("MIXED_DATA_TYPES")
        if unique_ratio < 0.01 and len(non_blank_values) > 100:
            flags.append("LOW_VARIETY")
        if unique_ratio > 0.95 and len(non_blank_values) > 100:
            flags.append("HIGH_VARIETY")

        # Inferred type
        if numeric_count > 0 and numeric_count == len(non_blank_values):
            inferred_type = 'numeric'
        elif len(non_blank_values) > 0:
            inferred_type = 'text'
        else:
            inferred_type = 'unknown'

        quality_report.append({
            'column_name': col_name,
            'position_start': col_info.get('start', ''),
            'position_end': col_info.get('end', ''),
            'declared_width': col_info.get('width', ''),
            'declared_type': col_info.get('type', ''),
            'total_rows': total_rows,
            'non_blank_count': non_blank_count,
            'blank_count': blank_count,
            'null_count': null_count,
            'pct_non_blank': round(pct_non_blank, 2),
            'pct_null': round((null_count / total_rows) * 100, 2),
            'inferred_type': inferred_type,
            'unique_values': str(unique_count),
            'unique_ratio': round(unique_ratio, 4),
            'min_length': min_length,
            'max_length': max_length,
            'avg_length': round(avg_length, 2),
            'sample_values': '; '.join(sample_values),
            'common_patterns': '; '.join(pattern_summary),
            'quality_flags': '; '.join(flags) if flags else 'OK',
            'analysis_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })

    return pd.DataFrame(quality_report)

def analyze_patterns_fast(series):
    """
    Fast pattern analysis using Counter
    """
    if len(series) == 0:
        return {}

    patterns = Counter()

    for value in series.head(500):
        # Simple pattern generation
        pattern = ''.join('D' if c.isdigit() else 'A' if c.isalpha() else 'S' if c.isspace() else 'X'
                         for c in str(value))

        # Compress repeated characters
        compressed = re.sub(r'(.)\1+', r'\1+', pattern)
        patterns[compressed] += 1

    return dict(patterns.most_common(10))

def save_to_parquet_optimized(df, output_path):
    """
    Optimized Parquet saving with compression and type optimization
    """
    print(f"💾 Saving to Parquet with optimization...")

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Convert all columns to string type for consistency
    for col in df.columns:
        df[col] = df[col].astype('string')

    # Save with optimal settings
    df.to_parquet(output_path,
                  compression='snappy',
                  index=False,
                  engine='pyarrow')

    print(f"✓ Parquet file saved successfully")

def save_quality_report(quality_df: pd.DataFrame, output_path: str):
    """
    Save the quality report as CSV with formatting.
    """
    print(f"\n=== Saving Quality Report ===")

    # Sort by quality issues (flagged columns first, then by blank rate)
    quality_df['has_flags'] = quality_df['quality_flags'] != 'OK'
    quality_df_sorted = quality_df.sort_values(['has_flags', 'pct_non_blank'], ascending=[False, True])

    # Drop the temporary sorting column
    quality_df_sorted = quality_df_sorted.drop('has_flags', axis=1)

    # Save to CSV
    quality_df_sorted.to_csv(output_path, index=False)

    print(f"✓ Quality report saved to: {output_path}")

    # Print summary statistics
    total_columns = len(quality_df)
    flagged_columns = len(quality_df[quality_df['quality_flags'] != 'OK'])
    high_blank_columns = len(quality_df[quality_df['pct_non_blank'] < 50])

    print(f"\n📊 Quality Summary:")
    print(f"  Total columns analyzed: {total_columns}")
    print(f"  Columns with quality flags: {flagged_columns}")
    print(f"  Columns with >50% blank rate: {high_blank_columns}")

    if flagged_columns > 0:
        print(f"\n⚠️  Columns requiring attention:")
        problem_cols = quality_df[quality_df['quality_flags'] != 'OK'][['column_name', 'pct_non_blank', 'quality_flags']]
        for _, row in problem_cols.head(10).iterrows():
            print(f"    {row['column_name']}: {row['pct_non_blank']:.1f}% filled, flags: {row['quality_flags']}")

        if len(problem_cols) > 10:
            print(f"    ... and {len(problem_cols) - 10} more (see CSV report)")

def extract_zip_file(zip_path, extraction_dir):
    """
    Extract a single zip file and return the paths of extracted files
    """
    print(f"📦 Extracting: {os.path.basename(zip_path)}")

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get list of files in the zip
            file_list = zip_ref.namelist()

            # Extract all files
            zip_ref.extractall(extraction_dir)

            # Find .sas and .dat files
            sas_files = [f for f in file_list if f.lower().endswith('.sas')]
            dat_files = [f for f in file_list if f.lower().endswith('.dat')]

            print(f"  ✓ Extracted {len(file_list)} files")
            print(f"  📋 Found {len(sas_files)} .sas files, {len(dat_files)} .dat files")

            return {
                'sas_files': [os.path.join(extraction_dir, f) for f in sas_files],
                'dat_files': [os.path.join(extraction_dir, f) for f in dat_files],
                'all_files': [os.path.join(extraction_dir, f) for f in file_list]
            }

    except Exception as e:
        print(f"❌ Error extracting {zip_path}: {e}")
        return None

def find_zip_files(base_path):
    """
    Find all zip files in the base directory
    """
    zip_pattern = os.path.join(base_path, "*.zip")
    zip_files = glob.glob(zip_pattern)

    print(f"🔍 Found {len(zip_files)} zip files in {base_path}")
    for zip_file in zip_files:
        print(f"  📦 {os.path.basename(zip_file)}")

    return zip_files

def extract_year_from_filename(filename):
    """
    Extract year from filename like 'opafy07nid.zip' -> '07'
    """
    match = re.search(r'fy(\d{2})', filename.lower())
    if match:
        return match.group(1)
    return "unknown"

def sas_to_parquet_optimized(sas_file_path, dat_file_path, parquet_output_path,
                           json_output_path=None, quality_report_path=None,
                           chunk_size=50000, quality_sample_size=10000):
    """
    Highly optimized workflow: SAS layout → Parse DAT → Quality Analysis → Save Parquet
    Uses vectorized operations, memory mapping, and sampling for maximum speed.
    """

    print(f"\n=== OPTIMIZED SAS to Parquet Workflow ===")
    print(f"Processing: {os.path.basename(sas_file_path)} & {os.path.basename(dat_file_path)}")
    print(f"Parse chunk size: {chunk_size:,} rows")
    print(f"Quality analysis sample size: {quality_sample_size:,} rows")

    # STEP 1: Parse SAS layout file
    print(f"\nStep 1: Reading SAS layout from {os.path.basename(sas_file_path)}")

    try:
        with open(sas_file_path, 'r', encoding='utf-8') as f:
            sas_content = f.read()

        result = parse_sas_file(sas_content)
        print(f"✓ Found {len(result['columns'])} columns in SAS layout")

    except FileNotFoundError:
        print(f"❌ Error: SAS file not found: {sas_file_path}")
        return None
    except UnicodeDecodeError:
        try:
            with open(sas_file_path, 'r', encoding='latin-1') as f:
                sas_content = f.read()
            result = parse_sas_file(sas_content)
            print(f"✓ Found {len(result['columns'])} columns in SAS layout (latin-1 encoding)")
        except Exception as e:
            print(f"❌ Error parsing SAS file: {e}")
            return None
    except Exception as e:
        print(f"❌ Error parsing SAS file: {e}")
        return None

    # STEP 2: Create JSON metadata
    print(f"\nStep 2: Creating JSON metadata")

    json_metadata = {}
    for col in result['columns']:
        json_metadata[col['name']] = {
            "start": col['start'],
            "end": col['end'],
            "width": col['width'],
            "type": col.get('type', 'numeric')
        }

    # Save JSON metadata if requested
    if json_output_path:
        os.makedirs(os.path.dirname(json_output_path), exist_ok=True)
        with open(json_output_path, 'w') as f:
            json.dump(json_metadata, f, indent=2)
        print(f"✓ JSON metadata saved to: {os.path.basename(json_output_path)}")

    # STEP 3: Detect encoding and estimate file size
    print(f"\nStep 3: File analysis")
    encoding = detect_file_encoding(dat_file_path)

    # STEP 4: Parse data with optimized method
    print(f"\nStep 4: Parsing data with optimized method")

    # Convert to pandas format
    sorted_columns = sorted(json_metadata.items(), key=lambda x: x[1]['start'])
    colspecs = [(col_info['start'], col_info['end']) for col_name, col_info in sorted_columns]
    column_names = [col_name for col_name, col_info in sorted_columns]

    try:
        # Use optimized parsing
        df = optimized_parse_fixed_width(dat_file_path, colspecs, column_names, encoding, chunk_size)
        total_rows = len(df)

        print(f"✓ Parsed {total_rows:,} rows, {len(df.columns)} columns")

    except Exception as e:
        print(f"❌ Error parsing data: {e}")
        return None

    # STEP 5: Quality analysis
    print(f"\nStep 5: Running optimized quality analysis")

    try:
        quality_df = optimized_quality_analysis(df, json_metadata, quality_sample_size)
        print(f"✓ Quality analysis completed for {len(quality_df)} columns")

        # Save quality report if requested
        if quality_report_path:
            os.makedirs(os.path.dirname(quality_report_path), exist_ok=True)
            save_quality_report(quality_df, quality_report_path)

    except Exception as e:
        print(f"⚠️  Warning: Quality analysis failed: {e}")
        quality_df = None

    # STEP 6: Save to Parquet
    print(f"\nStep 6: Saving to Parquet")

    try:
        save_to_parquet_optimized(df, parquet_output_path)

        # Check file sizes
        dat_size_mb = os.path.getsize(dat_file_path) / 1024**2
        parquet_size_mb = os.path.getsize(parquet_output_path) / 1024**2

        print(f"✓ Files saved!")
        print(f"  Original DAT: {dat_size_mb:.1f} MB")
        print(f"  Parquet: {parquet_size_mb:.1f} MB")
        print(f"  Compression: {(1 - parquet_size_mb/dat_size_mb)*100:.1f}% smaller")

    except Exception as e:
        print(f"❌ Error saving Parquet: {e}")
        return None

    # STEP 7: Show results
    print(f"\n✅ SUCCESS! File processing completed")
    print(f"Total rows processed: {total_rows:,}")
    print(f"Columns: {len(column_names)}")

    # Clean up the DataFrame from memory
    del df

    if quality_df is not None:
        return total_rows, quality_df
    else:
        return total_rows, None

def load_processed_files_log(log_path):
    """Loads the list of already processed zip filenames from a log file."""
    processed_files = set()
    if os.path.exists(log_path):
        try:
            with open(log_path, 'r') as f:
                for line in f:
                    processed_files.add(line.strip())
            print(f"Loaded {len(processed_files)} processed files from log: {os.path.basename(log_path)}")
        except Exception as e:
            print(f"⚠️  Warning: Could not read processed files log {os.path.basename(log_path)}: {e}")
    else:
        print(f"Processed files log not found: {os.path.basename(log_path)}. Starting fresh.")
    return processed_files

def log_processed_file(log_path, filename):
    """Appends a successfully processed zip filename to the log file."""
    try:
        with open(log_path, 'a') as f:
            f.write(f"{filename}\n")
        print(f"Appended {filename} to processed files log.")
    except Exception as e:
        print(f"❌ Error writing to processed files log {os.path.basename(log_path)}: {e}")


def process_all_zip_files(base_path, folder_id, processed_log_path, chunk_size=50000, quality_sample_size=10000):
    """
    Main function to process all zip files in the directory, now restartable.
    """

    print("🚀 === BATCH PROCESSING ALL ZIP FILES (Restartable) ===")
    print(f"Base path: {base_path}")
    print(f"Folder ID: {folder_id}")
    print(f"Processed log: {os.path.basename(processed_log_path)}")


    # Create necessary directories
    parquet_dir = os.path.join(base_path, "parquet_files")
    metadata_dir = os.path.join(base_path, "metadata_logs")
    error_dir = os.path.join(base_path, "error_logs")

    os.makedirs(parquet_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)
    os.makedirs(error_dir, exist_ok=True)

    # Load already processed files
    processed_files = load_processed_files_log(processed_log_path)

    # Find all zip files
    all_zip_files = find_zip_files(base_path)

    if not all_zip_files:
        print("❌ No zip files found!")
        return

    # Filter out already processed files
    zip_files_to_process = [f for f in all_zip_files if os.path.basename(f) not in processed_files]

    print(f"📋 Found {len(all_zip_files)} total zip files.")
    print(f"⏭️ Skipping {len(all_zip_files) - len(zip_files_to_process)} already processed files.")
    print(f"🎯 Will process {len(zip_files_to_process)} files in this run.")

    if not zip_files_to_process:
        print("🎉 No new files to process. Exiting.")
        return

    # Process each zip file
    results = []
    successful_processes = 0
    failed_processes = 0

    for i, zip_file_path in enumerate(zip_files_to_process, 1):
        zip_filename = os.path.basename(zip_file_path)
        year = extract_year_from_filename(zip_filename)

        print(f"\n{'='*80}")
        print(f"🔄 PROCESSING FILE {i}/{len(zip_files_to_process)}: {zip_filename}")
        print(f"{'='*80}")

        try:
            # Extract the zip file
            extracted_files = extract_zip_file(zip_file_path, base_path)

            if not extracted_files:
                print(f"❌ Failed to extract {zip_filename}")
                failed_processes += 1
                continue

            # Find matching .sas and .dat files
            sas_files = extracted_files['sas_files']
            dat_files = extracted_files['dat_files']

            if not sas_files or not dat_files:
                print(f"❌ No matching .sas/.dat files found in {zip_filename}")
                failed_processes += 1
                continue

            # Assume first matching pair (they should have same base name)
            sas_file = sas_files[0]
            dat_file = dat_files[0]

            # Define output paths
            base_name = os.path.splitext(os.path.basename(sas_file))[0]
            parquet_file = os.path.join(parquet_dir, f"{base_name}.parquet")
            json_file = os.path.join(metadata_dir, f"column_metadata_fy{year}.json")
            quality_report = os.path.join(error_dir, f"quality_audit_report_fy{year}.csv")

            print(f"📋 Processing files:")
            print(f"  SAS: {os.path.basename(sas_file)}")
            print(f"  DAT: {os.path.basename(dat_file)}")
            print(f"  Output: {os.path.basename(parquet_file)}")

            # Process the files
            result = sas_to_parquet_optimized(
                sas_file, dat_file, parquet_file,
                json_file, quality_report,
                chunk_size, quality_sample_size
            )

            if result is not None:
                total_rows, quality_df = result

                results.append({
                    'file': base_name,
                    'year': year,
                    'rows': total_rows,
                    'columns': len(quality_df) if quality_df is not None else 0,
                    'status': 'SUCCESS',
                    'parquet_path': parquet_file
                })

                successful_processes += 1
                print(f"✅ Successfully processed {zip_filename}")
                log_processed_file(processed_log_path, zip_filename) # Log success

            else:
                results.append({
                    'file': base_name,
                    'year': year,
                    'rows': 0,
                    'columns': 0,
                    'status': 'FAILED',
                    'parquet_path': None
                })
                failed_processes += 1
                print(f"❌ Failed to process {zip_filename}")

            # Clean up extracted files to save space (optional)
            print("🧹 Cleaning up extracted files...")
            for file_path in extracted_files['all_files']:
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                except:
                    pass  # Ignore cleanup errors

        except Exception as e:
            print(f"❌ Unexpected error processing {zip_filename}: {e}")
            failed_processes += 1
            results.append({
                'file': zip_filename,
                'year': year,
                'rows': 0,
                'columns': 0,
                'status': f'ERROR: {str(e)[:50]}',
                'parquet_path': None
            })

    # Final summary
    print(f"\n{'='*80}")
    print(f"🎉 BATCH PROCESSING COMPLETE!")
    print(f"{'='*80}")
    print(f"Total files found: {len(all_zip_files)}")
    print(f"Files attempted in this run: {len(zip_files_to_process)}")
    print(f"Successful in this run: {successful_processes}")
    print(f"Failed in this run: {failed_processes}")
    print(f"Files skipped (already processed): {len(all_zip_files) - len(zip_files_to_process)}")


    if results:
        print(f"\n📊 PROCESSING SUMMARY FOR THIS RUN:")
        total_rows = 0
        for result in results:
            status_icon = "✅" if result['status'] == 'SUCCESS' else "❌"
            print(f"  {status_icon} {result['file']} (FY{result['year']}): {result['rows']:,} rows, {result['columns']} cols")
            if result['status'] == 'SUCCESS':
                total_rows += result['rows']

        print(f"\n📈 TOTAL DATA PROCESSED IN THIS RUN: {total_rows:,} rows across {successful_processes} files")

        # Save summary report for this run
        summary_df = pd.DataFrame(results)
        # Append to existing summary or create new one
        summary_path = os.path.join(base_path, "processing_summary.csv")
        if os.path.exists(summary_path):
            existing_summary_df = pd.read_csv(summary_path)
            # Avoid duplicating entries for files that might have been processed in previous runs
            # Only append new results or update status for files attempted in this run
            combined_summary_df = pd.concat([existing_summary_df, summary_df]).drop_duplicates(subset=['file'], keep='last')
            combined_summary_df.to_csv(summary_path, index=False)
            print(f"💾 Processing summary updated at: processing_summary.csv")
        else:
             summary_df.to_csv(summary_path, index=False)
             print(f"💾 Processing summary saved to: processing_summary.csv")


        # List all created parquet files (from this run or previous successful runs)
        # Reload the summary to get the full list of successful files
        full_summary_df = pd.read_csv(summary_path)
        successful_files_overall = full_summary_df[full_summary_df['status'] == 'SUCCESS']

        if not successful_files_overall.empty:
            print(f"\n📦 ALL SUCCESSFUL PARQUET FILES:")
            for _, row in successful_files_overall.iterrows():
                 try:
                    file_size_mb = os.path.getsize(row['parquet_path']) / 1024**2
                    print(f"  📄 {os.path.basename(row['parquet_path'])} ({file_size_mb:.1f} MB)")
                 except FileNotFoundError:
                     print(f"  📄 {os.path.basename(row['parquet_path'])} (File not found, likely processed in another session/location)")


            print(f"\n🔗 To load any file later:")
            print(f"df = pd.read_parquet('/content/drive/.shortcut-targets-by-id/{folder_id}/Dissertation Data/parquet_files/FILENAME.parquet')")

    return results

# =============================================================================
# MAIN EXECUTION FOR GOOGLE COLAB
# =============================================================================

if __name__ == "__main__":

    # Define shared folder ID and base path (UPDATE THIS)
    folder_id = ""  # Replace with your actual folder ID
    base_path = f"c:/college/ualr/research/dissertation_data"

    # Define the path for the processed files log
    processed_log_path = os.path.join(base_path, "processed_zip_files.log")

    # Optimized settings for Colab
    PARSE_CHUNK_SIZE = 20000        # Smaller chunks for parsing to save memory
    QUALITY_SAMPLE_SIZE = 10000     # Sample size for quality analysis (much faster)

    print("🚀 Starting batch processing of all SAS/DAT zip files...")
    print(f"📁 Base directory: {base_path}")

    # Check if the base path exists
    if not os.path.exists(base_path):
        print(f"❌ Error: Base path does not exist: {base_path}")
        print("Please check your folder ID and make sure Google Drive is mounted.")
        exit()

    # Process all zip files
    results = process_all_zip_files(
        base_path=base_path,
        folder_id=folder_id,
        processed_log_path=processed_log_path, # Pass the log path
        chunk_size=PARSE_CHUNK_SIZE,
        quality_sample_size=QUALITY_SAMPLE_SIZE
    )

    if results is not None: # Check if process_all_zip_files didn't return None due to errors
        successful_count = len([r for r in results if r and r.get('status') == 'SUCCESS']) # Use .get for safety
        print(f"\n🎊 ALL DONE! Successfully processed {successful_count} files in this run.")
        print(f"📂 Check the following directories for outputs:")
        print(f"  📦 Parquet files: parquet_files/")
        print(f"  📋 Metadata: metadata_logs/")
        print(f"  📊 Quality reports: error_logs/")
        print(f"  📝 Processed log: processed_zip_files.log")
        print(f"  📈 Summary report: processing_summary.csv")
    else:
        print("\n❌ An error occurred during batch processing.")


# =============================================================================
# OPTIONAL: INDIVIDUAL FILE PROCESSING FUNCTION
# =============================================================================

def process_single_file(base_path, folder_id, filename, processed_log_path, chunk_size=50000, quality_sample_size=10000):
    """
    Process a single zip file (useful for testing or re-processing specific files)

    Usage:
    # Define shared folder ID and base path (UPDATE THIS)
    folder_id = "1X6i8PenemApzzoWSZE3quuVZ_ZKayaav"  # Replace with your actual folder ID
    base_path = f"/content/drive/.shortcut-targets-by-id/{folder_id}/Dissertation Data"
    processed_log_path = os.path.join(base_path, "processed_zip_files.log")

    process_single_file(base_path, folder_id, "opafy07nid.zip", processed_log_path)
    """

    zip_file_path = os.path.join(base_path, filename)

    if not os.path.exists(zip_file_path):
        print(f"❌ File not found: {filename}")
        return None

    # Check if the file is already processed
    processed_files = load_processed_files_log(processed_log_path)
    if filename in processed_files:
        print(f"⏭️ File {filename} is already marked as processed. Skipping.")
        return {'file': filename, 'year': extract_year_from_filename(filename), 'rows': 0, 'columns': 0, 'status': 'SKIPPED', 'parquet_path': None}


    print(f"🔄 Processing single file: {filename}")

    # Extract year from filename
    year = extract_year_from_filename(filename)

    # Extract the zip file
    extracted_files = extract_zip_file(zip_file_path, base_path)

    if not extracted_files:
        print(f"❌ Failed to extract {filename}")
        return {'file': filename, 'year': year, 'rows': 0, 'columns': 0, 'status': 'EXTRACTION_FAILED', 'parquet_path': None}


    # Find matching .sas and .dat files
    sas_files = extracted_files['sas_files']
    dat_files = extracted_files['dat_files']

    if not sas_files or not dat_files:
        print(f"❌ No matching .sas/.dat files found in {filename}")
        return {'file': filename, 'year': year, 'rows': 0, 'columns': 0, 'status': 'NO_MATCHING_FILES', 'parquet_path': None}


    # Use first matching pair
    sas_file = sas_files[0]
    dat_file = dat_files[0]

    # Define output paths
    base_name = os.path.splitext(os.path.basename(sas_file))[0]
    parquet_file = os.path.join(base_path, "parquet_files", f"{base_name}.parquet")
    json_file = os.path.join(base_path, "metadata_logs", f"column_metadata_fy{year}.json")
    quality_report = os.path.join(base_path, "error_logs", f"quality_audit_report_fy{year}.csv")

    # Create directories
    os.makedirs(os.path.dirname(parquet_file), exist_ok=True)
    os.makedirs(os.path.dirname(json_file), exist_ok=True)
    os.makedirs(os.path.dirname(quality_report), exist_ok=True)

    # Process the files
    result = sas_to_parquet_optimized(
        sas_file, dat_file, parquet_file,
        json_file, quality_report,
        chunk_size, quality_sample_size
    )

    # Clean up extracted files
    print("🧹 Cleaning up extracted files...")
    for file_path in extracted_files['all_files']:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except:
            pass

    if result is not None:
         total_rows, quality_df = result
         # Log successful processing
         log_processed_file(processed_log_path, filename)
         return {'file': base_name, 'year': year, 'rows': total_rows, 'columns': len(quality_df) if quality_df is not None else 0, 'status': 'SUCCESS', 'parquet_path': parquet_file}
    else:
         return {'file': base_name, 'year': year, 'rows': 0, 'columns': 0, 'status': 'PROCESSING_FAILED', 'parquet_path': None}
