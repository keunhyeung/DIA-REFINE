# -*- coding: utf-8 -*-
"""
Anonymized Data Processing Script for Dialect Corpus

Description:
This script processes the *text portion* of the "Korean Dialect Data of 
Middle-aged and Elderly Speakers (NIA, 2022)" corpus, available at AI-Hub.

The original corpus contains both audio files (speech conversations) and their 
corresponding text transcriptions (in JSON files). This script focuses *only* on the JSON transcriptions, parsing them to extract parallel standard 
and dialect sentences (ignoring the audio data).

It dynamically generates paths for a *single selected dialect* using path templates, 
processes both its train and valid splits, and saves the results into structured CSV files.

Data Source: https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=71517

Instructions for Reproducibility:
1.  Download the data from the AI-Hub link above (Data Source).
2.  Unzip all dialect files (e.g., '제주도', '전라도', etc.) into a single 
    main directory. This will be your "corpus root".
3.  Set the `SOURCE_ROOT_DIR` variable (in section 1-B) to the absolute path 
    of this main directory (e.g., "/path/to/your/corpus_root_directory").
4.  Set the `DIALECT_TO_PROCESS` variable (in section 1-A) to the *one* dialect 
    you want to process (e.g., '제주도'). This key must be in Korean.
5.  Set the `OUTPUT_ROOT_DIR` (in section 1-B) to the base path where you want 
    to save the processed CSVs.
6.  Verify the `DIALECT_MAP` (in section 2) contains the correct English 
    folder name for each Korean dialect key.
7.  Verify/update the `TRAIN_PATH_TEMPLATES` and `VALID_PATH_TEMPLATES` (in section 3) 
    to match the *relative* sub-folder structure within your `SOURCE_ROOT_DIR`. 
    The `{dialect_ko}` placeholder will be automatically replaced.
8.  Run the script.
"""
import os
import json
import pandas as pd
from tqdm import tqdm

# --- 1. User Configuration ---

# --- A. Select Dialect ---
# Select *one* dialect to process from the 5 options.
# (This name must match a key in DIALECT_MAP.)
DIALECT_TO_PROCESS = "제주도" 


# --- B. Set Root Paths ---
# (Sensitive Info) Absolute base path to your raw corpus directory.
# e.g., "/nas_homes/username/backtranslation/Dialect_Dataset"
SOURCE_ROOT_DIR = "path/to/your/corpus_root_directory"

# (Sensitive Info) Absolute base path to save the processed CSVs.
# e.g., "/home/username/backtranslation/data/new_source"
OUTPUT_ROOT_DIR = "path/to/your/processed_data_directory"


# --- 2. Dialect Metadata ---
# Map of dialect names (Korean) to output folder names (English).
DIALECT_MAP = {
    "제주도": "jeju",
    "전라도": "jeonlla",
    "경상도": "gyeongsang",
    "강원도": "gangwon",
    "충청도": "chungcheong"
}

# --- 3. Path Templates ---
# The {dialect_ko} placeholder will be replaced by the `DIALECT_TO_PROCESS` value (e.g., "제주도").
# (This assumes the path structure is identical for all dialects.)

TRAIN_PATH_TEMPLATES = [
    "path/to/your/corpus/{dialect_ko}/source_1",
    "path/to/your/corpus/{dialect_ko}/source_2" # Example: different source
]

VALID_PATH_TEMPLATES = [
   "path/to/your/corpus/{dialect_ko}/valid_source_1",
   "path/to/your/corpus/{dialect_ko}/valid_source_2"
]
# --- End of Configuration ---


def process_and_save(relative_paths, output_csv_path):
    """
    Processes all JSON files from a list of relative directories 
    and merges them into a single CSV file.

    Args:
        relative_paths (list): A list of relative paths (from SOURCE_ROOT_DIR).
        output_csv_path (str): The full, absolute path for the output CSV file.
        
    Returns:
        int: The number of extracted sentence pairs.
    """
    
    # 1. Build absolute source directories
    json_source_dirs = [os.path.join(SOURCE_ROOT_DIR, p) for p in relative_paths]

    # 2. Collect all JSON file paths
    all_json_files = []
    for json_dir in json_source_dirs:
        if not os.path.isdir(json_dir):
            print(f"    [Warning] Directory not found, skipping: {json_dir}")
            continue
        try:
            for filename in os.listdir(json_dir):
                if filename.endswith(".json"):
                    all_json_files.append(os.path.join(json_dir, filename))
        except OSError as e:
            print(f"    [Error] Cannot access directory {json_dir}: {e}")

    if not all_json_files:
        print(f"    [Info] No JSON files found for this task.")
        return 0

    # 3. List to store extracted sentence pairs
    extracted_data = []

    # Iterate through all JSON files and extract parallel sentences
    print(f"    [Info] Found {len(all_json_files)} JSON files. Processing...")
    for file_path in tqdm(all_json_files, desc="  Processing JSONs", leave=False, ncols=80):
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                json_data = json.load(f)
                # Navigate the JSON to get the list of sentence transcriptions
                sentences = json_data.get("transcription", {}).get("sentences", [])
                for sentence_pair in sentences:
                    standard_form = sentence_pair.get("standard", "").strip()
                    dialect_form = sentence_pair.get("dialect", "").strip()
                    
                    # Append only if both standard and dialect forms exist
                    if standard_form and dialect_form:
                        extracted_data.append({
                            "standard_form": standard_form, 
                            "dialect_form": dialect_form
                        })
            except json.JSONDecodeError:
                print(f"    [Error] Could not decode JSON: {os.path.basename(file_path)}")
            except Exception as e:
                print(f"    [Error] Unexpected error processing {os.path.basename(file_path)}: {e}")

    if not extracted_data:
        print("    [Warning] No data was extracted. Output file will not be created.")
        return 0

    # 4. Create a DataFrame
    df = pd.DataFrame(extracted_data)

    # 5. Save the DataFrame to a CSV file
    try:
        output_dir = os.path.dirname(output_csv_path)
        os.makedirs(output_dir, exist_ok=True)
        
        df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        return len(df)

    except OSError as e:
        print(f"    [Fatal Error] Failed to write output file {output_csv_path}: {e}")
        return 0


def main():
    """
    Main function to process the *single selected dialect* using path templates.
    """
    print("--- Starting Dialect Data Processing ---")
    
    dialect_ko = DIALECT_TO_PROCESS  # e.g., "제주도"
    
    # 1. Get metadata for the selected dialect
    output_folder = DIALECT_MAP.get(dialect_ko) # e.g., "jeju"
    output_prefix = dialect_ko                 # e.g., "제주도"
    
    if not output_folder:
        print(f"▶ [Fatal Error] Dialect '{dialect_ko}' not found in DIALECT_MAP.")
        print(f"  Available dialects are: {list(DIALECT_MAP.keys())}")
        print("--- Processing Aborted ---")
        return

    print(f"▶ Processing Selected Dialect: {dialect_ko}")

    # --- 2. Process TRAINING data ---
    
    # Dynamically create train data paths using templates
    train_relative_paths = [t.format(dialect_ko=dialect_ko) for t in TRAIN_PATH_TEMPLATES]
    
    # Set output path
    output_filename_train = f"{output_prefix}_training.csv"
    output_path_train = os.path.join(OUTPUT_ROOT_DIR, output_folder, output_filename_train)
    
    print(f"  -> Processing train data...")
    count_train = process_and_save(train_relative_paths, output_path_train)
    if count_train > 0:
        print(f"  -> [OK] Saved {count_train} training sentences to: {output_path_train}")
    else:
        print(f"  -> [Info] No data processed for train.")


    # --- 3. Process VALIDATION data ---
    
    # Dynamically create validation data paths using templates
    valid_relative_paths = [t.format(dialect_ko=dialect_ko) for t in VALID_PATH_TEMPLATES]

    # Set output path
    output_filename_valid = f"{output_prefix}_valid.csv"
    output_path_valid = os.path.join(OUTPUT_ROOT_DIR, output_folder, output_filename_valid)
    
    print(f"  -> Processing valid data...")
    count_valid = process_and_save(valid_relative_paths, output_path_valid)
    if count_valid > 0:
        print(f"  -> [OK] Saved {count_valid} validation sentences to: {output_path_valid}")
    else:
        print(f"  -> [Info] No data processed for valid.")

    print(f"\n--- Processing complete for {dialect_ko}. ---")

if __name__ == '__main__':
    main()