import os
import requests
from tqdm import tqdm
import json

def download_file(file_id, filename, output_dir, total_files, current_file):
    """
    Download a file from the GDC API using its file_id and track progress using tqdm.
    """
    base_url = "https://api.gdc.cancer.gov/data/"
    url = f"{base_url}{file_id}"
    headers = {"Content-Type": "application/json"}

    try:
        # Send request
        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()  # Raise an exception for HTTP errors

        filepath = os.path.join(output_dir, filename)

        # Get total file size from headers
        total_size = int(response.headers.get("Content-Length", 0))
        chunk_size = 1024 * 1024  # 1 MB chunks for faster download

        # Initialize tqdm progress bar
        with tqdm(total=total_size, unit="B", unit_scale=True, unit_divisor=1024, 
                  desc=f"Downloading {current_file}/{total_files}: {filename}") as progress_bar:
            with open(filepath, "wb") as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:  # Ensure chunk is not empty
                        f.write(chunk)
                        progress_bar.update(len(chunk))
        print(f"Downloaded: {filename} to {output_dir}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to download file_id {file_id}: {e}")


def process_json_and_download(main_json_path, output_dir):
    # Load the main JSON file
    with open(main_json_path, 'r') as main_file:
        main_data = json.load(main_file)

    for project_name, cases in tqdm(main_data.items(), desc="Processing projects"):
        # Create subdirectory for each project (without 'TCGA-')
        project_dir = os.path.join(output_dir, project_name.replace("TCGA-", ""))
        if not os.path.exists(project_dir):
            os.makedirs(project_dir)

        # Count total files for progress tracking
        total_files = sum(len(files) for files in cases.values())
        current_file = 0

        for case_name, files in tqdm(cases.items(), desc=f"Processing cases in {project_name}", leave=False):
            for file_entry in files:
                current_file += 1
                file_id = file_entry['file_id']
                file_name = file_entry['file_name']

                if not file_id:
                    tqdm.write(f"Skipping file {file_name} as it has no file_id.")
                    continue

                # Define file path
                file_path = os.path.join(project_dir, file_name)

                # Check if file already exists
                if os.path.exists(file_path):
                    existing_size = os.path.getsize(file_path)

                    # Fetch the file size from the server
                    try:
                        response = requests.head(f"https://api.gdc.cancer.gov/data/{file_id}", headers={"Content-Type": "application/json"})
                        response.raise_for_status()
                        remote_size = int(response.headers.get("Content-Length", 0))

                        if existing_size == remote_size:
                            tqdm.write(f"Skipping download for {file_name} (already downloaded with correct size).")
                            continue
                        else:
                            tqdm.write(f"File size mismatch for {file_name}. Redownloading.")
                            os.remove(file_path)
                    except requests.exceptions.RequestException as e:
                        tqdm.write(f"Failed to fetch size for {file_name}: {e}")
                        continue

                # Download the file
                download_file(file_id, file_name, project_dir, total_files, current_file)

# Example usage
if __name__ == "__main__":
    main_json_path = "grandqc_wsi.json"  # Path to your main JSON file
    output_dir = "/media/ceachi/Storage_2/Tissue Segmentation/data/TCGA/wsi"  # Directory to save downloaded files

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    process_json_and_download(main_json_path, output_dir)
