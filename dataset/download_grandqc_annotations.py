import requests
from tqdm import tqdm
import os

def download_file(url, output_dir):
    """
    Download a file from a given URL and save it to the specified directory.

    Args:
        url (str): The URL to download the file from.
        output_dir (str): The directory to save the downloaded file.
    """
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get the filename from the URL
    filename = url.split("/")[-1].split("?")[0]  # Extract filename before "?" if present
    output_path = os.path.join(output_dir, filename)

    # Send a GET request with streaming
    with requests.get(url, stream=True) as response:
        response.raise_for_status()  # Raise an error for HTTP issues
        total_size = int(response.headers.get("content-length", 0))

        # Use tqdm for the progress bar
        with open(output_path, "wb") as file:
            with tqdm(
                desc=f"Downloading {filename}",
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as progress_bar:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
                    progress_bar.update(len(chunk))

    print(f"Download completed: {output_path}")

def download_files(urls, output_dir):
    """
    Download multiple files from a list of URLs and save them to the specified directory.

    Args:
        urls (list): A list of URLs to download files from.
        output_dir (str): The directory to save the downloaded files.
    """
    for url in urls:
        try:
            download_file(url, output_dir)
        except Exception as e:
            print(f"Failed to download {url}: {e}")

# Example usage
if __name__ == "__main__":
    urls = [
        "https://zenodo.org/api/records/14041578/files-archive"
    ]
    output_dir = "/home/bogdan/indonezia/data/GRAND-QC/data/manual_annotations"  # Change this to your desired output directory

    download_files(urls, output_dir)