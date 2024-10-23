import os
from bs4 import BeautifulSoup
import sys
import requests

#Provide the info necessary to download files from a specified website
def get_args():
    """Parse command-line arguments."""
    if len(sys.argv) != 5:
        print("Invalid input, insert correct inputs only.")
        sys.exit(1)
    url = sys.argv[1]
    file_type = sys.argv[2]
    memory_limit_gb = float(sys.argv[3])
    target_folder = sys.argv[4]
    return url, file_type, memory_limit_gb, target_folder

#Prevent excesseive use of memory
def get_total_used_space_gb(path):
    """Calculate the total used space in the target folder (in GB)."""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size / (1024**3)  # Convert bytes to GB

def download_file(url, target_path):
    """Download a file from a URL to a specific path."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(target_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return True
    except requests.RequestException as e:
        print(f"Download failed: {e}")
        return False

def check_and_handle_conflict(file_name, target_folder):
    """Resolve file name conflicts."""
    base_name, ext = os.path.splitext(file_name)
    counter = 1
    new_name = file_name
    while os.path.exists(os.path.join(target_folder, new_name)):
        new_name = f"{base_name}({counter}){ext}"
        counter += 1
    return new_name

def main():
    # Get the arguments
    url, file_type, memory_limit_gb, target_folder = get_args()

    # Create a log file
    log_file = os.path.join(target_folder, f"download_log_{os.path.basename(target_folder)}.txt")
    with open(log_file, 'w') as log:
        log.write("Download Log:\n")

    # Get the list of files to download
    # (This is a placeholder; you should implement code to fetch file list from the URL)
    file_list = []  # Replace with actual file list extraction based on file_type
    
    # Track the used space
    used_space_gb = get_total_used_space_gb(target_folder)

    # Download each file
    for file_name in file_list:
        if used_space_gb >= memory_limit_gb:
            print("Data use limit reached.")
            with open(log_file, 'a') as log:
                log.write("Data use limit reached.\n")
            sys.exit(1)

        # Resolve conflicts in the file name
        file_name = check_and_handle_conflict(file_name, target_folder)
        target_path = os.path.join(target_folder, file_name)

        # Download the file
        if download_file(url, target_path):
            used_space_gb += os.path.getsize(target_path) / (1024**3)  # Update used space
            status = "Success"
        else:
            status = "Failed"
            if os.path.exists(target_path):
                os.remove(target_path)  # Clean up partially downloaded file

        # Log the result
        with open(log_file, 'a') as log:
            log.write(f"{file_name}: {status}\n")

    print("Download process completed.")

if __name__ == "__main__":
    main()
