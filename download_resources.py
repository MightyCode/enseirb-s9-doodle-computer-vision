import os
import zipfile
import gdown

resources_folder: str = "resources"

def down_drive(url: str, dest: str):
    gdown.download(url, dest, quiet=False)

def unzip_file(file_path: str, extract_path: str):
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

def download():
    link = 'https://drive.google.com/uc?export=download&id=18HcPbWbgcGSpO0iRqqeUlZ2pnXvaQQop'
    zip_file_path = 'downloaded_file.zip'
    
    down_drive(link, zip_file_path)

    unzip_file(zip_file_path, resources_folder)

    os.remove(zip_file_path)

if __name__ == "__main__":
    download()
