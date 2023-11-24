import os
import zipfile
import gdown

def down_drive(url: str, dest: str):
    gdown.download(url, dest, quiet=False)

def unzip_file(file_path: str, extract_path: str):
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

def download(link, dest_folder):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    zip_file_path = 'downloaded_file.zip'
    
    down_drive(link, zip_file_path)

    unzip_file(zip_file_path, dest_folder)

    os.remove(zip_file_path)

    

if __name__ == "__main__":
    resources_folder: str = "resources"
    if not os.path.exists(resources_folder):
        os.makedirs(resources_folder)


    classification_folder: str = os.path.join(resources_folder, "classification")

    download('https://drive.google.com/uc?export=download&id=18HcPbWbgcGSpO0iRqqeUlZ2pnXvaQQop', classification_folder)

    download("https://drive.google.com/u/0/uc?id=1LS60vVxNxQIZ_aPyxvEBGGsxDN8aSSp2&export=download", resources_folder)
