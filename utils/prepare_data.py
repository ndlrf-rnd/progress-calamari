import zipfile
import requests
import io


def get_zip(file_url, output_dir):
    print('Downloading {} ...'.format(file_url))
    url = requests.get(file_url)
    zf = zipfile.ZipFile(io.BytesIO(url.content))
    zip_names = zf.namelist()
    print('Extracting {} files -> {} ...'.format(len(zip_names), output_dir))
    zf.extractall(output_dir)
