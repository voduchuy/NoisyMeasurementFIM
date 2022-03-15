from pathlib import Path
import zipfile
import shutil
import wget
# URL to Zenodo upload
url = "https://zenodo.org/record/6354728/files/zenodo_upload.zip?download=1"
# Destination folder to unpack the files
# Uncomment for testing
# dest = Path("./zenodo_download")
# Uncomment for unpacking directly to the local repository
dest = "./"
if not dest.exists():
    dest.mkdir()
#%% Download results from Zenodo
wget.download(url=url)
#%% Upzip the downloaded file into a folder "zenodo_upload" in the cwd
with zipfile.ZipFile("./numerical_results.zip", "r") as zf:
    for file in zf.infolist():
        if file.filename.startswith("zenodo_upload"):
            zf.extract(file)
unpacked_loc = Path("./zenodo_upload")
#%% Move files in unpacked subfolders into dest
# Loop through three gene model examples
for example in ["bursting_gene", "toggle_switch", "yeast"]:
    # This will create the example structure for the example with <model>/results and <model>/figs
    if not dest.joinpath(example).joinpath("results").exists():
        dest.joinpath(example).joinpath("results").mkdir(parents=True)
    if not dest.joinpath(example).joinpath("figs").exists():
        dest.joinpath(example).joinpath("figs").mkdir(parents=True)

    # Copy all files from the unpacked folder to the destination
    shutil.copytree(unpacked_loc.joinpath(example).joinpath("results/"), dest.joinpath(example).joinpath("results/"), dirs_exist_ok=True)
    shutil.copytree(unpacked_loc.joinpath(example).joinpath("figs/"), dest.joinpath(example).joinpath("figs"), dirs_exist_ok=True)
#%%
shutil.rmtree(unpacked_loc)
shutil.rm("./numerical_results.zip")








