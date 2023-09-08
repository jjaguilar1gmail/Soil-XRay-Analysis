# Soil-XRay-Analysis: Soil Density and Heterogeneity Analysis

This repository contains code to analyze local soil density from x-ray scans and characterize soil heterogeneity using statistical metrics like skewness, kurtosis, and Sobel edge detection. The analysis explores differences in soil structure across various soil treatments and management practices. The scanned soils cores that were analyzed for this project are from a long-term experiment at the North Agronomy Farm of Kansas State University in Manhattan, KS. For more info on data from that experiment:

Nicoloso, Rodrigo S., et al. "Carbon saturation and translocation in a no-till soil under organic amendments." Agriculture, Ecosystems & Environment 264 (2018): 73-84.

The soils are divided into two different practices each of tillage (chisel till - T, and no till - NT) and fertilizer (high fertilizer - HF, and high manure - HM). Native Prairie (NP) soil that is undisturbed by agricultural practice was also sampled. There are 54 scans in total used for this analysis. Metadata for the scans is provided in meta.csv. Within this project structure, all loadable data except for the xray scans themselves are located under ../data/.

## Directory structure

- `data/`: Contains raw and processed data
  - `meta.csv`: Metadata about scans
  - `precomputed_soil_stats*.csv`: Precomputed metrics
  - `sample_images/`: Example scan slices
  - `sample_images_recompute/`: Scans for recomputing metrics
- `notebooks/`: Jupyter notebooks for analysis
  - `N1_Basics.ipynb`: Initial processing
  - `N2_Local_Stats.ipynb`: Density calculation
  - `N3_Batch_Calculate_Soil_Statistics.ipynb`: Metric computation
  - `N4_Statistical_Metrics_Data_Analysis.ipynb`: Statistical analysis
  - 'N4b_Soil_Metrics_vs_XRay_Images.ipynb': Visualize xray slices vs computed metrics
  - `N5_Appendix_Segmentation.ipynb`: Image segmentation
- `slides/`: PDF slides explaining analysis 
- `src/xray_stats/`: Python module with core functions
- `streamlit/`: Streamlit app for visualization of metrics
- 'requirements.txt'
- 'setup.py'

## Analysis overview

- Compute local density from scans
- Extract statistical metrics for heterogeneity
- Apply techniques like ANOVA, t-SNE, SVM  
- Quantify impact on x-ray local heterogeneity of agricultural practices

## Usage

Jupyter notebooks numbered by order of usage/analysis. The Jupyter notebooks make use of modules custom written for this project (xray_stats/load_process.py and xray_stats/df_plotting.py). To run these notebooks, you will need to install these modules along with the packages in requirements.txt.

### Setup
**It is not necessary but highly recommended that you create a virtual environment.** To do so, prior to installing packages, open a terminal window and, from the main directory, run the following commands:

From the main directory, run:

```bash
python -m venv .venv
```

Activate the virtual environment:

```bash
# Linux/Mac
source .venv/bin/activate

# Windows 
.venv\Scripts\activate.bat
```

Install requirements:

```bash 
pip install -r requirements.txt
```

Install xray_stats package:

```bash
pip install --editable .
```

This will install xray_stats by running `setup.py`. 

The `--editable` option allows you to make changes to `xray_stats/` files without reinstalling. You may need to restart your Jupyter kernel.

After installation you can run the notebooks by running:

```bash
jupyter notebook
```
Once done, you can deactivate the virtual environment with the `deactivate` command.

### IMPORTANT: 
This repository does not contain the x-ray scans themselves, to use the code found in the notebooks used here to process raw x-ray scans, the load_process module of the provided xray_stats library has many functions to access the xray scans, process statistics and visualize. This code assumes the xray data is located on a locally accessible directory (if the files need to be accessed remotely through ftp, sftp or other such protocol, significant modifications to the code are required). The location of all these files is by default set to **path_to_tiff_folders="/Volumes/FreeAgent GoFlex Drive/DanforthXRAYData/"** in the xray_stats/load_process library. *Make sure to change this to the correct location of the xray data by editing get_tiff_stack in load_process.py prior to installing the library.* The structure should look as follows:

- path_to_tiff_folders/
    - Scan_Number1_39um/
        - Im_0000.tif
        - Im_0001.tif
        - Im_0002.tif
        - ...
    - Scan_Number2_39um/
        - Im_0000.tif
        - ...
    - ...

The folder and file naming convention is important to how the library parses and indexes the scans and tiff files. **For soil scan folders**, make sure the folder name is divided by underscores. The library takes the second to last underscore-seperated section and gets the number to assign the index. For example: 20210501Sbz_VBCSoilCores7_39um, its scan (or stack for tiff stack) index will be 7. **For file names**, it will use the last four digits before the extension for the image file's index. For example: 20210501_VBCsoilcore_7_39um 3_0002.tif will have an index of 2. Useful for ordering and assigning depth.

Notebooks N1, N2, N3 and N5 all make computations that require access to raw xrays. However, for the x-ray data used in this analysis, I have provided procomputed metrics for further analysis and inspection without need for access to raw xrays (see notebooks N4 and N4b).
See requirements.txt for required libraries.
