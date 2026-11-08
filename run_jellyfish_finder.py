import sys
import configparser
from pathlib import Path
import os
import urllib.request
import ssl

from torchvision import transforms
import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
import pandas as pd
import shutil
import time
import sep
from astropy.io import fits
from astropy.table import vstack
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astroquery.mast import Observations
from astropy.table import Table
from astropy.coordinates import SkyCoord
from skimage import measure
from regions import PolygonSkyRegion
from PIL import Image


#Define repository paths
BASE_DIR = Path(__file__).resolve().parent

LOCAL_FITS_DIR = BASE_DIR / "local_fits"
WEIGHTS_DIR = BASE_DIR / "weights"

TEMP_DOWNLOADS_DIR = BASE_DIR / "temp_downloads"
OUTPUT_DIR = BASE_DIR / "output_candidates"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

valid_filters = (
    # --- JWST NIRCam Filters ---
    "F070W", "F090W", "F115W", "F150W", "F200W", "F140M", "F162M", "F182M", 
    "F210M", "F164N", "F187N", "F212N", "F277W", "F356W", "F444W", "F322W2", 
    "F444W2", "F250M", "F300M", "F335M", "F360M", "F410M", "F430M", "F460M", 
    "F480M", "F323N", "F405N", "F466N", "F470N",
    
    # --- JWST MIRI Filters ---
    "F560W", "F770W", "F1000W", "F1130W", "F1280W", "F1500W", "F1800W", "F2100W", 
    "F2550W", "F1065C", "F1140C", "F1550C", "F2300C",
    
    # --- JWST NIRISS Filters (Unique to NIRISS) ---
    "F158M", "F380M",
    
    # --- Hubble WFC3 UVIS & IR Filters ---
    "F218W", "F225W", "F275W", "F336W", "F390W", "F438W", "F475W", "F555W", 
    "F606W", "F625W", "F775W", "F814W", "F850LP", "F390M", "F410M", "F467M", 
    "F547M", "F689M", "F763M", "F845M", "F280N", "F343N", "F373N", "F395N", 
    "F469N", "F487N", "F502N", "F631N", "F645N", "F656N", "F657N", "F658N", 
    "F660N", "F673N", "F680N", "F953N", "F105W", "F110W", "F125W", "F140W", 
    "F160W", "F098M", "F127M", "F139M", "F153M", "F126N", "F128N", "F130N", 
    "F167N",
    
    # --- Hubble ACS Filters (Unique to ACS) ---
    "F435W", "F550M", "F892N", "F115LP", "F122LP", "F125LP", "F140LP", "F150LP", 
    "F165LP", "F122M",
    
    # --- Hubble STIS Filters ---
    "50CCD", "F28X50LP", "F28X50OIII", "F28X50OII", "F25SRF2", "F25QTZ", "F25CN270")


def get_config_val(cfg, section, key, fallback=None, cast_type=str, is_list=False):

    # If the specified key doesn't exist, an empty string will be returned
    val = cfg.get(section, key, fallback="").strip()

    # Fallback to [DEFAULT] section if blank
    if not val and section != "DEFAULT":
        val = cfg.get("DEFAULT", key, fallback="").strip()

    # Fallback to hardcoded Python default if still blank
    if not val:
        if fallback is None:
            return None
        return fallback

    # Turns filter list parameter into Python list
    if is_list:
        return [x.strip() for x in val.split(",") if x.strip()]

    # Sets final variable type
    if cast_type == bool:
        val_clean = val.lower()
        if val_clean in ("true", "1", "yes", "on"):
            return True
        elif val_clean in ("false", "0", "no", "off"):
            return False
        else:
            print(f"[ERROR] Invalid boolean value '{val}' for setting '{key}' in section [{section}]. Expected True/False, 1/0, Yes/No, or On/Off.")
            sys.exit(1)

    elif cast_type == float:
        return float(val)
    elif cast_type == int:
        return int(val)

    return val


def load_config(config_path):
    config_file = Path(config_path)

    # Checks if path given is absolute. If not,
    # it joins the relative path to the BASE_DIR to check there
    if not config_file.is_absolute():
        config_file = (BASE_DIR / config_file).resolve()

    # If the config file can't be found, user gets an error
    if not config_file.exists():
        print(f"[ERROR] Config file not found at: {config_file}")
        sys.exit(1)

    cfg = configparser.ConfigParser(
        delimiters=('=', ':'),
        inline_comment_prefixes=('#')
    )

    cfg.read(config_file)

    # Get selected mode from config file
    raw_mode = get_config_val(cfg, "MODE", "mode")
    if not raw_mode:
        print("[ERROR] The 'mode' key under [MODE] cannot be empty. Options: 'obs_id', 'name_coords', or 'local'.")
        sys.exit(1)

    if raw_mode.lower().replace("-", "_") in (('obs_id', 'name_coords', 'local')):
        mode = raw_mode.lower().replace("-", "_")
    else:
        print("[ERROR] Invalid Value for the 'mode' key under [MODE] section. Options: 'obs_id', 'name_coords', or 'local'.")
        sys.exit(1)


    # Get the weights path from config file
    weights_str = get_config_val(cfg, "MODEL", "weights_file")
    if not weights_str:
        print("[ERROR] No path given for 'weights_path' under [MODEL] section. This is required.")
        sys.exit(1)

    weights_path = Path(weights_str)
    if not weights_path.is_absolute():

        if weights_path.parts and weights_path.parts[0] == "weights":
            weights_path = (BASE_DIR / weights_path).resolve()
        else:
            weights_path = (WEIGHTS_DIR / weights_path).resolve()

    if not weights_path.exists():
        print(f"[ERROR] Weights file cannot be found at: {weights_path}")
        sys.exit(1)

    # Get the minimum percent confidence from config file
    percent_conf = get_config_val(cfg, "MODEL", "percent_conf", fallback=0.6, cast_type=float)
    if percent_conf < 0.0 or percent_conf > 1.0:
        print("[ERROR] Invalid value given for 'percent_conf' under [MODEL] section. Please enter a value between 0 and 1, or leave blank for the default value of 0.6 (60%).")
        sys.exit(1)

    #Create a set of parameters to add to depending on mode
    params = {
        "mode": mode,
        "weights_path": weights_path,
        "percent_conf": percent_conf
    }


    # --- OBS_ID MODE --- #
    if params["mode"] == "obs_id":

        # Get CSV path containing cluster_name, observation_ID
        csv_str = get_config_val(cfg, "OBS_ID_MODE", "csv_path")
        if not csv_str:
            print("[ERROR] No path given for 'csv_path' under [OBS_ID_MODE] section, which is required.")
            sys.exit(1)

        csv_path = Path(csv_str)
        if not csv_path.is_absolute():
            csv_path = (BASE_DIR / csv_path).resolve()

        if not csv_path.exists():
            print(f"[ERROR] CSV file cannot be found at: {csv_path}")
            sys.exit(1)

        params["csv_path"] = csv_path


    # --- LOCAL MODE --- #
    elif params["mode"] == "local":

        # Get folder path containing local fits files (Checks later that only fits files are used)
        data_path_str = get_config_val(cfg, "LOCAL_MODE", "data_dir")

        if data_path_str:
            local_dir = Path(data_path_str)
            if not local_dir.is_absolute():
                local_dir = (BASE_DIR / local_dir).resolve()
            if not local_dir.exists() or not local_dir.is_dir():
                print(f"[ERROR] Data folder path given for 'data_dir' under [LOCAL_MODE] section: {local_dir} does not exist. Fix the folder path or leave blank to use the default './local_fits/' folder")
                sys.exit(1)

            valid_ext = (".fits", ".fit", ".fits.gz")
            fits_files = [
                f for f in local_dir.iterdir() if f.is_file() and f.name.lower().endswith(valid_ext)
            ]
            if not fits_files:
                print(f"[ERROR] Data directory was found, but no FITS files exist within the folder {local_dir}.")
                sys.exit(1)

        else:
            local_dir = LOCAL_FITS_DIR
            if not local_dir.exists() or not local_dir.is_dir():
                print("[ERROR] Under mode 'local' no data path was given. Tried to fallback to './local_fits/', but the folder no longer exists.")
                sys.exit(1)

            valid_ext = (".fits", ".fit", ".fits.gz")
            fits_files = [
                f for f in local_dir.iterdir() if f.is_file() and f.name.lower().endswith(valid_ext)
            ]
            if not fits_files:
                print(f"[ERROR] Data directory was found, but no FITS files exist within the folder {local_dir}.")
                sys.exit(1)

        params["local_data_dir"] = local_dir
        params["file_names"] = fits_files
        print(f"[INFO] Under 'local' mode: Found {len(fits_files)} FITS file(s) within {local_dir} to process")


    # --- NAME_COORDS MODE --- #
    elif params["mode"] == "name_coords":        

        #Get CSV path containing cluster_name, RA, DEC, radius
        csv_str = get_config_val(cfg, "NAME_COORDS_MODE", "csv_path")
        if not csv_str:
            print("[ERROR] No path given for 'csv_path' under [NAME_COORDS_MODE] section, which is required.")
            sys.exit(1)

        csv_path = Path(csv_str)
        if not csv_path.is_absolute():
            csv_path = (BASE_DIR / csv_path).resolve()

        if not csv_path.exists():
            print(f"[ERROR] CSV file cannot be found at: {csv_path}")
            sys.exit(1)

        params["csv_path"] = csv_path

        #Get Filter List Preference
        filter_list = get_config_val(cfg, "NAME_COORDS_MODE", "filt_list", fallback=[], is_list=True)
        if filter_list:
            invalid_filters = []
            keep_filters = []
            for filter in filter_list:
                if filter.upper() not in valid_filters:
                    invalid_filters.append(filter)
                else:
                    keep_filters.append(filter.upper())

            if not keep_filters:
                print("[ERROR] No valid filters given for 'filt_list' under [NAME_COORDS_MODE] section.")
                print("Please pick filters from the following:")
                print(valid_filters)
                sys.exit(1)

            elif invalid_filters:
                print(f"[WARNING] Invalid filters given in 'filt_list' under [NAME_COORDS_MODE] section. The following filters will be ignored: {invalid_filters}")
                print(f"The new filter list in order of priority is {keep_filters}")

            else:
                print(f"Using given prioritized filter list: {keep_filters}")

            filter_list = keep_filters
            params["filt_list"] = filter_list

        else:
            params["filt_list"] = []

        #Get minimum data product level
        prod_level = get_config_val(cfg, "NAME_COORDS_MODE", "min_prod_level", fallback=3, cast_type=int)
        if prod_level not in ((1, 2, 3)):
            print("[ERROR] Invalid value given for 'min_prod_level' under [NAME_COORDS_MODE] section. Please choose from (1, 2, 3) or leave blank for default level 3.")
            sys.exit(1)

        params["min_prod_level"] = prod_level

        #Get MAST Fallback Radius
        fallback_radius = get_config_val(cfg, "NAME_COORDS_MODE", "fallback_radius", fallback=10.0, cast_type=float)
        if fallback_radius > 30.0 or fallback_radius < 0.0:
            print(f"[ERROR] Invalid fallback radius given: {fallback_radius} arcmin. Valid range: 0-30. Please enter a valid radius or leave empty to use the default radius of 10 arcmin.")
            sys.exit(1)

        params["fallback_radius"] = fallback_radius

        #Get preference of JWST over HST
        JWST_pref = get_config_val(cfg, "NAME_COORDS_MODE", "JWST_over_HST", fallback=True, cast_type=bool)
        params["jwst_over_hst"] = JWST_pref

        #Get Maximum Allowed Overlap between Pointings
        max_overlap = get_config_val(cfg, "NAME_COORDS_MODE", "max_overlap", fallback=0.25, cast_type=float)
        if max_overlap < 0.00 or max_overlap > 1.0:
            print(f"[ERROR] Invalid value of ({max_overlap}) given to 'max_overlap' under [NAME_COORDS_MODE] section. Please enter a value between 0 and 1, or leave empty to use the default value of 0.25 (25%).")
            sys.exit(1)

        params["max_overlap"] = max_overlap

    return params


def get_transforms():

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    return transforms.Compose([
        transforms.Resize((518, 518)),
        transforms.ToTensor(),
        normalize,
    ])


def get_device():
    """
    Priority order:
    1. CUDA (NVIDIA GPU)
    2. MPS (Apple Silicon)
    3. CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


class DINOv2ForClassification(nn.Module):

    def __init__(self, encoder, num_classes=2):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.LayerNorm(encoder.embed_dim),
            nn.Linear(encoder.embed_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, num_classes),
        )

        # Initialize all Linear layers
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        """Forward pass through the DINOv2 encoder and classification head."""
        features = self.encoder(x)
        if isinstance(features, tuple):  # handle DINOv2 returning (features, cls_tokens)
            features = features[0]
        logits = self.classifier(features)
        return logits


def load_pretrained_dinov2(device):
    """
    Loads the pretrained DINOv2 ViT-S/14 encoder from Facebook Research hub.
    """
    #Changed encoder to pull the register tokens version instead of "dinov2_vits14"
    encoder = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg").to(device)
    return encoder


def load_model(weights_path, device):

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    encoder = load_pretrained_dinov2(device)
    model = DINOv2ForClassification(encoder, num_classes=2).to(device)

    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced binary classification,
    as used for retraining on imbalanced or harder data.
    Alpha balances class weights, and gamma focuses on harder examples.

    Parameters
    ----------
    alpha : list[float]
        Weighting factors for each class [NoRPS, RPS].
    gamma : float
        Focusing parameter that reduces the loss for easy examples.
    reduction : str
        Specifies reduction ('mean' or 'sum').
    """

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        if alpha is None:
            alpha = [1.0, 1.0]
        self.alpha = torch.tensor(alpha, dtype=torch.float32)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        if inputs.ndim > 1 and inputs.size(1) > 1:
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        else:
            ce_loss = F.binary_cross_entropy_with_logits(inputs.squeeze(), targets.float(), reduction='none')

        pt = torch.exp(-ce_loss)
        alpha_t = self.alpha.to(inputs.device)[targets]
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


def calculate_circle_overlap(coord1, r1_deg, coord2, r2_deg):
    d = coord1.separation(coord2).deg

    # No overlap
    if d >= (r1_deg + r2_deg):
        return 0.0

    # One pointing completely inside the other
    if d <= abs(r1_deg - r2_deg):
        return 1.0

    # Partial overlap
    r1, r2 = r1_deg, r2_deg
    r1_sq, r2_sq = r1**2, r2**2

    # Clip values to [-1.0, 1.0] to prevent floating-point arccos NaN
    val1 = np.clip((d**2 + r1_sq - r2_sq) / (2 * d * r1), -1.0, 1.0)
    val2 = np.clip((d**2 + r2_sq - r1_sq) / (2 * d * r2), -1.0, 1.0)

    part1 = r1_sq * np.arccos(val1)
    part2 = r2_sq * np.arccos(val2)
    
    arg3 = (-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2)
    part3 = 0.5 * np.sqrt(max(0.0, arg3))

    intersection_area = part1 + part2 - part3
    min_area = np.pi * min(r1, r2)**2

    return float(intersection_area / min_area)


def get_obs_priority(row, params):
    mission = str(row['obs_collection']).strip().upper()

    # Rule 1: Mission Priority (JWST vs HST)
    if params["jwst_over_hst"]:
        mission_rank = 0 if mission == "JWST" else 1
    else:
        mission_rank = 0

    # Rule 2: Filter Priority
    obs_filt = ""
    if 'filters' in row.colnames:
        obs_filt = str(row['filters']).strip().upper()
    elif 'filter' in row.colnames:
        obs_filt = str(row['filter']).strip().upper()

    filt_rank = 999
    if params["filt_list"]:
        for idx, target_filt in enumerate(params["filt_list"]):
            if target_filt.upper() in obs_filt:
                filt_rank = idx
                break

    # Rule 3: Data Level Rank
    level_rank = 0
    if 'calib_level' in row.colnames and row['calib_level'] is not None:
        try:
            level_rank = -int(row['calib_level'])
        except (ValueError, TypeError):
            level_rank = 0

    return (mission_rank, filt_rank, level_rank)


# Inference Function
def modified_predict(img, device, model, transform):

    image = img
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]

    return {
        "NoRPS": float(probs[0]),
        "RPS": float(probs[1])
    }

# ----------------------------------------------------------------------------
# Main Block
def main():

    #Prepare Encoder Request
    ssl_context = ssl._create_unverified_context()
    opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_context))
    urllib.request.install_opener(opener)

    # Get config file path
    if len(sys.argv) > 2:
        print(f"[ERROR] Too many arguments given when trying to run 'run_jellyfish_finder.py'. Expected 1 (default config path and name) or 2 (custom config path or name), but received {len(sys.argv)}.")
        sys.exit(1)

    if len(sys.argv) > 1:                           # Custom name or path for config file
        config_path = sys.argv[1]
    else:                                           # Default name and path for config file
        config_path = BASE_DIR / "config.ini"

    # Return Parameters from Config File
    params = load_config(config_path)

    # Load Weights and Model
    weights_path = params["weights_path"]
    device = get_device()                           # Need to include in the modified_predict function arguments
    model = load_model(weights_path, device)        # Need to include in the modified_predict function arguments
    transform = get_transforms()                    # Need to include in the modified_predict function arguments

    # Set Source Extractor Limit
    sep.set_extract_pixstack(10_000_000)


    # Mode Specific Pipelines

    # OBS_ID MODE ------------------------------------------------------------
    if params["mode"] == "obs_id":
        print(f"[INFO] Running OBS_ID mode using CSV: {params['csv_path']}")

        obs = Table.read(params["csv_path"], format='csv')

        #Create dictionary to hold a list for each cluster - {cluster1:[obs1, obs2], cluster2:[obs1, obs2]}
        cluster_observations = {}
        unique_targets = np.unique(obs['cluster_name'], return_counts=True)
        for cluster in unique_targets[0]:
            obs_list = []
            for row in obs:
                if row['cluster_name'] == cluster:
                    obs_list.append(row['obs_id'])

            cluster_observations[cluster] = obs_list

        count = 1
        #Create outside loop to go through each cluster
        for cluster in cluster_observations:
            print(f"Cluster {count}/{len(cluster_observations)}")
            count += 1

            #Create list of masked Sky Regions for clusters with multiple observations
            sky_regions = []

            #Delete data from temp_downloads after each cluster
            folder_path = TEMP_DOWNLOADS_DIR
            if os.path.exists(folder_path) and os.path.isdir(folder_path):
                shutil.rmtree(folder_path)  # Deletes the folder and everything inside it
                os.makedirs(folder_path)    #Create empty folder
            else:
                os.makedirs(folder_path)    #Create empty folder

            #Create inner loop to go through each observation in that cluster
            for i, observation in enumerate(cluster_observations[cluster]):
            
                #Make list to hold temporary image specific pixel regions
                pixel_regions = []

                #Download data with MAST from observation ID
                attempts = 3
                product = None

                print("Searching for product from observation ID and Filtering...")
                start_time = time.perf_counter()
                for attempt in range(attempts):   

                    #Try a max of 3 times before skipping observation
                    try:
                        obs_table = Observations.query_criteria(obs_id=observation)
                        product = Observations.get_product_list(obs_table)
                        break
                    except Exception as e:
                        if attempt < attempts - 1:
                            time.sleep(10)
                        else:
                            print(f"Skipping Observation {observation} after {str(attempts)} failed attempts")

                if product is None:
                    continue
                end_time = time.perf_counter()
                print(f"Astroquery found product in {end_time - start_time:.2f} seconds")

                #Filter to single observation
                time_start = time.perf_counter()
                product = product[product['obs_id'] == observation]
                product = product[product['productType'] == 'SCIENCE']

                row = None
                for j, prod in enumerate(product):
                    if ".fits" in prod['productFilename']:
                        row = j
                        break

                if row is None:
                    continue

                product = product[row:row+1]
                time_end = time.perf_counter()
                print(f"Filtered to single observation in {time_end - time_start:.2f} seconds")


                time_start = time.perf_counter()
                manifest = None
                try:
                    Observations.enable_cloud_dataset()
                    manifest = Observations.download_products(product, download_dir='temp_downloads/', flat=True, cache=True)
                except Exception as e:
                    print("AWS Unavailable, falling back to MAST")
                    pass

                if manifest is None:
                    try:
                        Observations.disable_cloud_dataset()
                        manifest = Observations.download_products(product, download_dir='temp_downloads/', flat=True, cache=True)
                    except Exception as e:
                        print(f"Failed to download {observation}: {e}")

                if manifest is None:
                    print(f"Skipping observation {observation}: Failed to download")
                    continue
                    
                time_end = time.perf_counter()
                print(f"Downloaded Cluster, took {time_end - time_start:.2f} seconds")

                #Open Downloaded Data and get WCS
                time_start = time.perf_counter()
                path = manifest['Local Path'][0]

                with fits.open(path) as image:
                    sci_hdu = image['SCI'] if 'SCI' in image else image[0]
                    data = sci_hdu.data.copy()
                    header = sci_hdu.header.copy()
                
                wcs = WCS(header)
                time_end = time.perf_counter()
                print(f"Finished Opening Data, took {time_end - time_start:.2f} seconds")

                #First turn sky regions into pixel masks for the current image
                time_start = time.perf_counter()
                if len(sky_regions) == 0:
                    final_mask = None
                else:
                    final_mask = np.zeros(data.shape, dtype=int)
                    print("Number of masks made from previous image:", len(sky_regions))
                    print(f"Made zeros mask in shape of data, took {time.perf_counter() - time_start} seconds")
                    for reg in sky_regions:
                        pixel_reg = reg.to_pixel(wcs)
                        local_mask = pixel_reg.to_mask()
                        if local_mask is None:
                            continue
                    
                        mask_array = local_mask.to_image(shape=data.shape)
                        if mask_array is None:
                            continue
                        else:
                            binary_mask = np.where(mask_array > 0, 1, 0)
                            final_mask |= binary_mask
                            
                time_end = time.perf_counter()
                print(f"Finished Applying Sky Masks, took {time_end - time_start:.2f} seconds")

                #Do source detection with mask included in the segmentation function
                time_start = time.perf_counter()
                data = data.astype(np.float32, copy=False)
                if not data.flags['C_CONTIGUOUS']:
                    data = np.ascontiguousarray(data)

                bkg = sep.Background(data, bw=200, bh=200, fw=3, fh=3)
                data_subtracted = data - bkg

                sigma = 4.0
                objects = sep.extract(data_subtracted, thresh=sigma, err=bkg.globalrms, minarea=150, mask=final_mask) #Was 300
                print(f"Number of Sources: {len(objects)}")

                #Filter out stars with saturated cores
                #objects = objects[objects['peak'] <= 35]

                #Filter out sources with too small of size based on bounding box size
                widths = objects['xmax'] - objects['xmin']
                heights = objects['ymax'] - objects['ymin']
                size_squared = (widths**2) + (heights**2)

                filtered_objects = objects[size_squared >= 25.0**2]
                print(f"Number of filtered sources: {len(filtered_objects)}")

                time_end = time.perf_counter()
                print(f"Finished Source Detection and Filtering, took {time_end - time_start:.2f} seconds")

                #Create a 2D cutot for each source
                scale_factor = 2.5
                for obj in filtered_objects:
                    width = obj['xmax'] - obj['xmin']
                    height = obj['ymax'] - obj['ymin']
                    length = np.sqrt((width)**2 + (height)**2)
                    scaled_length = length * scale_factor

                    center_x = obj['x']
                    center_y = obj['y']

                    coord = SkyCoord.from_pixel(center_x, center_y, wcs = wcs, origin=0)

                    cutout = Cutout2D(data, coord, scaled_length, wcs=wcs, mode='trim')
                    cutout_data = cutout.data

                    clipped_data = np.clip(cutout_data, np.nanpercentile(cutout_data, 0.25), np.nanpercentile(cutout_data, 99.75))
                    clipped_data -= np.nanmin(clipped_data)
                    log_stretched = np.log1p(15 * clipped_data)
                    
                    max_val = np.nanmax(log_stretched)
                    if max_val > 0:
                        scaled_data = (log_stretched / max_val * 255)
                        clean_data = np.nan_to_num(scaled_data, nan=0.0, posinf=255.0, neginf=0.0)
                        final_image_array = clean_data.astype(np.uint8)
                    else:
                        final_image_array = np.zeros(log_stretched.shape, dtype=np.uint8)

                    img = Image.fromarray(final_image_array, mode='L')
                    img_RGB = img.convert("RGB")

                    #Run Inference on cutout image
                    result = None
                    attempts = 3
                    for attempt in range(attempts):
                        try:
                            result = modified_predict(img_RGB, device, model, transform)
                            break
                        except Exception as e:
                            if attempt < attempts - 1:
                                time.sleep(5)

                    if result is None:
                        img.close()
                        img_RGB.close()
                        continue

                    try:
                        prob = result['RPS'] * 100
                        percent_jellyfish = np.round(prob, 2)

                        #Give final file name and save if score > 60%
                        if percent_jellyfish >= (100 * params["percent_conf"]):
                            filename = f"Score-{percent_jellyfish}%__{cluster}__{observation}__RA-{coord.ra.value:.3f}-DEC-{coord.dec.value:.3f}.jpg"
                            output_file_path = OUTPUT_DIR / filename
                            img.save((output_file_path), "JPEG", quality=100)

                    finally:
                        #Close Images to free up memory
                        img.close()
                        img_RGB.close()
            

                #Add to masked region only if there are more than one observations to save computing time. Added at the end
                if len(cluster_observations[cluster]) > 1 and len(cluster_observations[cluster]) - (i + 1) != 0:
                    
                    valid_space = ~np.isnan(data) & (data != 0)
                    data_mask = np.where(valid_space, 1, 0)
                    contours = measure.find_contours(data_mask, level=0.5)
                    contours = sorted(contours, key=len, reverse=True)

                    if len(contours) > 0:
                        #Save only the two largest contours to cover multiple chips
                        contour1 = contours[0]
                        contour2 = contours[1]

                        #Reduce the number of polygon sides using the Ramer-Douglas-Peucker algorithm
                        reduced_contour1 = measure.approximate_polygon(contour1, tolerance=20.0)
                        reduced_contour2 = measure.approximate_polygon(contour2, tolerance=20.0)

                        x_pixels1 = reduced_contour1[:, 1]
                        y_pixels1 = reduced_contour1[:, 0]

                        x_pixels2 = reduced_contour2[:, 1]
                        y_pixels2 = reduced_contour2[:, 0]

                        sky_coords1 = wcs.pixel_to_world(x_pixels1, y_pixels1)
                        poly_region1 = PolygonSkyRegion(vertices=sky_coords1)

                        sky_coords2 = wcs.pixel_to_world(x_pixels2, y_pixels2)
                        poly_region2 = PolygonSkyRegion(vertices=sky_coords2)

                        sky_regions.append(poly_region1)
                        sky_regions.append(poly_region2)


    elif params["mode"] == "local":
        print(f"[INFO] Running LOCAL mode on {len(params['file_names'])} fits files.")

        folder_path = TEMP_DOWNLOADS_DIR
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            shutil.rmtree(folder_path)  # Deletes the folder and everything inside it
            os.makedirs(folder_path)    #Create empty folder
        else:
            os.makedirs(folder_path)    #Create empty folder

        for file in params["file_names"]:

            #Open Downloaded Data and get WCS
            time_start = time.perf_counter()
            with fits.open(file) as image:
                sci_hdu = image['SCI'] if 'SCI' in image else image[0]
                data = sci_hdu.data.copy()
                header = sci_hdu.header.copy()
            
            wcs = WCS(header)
            time_end = time.perf_counter()
            print(f"Finished Opening Data, took {time_end - time_start:.2f} seconds")


            #Do source detection with mask included in the segmentation function
            time_start = time.perf_counter()
            data = data.astype(np.float32, copy=False)
            if not data.flags['C_CONTIGUOUS']:
                data = np.ascontiguousarray(data)

            bkg = sep.Background(data, bw=200, bh=200, fw=3, fh=3)
            data_subtracted = data - bkg

            sigma = 4.0
            objects = sep.extract(data_subtracted, thresh=sigma, err=bkg.globalrms, minarea=150) #Was 300
            print(f"Number of Sources: {len(objects)}")

            #Filter out stars with saturated cores
            #objects = objects[objects['peak'] <= 35]

            #Filter out sources with too small of size based on bounding box size
            widths = objects['xmax'] - objects['xmin']
            heights = objects['ymax'] - objects['ymin']
            size_squared = (widths**2) + (heights**2)

            filtered_objects = objects[size_squared >= 25.0**2]
            print(f"Number of filtered sources: {len(filtered_objects)}")

            time_end = time.perf_counter()
            print(f"Finished Source Detection and Filtering, took {time_end - time_start:.2f} seconds")

            #Create a 2D cutot for each source
            scale_factor = 2.5
            for obj in filtered_objects:
                width = obj['xmax'] - obj['xmin']
                height = obj['ymax'] - obj['ymin']
                length = np.sqrt((width)**2 + (height)**2)
                scaled_length = length * scale_factor

                center_x = obj['x']
                center_y = obj['y']

                coord = SkyCoord.from_pixel(center_x, center_y, wcs = wcs, origin=0)

                cutout = Cutout2D(data, coord, scaled_length, wcs=wcs, mode='trim')
                cutout_data = cutout.data

                clipped_data = np.clip(cutout_data, np.nanpercentile(cutout_data, 0.25), np.nanpercentile(cutout_data, 99.75))
                clipped_data -= np.nanmin(clipped_data)
                log_stretched = np.log1p(15 * clipped_data)
                
                max_val = np.nanmax(log_stretched)
                if max_val > 0:
                    scaled_data = (log_stretched / max_val * 255)
                    clean_data = np.nan_to_num(scaled_data, nan=0.0, posinf=255.0, neginf=0.0)
                    final_image_array = clean_data.astype(np.uint8)
                else:
                    final_image_array = np.zeros(log_stretched.shape, dtype=np.uint8)

                img = Image.fromarray(final_image_array, mode='L')
                img_RGB = img.convert("RGB")

                #Run Inference on cutout image
                result = None
                attempts = 3
                for attempt in range(attempts):
                    try:
                        result = modified_predict(img_RGB, device, model, transform)
                        break
                    except Exception as e:
                        if attempt < attempts - 1:
                            time.sleep(5)

                if result is None:
                    img.close()
                    img_RGB.close()
                    continue

                try:
                    prob = result['RPS'] * 100
                    percent_jellyfish = np.round(prob, 2)

                    #Give final file name and save if score > user given percentage
                    if percent_jellyfish >= (100 * params["percent_conf"]):
                        file_stem = file.name

                        for ext in (".fits.gz", ".fits", ".fit"):
                            if file_stem.lower().endswith(ext):
                                file_stem = file_stem[:-len(ext)]
                                break

                        filename = f"Score-{percent_jellyfish}%__{file_stem}__RA-{coord.ra.value:.3f}-DEC-{coord.dec.value:.3f}.jpg"
                        output_file_path = OUTPUT_DIR / filename
                        img.save((output_file_path), "JPEG", quality=100)

                finally:
                    #Close Images to free up memory
                    img.close()
                    img_RGB.close()



    elif params["mode"] == "name_coords":
        print(f"[INFO] Running NAME_COORDS mode.")

        obs = pd.read_csv(params["csv_path"])
        # Make outer loop to go through each cluster in the csv file
        for index, row in obs.iterrows():

            # Get cluster name from CSV
            if pd.isna(row["cluster_name"]):
                has_name = False
                name = None
            else:
                name = str(row["cluster_name"]).strip()
                has_name = True

            # Get RA and DEC from CSV
            if pd.isna(row["RA"]) or pd.isna(row["DEC"]):
                has_coords = False
                ra = None
                dec = None
            else:
                ra = float(row["RA"])
                dec = float(row["DEC"])
                has_coords = True

            # Get search radius from CSV
            if pd.isna(row["radius"]):
                radius = params["fallback_radius"]
                print(f"[INFO] No radius given for row {index + 1}. Using default value of {params['fallback_radius']} arcmin.")
            else:
                radius = float(row["radius"])

            radius = str(radius) + " arcmin"
            

            if not has_name and not has_coords:
                print(f"[WARNING] Skipping row {index + 1}. Both cluster name and coordinates are missing... Skipping Row.")
                continue

            #Script for name only
            if has_name and not has_coords:
                print(f"[ROW {index + 1}]: Name only -> Searching MAST with cluster name.")
                
                try:
                    obs_table = Observations.query_object(name, radius=radius)
                except Exception as e:
                    print(f"[ERROR] Target name search failed for '{name}' on Row {index + 1}: {e}")
                    continue

                obs_collection_upper = np.char.upper(obs_table['obs_collection'].astype(str))
                allowed_missions_mask = np.isin(obs_collection_upper, ['JWST', 'HST'])
                obs_table = obs_table[allowed_missions_mask]

                if len(obs_table) == 0:
                    print(f"[WARNING] No JWST or HST observations returned for the cluster: {name}... Skipping.")
                    continue

                if 'calib_level' in obs_table.colnames:
                    obs_table = obs_table[obs_table['calib_level'] >= params["min_prod_level"]]

                if len(obs_table) == 0:
                    print(f"[WARNING] No observations met min_prod_level >= {params['min_prod_level']}. Skipping.")
                    continue


                # Sort observations by priority (JWST_over_HST, then filt_list order)
                sorted_indices = sorted(range(len(obs_table)), key=lambda idx: get_obs_priority(obs_table[idx], params))
                sorted_obs_table = obs_table[sorted_indices]

                # Check for overlaps
                kept_observations = []

                for candidate in sorted_obs_table:
                    c_ra = float(candidate['s_ra'])
                    c_dec = float(candidate['s_dec'])
                    c_coord = SkyCoord(ra=c_ra, dec=c_dec, unit='deg')

                    # Estimate FOV radius in degrees
                    c_fov = candidate['fov'] if 'fov' in candidate.colnames else None
                    c_radius = (float(c_fov) / 2.0) if (c_fov is not None and not np.isnan(float(c_fov)) and float(c_fov) > 0) else 0.03

                    is_overlapping = False
                    for kept in kept_observations:
                        k_coord = kept['coord']
                        k_radius = kept['radius']

                        overlap_fraction = calculate_circle_overlap(c_coord, c_radius, k_coord, k_radius)

                        if overlap_fraction > params["max_overlap"]:
                            is_overlapping = True
                            break  # Overlaps higher-priority accepted pointing; reject candidate

                    if not is_overlapping:
                        kept_observations.append({
                            'row': candidate,
                            'coord': c_coord,
                            'radius': c_radius
                        })

                if len(kept_observations) == 0:
                    print(f"[WARNING] No observations remained after overlap filtering for Row {index + 1}. Skipping.")
                    continue


                # Rebuild table of non-overlapping, high-priority observations
                filtered_obs_table = vstack([item['row'] for item in kept_observations])

                try:
                    products = Observations.get_product_list(filtered_obs_table)
                except Exception as e:
                    print(f"[ERROR] Failed to fetch product list for Row {index + 1}: {e}")
                    continue

                if products is None or len(products) == 0:
                    print(f"[WARNING] No data products returned for Row {index + 1}. Skipping.")
                    continue


                # Filter products by image type, calibration level, and file extension
                products = products[np.char.upper(products['productType'].astype(str)) == 'SCIENCE']

                if 'calib_level' in products.colnames:
                    products = products[products['calib_level'] >= params["min_prod_level"]]

                valid_instruments = ["NIRCAM", "MIRI", "NIRISS", "WFC3", "ACS"] # Excluded "STIS" and "WFPC2"
                if 'instrument_name' in products.colnames:
                    prod_inst = np.char.upper(products['instrument_name'].astype(str))
                    inst_mask = np.zeros(len(products), dtype=bool)
                    
                    for inst in valid_instruments:
                        inst_mask |= (np.char.find(prod_inst, inst) >= 0)

                    if np.any(inst_mask):
                        products = products[inst_mask]
                    else:
                        print(f"[WARNING] No products matched target instruments: {valid_instruments}. Preserving available science products.")

                fn_upper = np.char.upper(products['productFilename'].astype(str))
                valid_ext_mask = np.char.endswith(fn_upper, '.FITS') | np.char.endswith(fn_upper, '.FITS.GZ') | np.char.endswith(fn_upper, '.FIT')
                products = products[valid_ext_mask]

                if len(products) == 0:
                    print(f"[WARNING] No valid FITS science products remain for Row {index + 1} after product filtering. Skipping.")
                    continue

                # Download Data Products
                if TEMP_DOWNLOADS_DIR.exists():
                    shutil.rmtree(TEMP_DOWNLOADS_DIR)
                TEMP_DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)

                print(f"[INFO] Downloading {len(products)} FITS science product(s)...")
                manifest = None
                try:
                    Observations.enable_cloud_dataset()
                    manifest = Observations.download_products(products, download_dir=str(TEMP_DOWNLOADS_DIR), flat=True, cache=True)
                except Exception:
                    pass

                if manifest is None:
                    try:
                        Observations.disable_cloud_dataset()
                        manifest = Observations.download_products(products, download_dir=str(TEMP_DOWNLOADS_DIR), flat=True, cache=True)
                    except Exception as e:
                        print(f"[ERROR] Failed to download products for Row {index + 1}: {e}")
                        continue

                fits_paths = [Path(p) for p in manifest['Local Path'] if Path(p).exists()]
                print(f"[INFO] Processing {len(fits_paths)} downloaded FITS file(s)...")

                for fits_file in fits_paths:
                    time_start = time.perf_counter()
                    with fits.open(fits_file) as image:
                        sci_hdu = image['SCI'] if 'SCI' in image else image[0]
                        data = sci_hdu.data.copy()
                        header = sci_hdu.header.copy()

                    wcs = WCS(header)

                    # Source extraction setup
                    data = data.astype(np.float32, copy=False)
                    if not data.flags['C_CONTIGUOUS']:
                        data = np.ascontiguousarray(data)

                    bkg = sep.Background(data, bw=200, bh=200, fw=3, fh=3)
                    data_subtracted = data - bkg

                    sigma = 4.0
                    objects = sep.extract(data_subtracted, thresh=sigma, err=bkg.globalrms, minarea=150)

                    # Filter sources by size
                    widths = objects['xmax'] - objects['xmin']
                    heights = objects['ymax'] - objects['ymin']
                    size_squared = (widths**2) + (heights**2)

                    filtered_objects = objects[size_squared >= 25.0**2]
                    print(f"[INFO] {fits_file.name}: Extracted {len(filtered_objects)} candidate sources.")

                    # Generate cutouts and run inference
                    scale_factor = 2.5
                    for obj in filtered_objects:
                        width = obj['xmax'] - obj['xmin']
                        height = obj['ymax'] - obj['ymin']
                        length = np.sqrt(width**2 + height**2)
                        scaled_length = length * scale_factor

                        center_x = obj['x']
                        center_y = obj['y']

                        coord = SkyCoord.from_pixel(center_x, center_y, wcs=wcs, origin=0)

                        cutout = Cutout2D(data, coord, scaled_length, wcs=wcs, mode='trim')
                        cutout_data = cutout.data

                        clipped_data = np.clip(cutout_data, np.nanpercentile(cutout_data, 0.25), np.nanpercentile(cutout_data, 99.75))
                        clipped_data -= np.nanmin(clipped_data)
                        log_stretched = np.log1p(15 * clipped_data)

                        max_val = np.nanmax(log_stretched)
                        if max_val > 0:
                            scaled_data = (log_stretched / max_val * 255)
                            clean_data = np.nan_to_num(scaled_data, nan=0.0, posinf=255.0, neginf=0.0)
                            final_image_array = clean_data.astype(np.uint8)
                        else:
                            final_image_array = np.zeros(log_stretched.shape, dtype=np.uint8)

                        img = Image.fromarray(final_image_array, mode='L')
                        img_RGB = img.convert("RGB")

                        result = None
                        max_attempts = 3
                        for attempt in range(max_attempts):
                            try:
                                result = modified_predict(img_RGB, device, model, transform)
                                break
                            except Exception:
                                if attempt < max_attempts - 1:
                                    time.sleep(5)

                        if result is None:
                            img.close()
                            img_RGB.close()
                            continue

                        try:
                            prob = result['RPS'] * 100
                            percent_jellyfish = np.round(prob, 2)

                            if percent_jellyfish >= (100 * params["percent_conf"]):
                                stem = fits_file.name
                                for ext in (".fits.gz", ".fits", ".fit"):
                                    if stem.lower().endswith(ext):
                                        stem = stem[:-len(ext)]
                                        break

                                filename = f"Score-{percent_jellyfish}%__{name}__{stem}__RA-{coord.ra.value:.3f}-DEC-{coord.dec.value:.3f}.jpg"
                                output_file_path = OUTPUT_DIR / filename
                                img.save(output_file_path, "JPEG", quality=100)
                        finally:
                            # Memory cleanup for image objects
                            img.close()
                            img_RGB.close()

                    print(f"[INFO] Completed {fits_file.name} in {time.perf_counter() - time_start:.2f}s.")


            # Script for coordinates only
            if not has_name and has_coords:
                print(f"[ROW {index + 1}]: Cordinates only -> Searching MAST with RA and DEC.")

                

            # Script for name and then for coordinates if name isn't resolved
            if has_name and has_coords:
                print(f"[ROW {index + 1}]: Name and Coordinates given -> Searching MAST with Name first, using RA and DEC as a fallback.")

                

                

# Execute Detection Pipeline
if __name__ == "__main__":
    main()



        

    

