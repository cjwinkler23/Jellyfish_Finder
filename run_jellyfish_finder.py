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
import shutil
import astroquery
import time
import sep
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astroquery.mast import Observations
from astropy.table import Table
from astropy.coordinates import SkyCoord
from skimage import measure
from regions import PolygonSkyRegion
from photutils.background import Background2D, MedianBackground
from photutils.segmentation import make_2dgaussian_kernel
from photutils.segmentation import detect_sources
from photutils.segmentation import SourceCatalog
from PIL import Image


#Define repository paths
BASE_DIR = Path(__file__).resolve().parent

LOCAL_FITS_DIR = BASE_DIR / "local_fits"
WEIGHTS_DIR = BASE_DIR / "weights"

TEMP_DOWNLOADS_DIR = BASE_DIR / "temp_downloads"
OUTPUT_DIR = BASE_DIR / "output_candidates"

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
                print(f"[ERROR] Invalid filters given in 'filt_list' under [NAME_COORDS_MODE] section. The following filters will be ignored: {invalid_filters}")
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
                for i, prod in enumerate(product):
                    if ".fits" in prod['productFilename']:
                        row = i
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
                    data = image['SCI'].data.copy()
                    header = image['SCI'].header.copy()
                
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
                    for attempt in range(5):
                        try:
                            result = modified_predict(img_RGB, device, model, transform)
                            break
                        except Exception as e:
                            if attempt < attempts - 1:
                                time.sleep(10)

                    if result is None:
                        continue

                    prob = result['RPS'] * 100
                    percent_jellyfish = np.round(prob, 2)

                    #Give final file name and save if score > 60%
                    if percent_jellyfish >= 60.0:
                        file_name = 'output_candidates/' + 'Score-' + str(percent_jellyfish) + '%__' + cluster + '__' + observation + '__RA-' + str(np.round(coord.ra.value, 3)) + '-DEC-' + str(np.round(coord.dec.value, 3)) + '.jpg'
                        img.save((file_name), "JPEG", quality=100)

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







        

    elif params["mode"] == "name_coords":
        print(f"[INFO] Running NAME_COORDS mode.")









        

# Execute Detection Pipeline
if __name__ == "__main__":
    main()



        

    

