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



ssl_context = ssl._create_unverified_context()
opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_context))
urllib.request.install_opener(opener)


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



        

    

