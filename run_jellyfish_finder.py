import sys
import configparser
from pathlib import Path

#Define repository paths
BASE_DIR = Path(__file__).resolve().parent

LOCAL_FITS_DIR = BASE_DIR / "local_fits"
WEIGHTS_DIR = BASE_DIR / "weights"

TEMP_DOWNLOADS_DIR = BASE_DIR / "temp_downloads"
OUTPUT_DIR = BASE_DIR / "output_candidates"

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
        return val.lower() in ("true", "1", "yes", "on")
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
        delimiters=('='),
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
        weights_path = (WEIGHTS_DIR / weights_path).resolve()

    if not weights_path.exists():
        print(f"[ERROR] Weights file cannot be found at: {weights_path}")
        sys.exit(1)

    # Get the minimum percent confidence from config file
    percent_conf = get_config_val(cfg, "MODEL", "percent_conf", fallback=0.6, cast_type=float)

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
    if params["mode"] == "local":

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
                print("[ERROR] Under mode 'local' no data path was given. Tried to fallback to './local_fits/', but the folder no longre exists.")
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

            

        

    

