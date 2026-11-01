# Jellyfish Finder

**Version:** V1

**Jellyfish Finder** is an end-to-end Python script that uses a Vision Transformer model to automatically search through desired galaxy cluster images from HST and JWST (with possibilities of Roman and Euclid) for new candidates of galaxies undergoing ram-pressure striping (RPS). The current version only allows the model to perform inferences using **CPU**, but later use of CUDA may be implemented.

## Installation
1. **Clone this repository:**
   ```bash
   git clone https://github.com/cjwinkler23/Jellyfish_Finder
   cd Jellyfish_Finder
   ```

2. **Create and activate the Conda environment:**
   ```bash
   conda env create -f environment.yml               #Change this to the actual environment file when done
   conda activate your_env_name
   ```

## Repository Structure

```
Jellyfish_Finder/
├── local_fits/                Recommended location for FITS images of clusters to be stored in 'local' mode
├── output_candidates/         Final output location of jellyfish candidate cutouts
├── temp_downloads/            Temporary storage location to download cluster images from MAST
├── weights/                   Folder to contain the .pth weights file
├── config.ini                 Configuration file used to specify mode and user parameters
├── run_jellyfish_finder.py    Main Python script
├── env.yml                    Environment file containing all required packages

```

## Modes
The run_jellyfish_finder.py script can be run in one of three modes depending on the user's needs and preferences. The selected mode changes how the script is fed the desired cluster images:

1. **obs_id:**
   - Requires the most preliminary work, but offers the most control over the final cluster image.
   - Requires a CSV file with column headers - cluster_name, obs_id - where each row represents one MAST resolvable HST or JWST observation of a particular cluster.
   - cluster_name is only used as a designating term that's put at the beginning of each cutout's file name that came from the corresponding observation, not for the MAST search.
   - obs_id is used to query the MAST database and download the fits file associated with that particular observation.
  
2. **name_coords:**
3. **local:**

## Usage


## Weights
The current model has resulted from:
- **Backbone Architecture:** Meta's [DINOv2 ViT-S/14](https://github.com/facebookresearch/dinov2 "facebookresearch/dinov2") weights.
- **Training Framework:** Fine-tuning was done using the script provided by [AstroVink](https://github.com/SaamieVincken/AstroVink).
- **Fine-Tuning:** All transformer layers and the final classification head were left unfrozen and fine-tuned on a sample of ~50 (V1) RPS galaxies.












## Acknowledgements

This project utilizes the DINOv2 architecture developed by Meta AI. 

```bibtex
@misc{oquab2023dinov2,
  title={DINOv2: Learning Robust Visual Features without Supervision},
  author={Oquab, Maxime and Darcet, Timothée and Moutakanni, Théo and Vo, Huy and Szafraniec, Marc and Khalidov, Vasil and Labatut, Patrick and Joulin, Armand and Bojanowski, Piotr},
  journal={arXiv preprint arXiv:2304.07193},
  year={2023}
}
```

This project utilizes the AstroVink pipeline developed by Saamie Vincken to fine-tune the vision transformer model.

```bibtex
@software{vincken2024astrovink,
  author       = {Vincken, Saamie},
  title        = {AstroVink: Visual Transformers for Astronomical Source Classification},
  year         = {2024},
  publisher    = {GitHub},
  journal      = {GitHub Repository},
  howpublished = {\url{https://github.com}}
}
```
