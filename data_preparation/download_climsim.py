import hydra
from omegaconf import DictConfig
import logging
from huggingface_hub import snapshot_download


@hydra.main(version_base=None, config_path="../config", config_name="download_data")
def main(cfg: DictConfig):

    hf_token = cfg.hf_token

    logging.info("Starting download of ClimSim_low-res dataset...")
    patterns = []
    for i in cfg.download_folders:
        patterns.append(f"train/000{str(i)}-*/*")
    logging.info(f"Downloading folders: {patterns}")
    snapshot_download(
        repo_id="LEAP/ClimSim_low-res",
        repo_type="dataset",
        local_dir="/gws/nopw/j04/iecdt/bstanleyclamp/ClimSim_lowres",
        max_workers=8,
        allow_patterns=patterns,
        token=hf_token,
    )
    logging.info("Download completed.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    main()
