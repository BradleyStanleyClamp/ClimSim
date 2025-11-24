import hydra
from omegaconf import DictConfig
import logging
from huggingface_hub import snapshot_download


@hydra.main(
    version_base=None, config_path="../config", config_name="download_data"
)
def main(cfg: DictConfig):

    logging.info("Starting download of ClimSim_low-res dataset...")
    snapshot_download(
        repo_id="LEAP/ClimSim_low-res",
        repo_type="dataset",
        local_dir="/gws/nopw/j04/iecdt/bstanleyclamp/ClimSim_lowres",
        max_workers=cfg.testing.num_workers,
    )
    logging.info("Download completed.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    main()
