from silence_tensorflow import silence_tensorflow

silence_tensorflow()
from src.task.latent_dialog_v2 import LatentDialogTaskV2
import hydra
from omegaconf import DictConfig
import os

os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

@hydra.main("../config", "latent-dialog-v2.yaml")
def main(config: DictConfig):
    LatentDialogTaskV2.main(config)


if __name__ == "__main__":
    main()
