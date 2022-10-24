from silence_tensorflow import silence_tensorflow

silence_tensorflow()
from src.task.latent_dialog import LatentDialogTask
import hydra
from omegaconf import DictConfig
import os

os.environ["HYDRA_FULL_ERROR"] = "1"


@hydra.main("../config", "latent-dialog.yaml")
def main(config: DictConfig):
    LatentDialogTask.main(config)


if __name__ == "__main__":
    main()
