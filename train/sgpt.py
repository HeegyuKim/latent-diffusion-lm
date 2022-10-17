from silence_tensorflow import silence_tensorflow

silence_tensorflow()
from src.task.sgpt import SGPTTask
import hydra
from omegaconf import DictConfig
import os

os.environ["HYDRA_FULL_ERROR"] = "1"


@hydra.main("../config", "sgpt.yaml")
def main(config: DictConfig):
    SGPTTask.main(config)


if __name__ == "__main__":
    main()
