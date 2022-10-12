from silence_tensorflow import silence_tensorflow

silence_tensorflow()
from src.task.optimus_v2 import OptimusTask
import hydra
from omegaconf import DictConfig


@hydra.main("../config", "optimus_v2.yaml")
def main(config: DictConfig):
    OptimusTask.main(config)


if __name__ == "__main__":
    main()
