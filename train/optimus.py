from silence_tensorflow import silence_tensorflow

silence_tensorflow()
from src.task.optimus import OptimusTask
import hydra
from omegaconf import DictConfig


@hydra.main("../config", "optimus.yaml")
def main(config: DictConfig):
    OptimusTask.main(config)


if __name__ == "__main__":
    main()
