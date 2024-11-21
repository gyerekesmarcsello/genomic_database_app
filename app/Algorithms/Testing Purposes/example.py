import loader
import visualization as vis
import sklearn as sk
from trainer import SkTrainer


if __name__ == "__main__":
    trainer = SkTrainer()
    trainer.preprocess()
    trainer.train()
    trainer.make_results()
