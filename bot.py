from staragent import StarAgent


class CompetitiveBot(StarAgent):
    def __init__(self):
        super().__init__(
            train_mode=False,
            log_mlflow=False,
            compile_model=False,
        )
        self.load_checkpoint("checkpoints/hannibal_p1_v11.pt")
