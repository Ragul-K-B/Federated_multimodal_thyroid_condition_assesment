from flwr.server.strategy import FedAvg

class ThyroidFedStrategy(FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def aggregate_fit(self, rnd, results, failures):
        # Default FedAvg aggregation
        return super().aggregate_fit(rnd, results, failures)
