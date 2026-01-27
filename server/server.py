import flwr as fl
import torch
import os

from flwr.server.strategy import FedAvg
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from glob_model import GlobModel

# ---------------- CONFIG ----------------
SAVE_DIR = "global_models"
LAST_MODEL = os.path.join(SAVE_DIR, "global_round_5.pth")
NUM_ROUNDS = 5
# ----------------------------------------

os.makedirs(SAVE_DIR, exist_ok=True)


class ResumeStrategy(FedAvg):
    """
    Custom strategy to save aggregated global model
    after every federated round
    """
    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            model = GlobModel()
            ndarrays = parameters_to_ndarrays(aggregated_parameters)

            state_dict = {
                k: torch.tensor(v)
                for k, v in zip(model.state_dict().keys(), ndarrays)
            }

            model.load_state_dict(state_dict)

            save_path = os.path.join(
                SAVE_DIR, f"global_round_{server_round}.pth"
            )
            torch.save(model.state_dict(), save_path)

            print(f"✅ Global model saved at {save_path}")

        return aggregated_parameters, metrics


def load_initial_parameters():
    """
    Load previous global model if exists
    (manual re-run support)
    """
    if os.path.exists(LAST_MODEL):
        print("🔁 Loading previous global model")
        model = GlobModel()
        model.load_state_dict(torch.load(LAST_MODEL, map_location="cpu"))

        return ndarrays_to_parameters(
            [v.cpu().numpy() for v in model.state_dict().values()]
        )

    print("🆕 No previous model found, starting fresh")
    return None


def main():
    strategy = ResumeStrategy(
        fraction_fit=1.0,
        min_fit_clients=3,
        min_available_clients=3,
        initial_parameters=load_initial_parameters(),
    )

    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
