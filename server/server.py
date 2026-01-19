import flwr as fl
from strategy import ThyroidFedStrategy

def main():
    strategy = ThyroidFedStrategy(
        fraction_fit=1.0,
        min_fit_clients=3,
        min_available_clients=3,
    )

    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy
    )

if __name__ == "__main__":
    main()
