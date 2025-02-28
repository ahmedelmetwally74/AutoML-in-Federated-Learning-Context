import flwr as fl
from server_utils.agg_strategy import CustomStrategy

class FlowerServer(fl.server.Server):
    def __init__(self, strategy: fl.server.strategy.Strategy):
        # Create a SimpleClientManager to manage the clients
        client_manager = fl.server.SimpleClientManager()

        # Initialize the Server class with the SimpleClientManager and custom strategy
        super().__init__(client_manager=client_manager, strategy=strategy)

if __name__ == "__main__":
    # Create an instance of your custom strategy
    round_number = 3
    custom_strategy = CustomStrategy(round_number=round_number)
    # Create an instance of your server with the custom strategy
    server = FlowerServer(strategy=custom_strategy)
    print("start server")
    # Start the Flower server with the custom strategy
    fl.server.start_server(server_address="localhost:5555", server=server
                           , config=fl.server.ServerConfig(num_rounds=round_number))
    


