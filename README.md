# AutoML for Time Series Data in Federated Learning Context
## Federated Learning Project

This project implements a federated learning system where a server coordinates training among multiple clients. Each client contributes its local model updates to the server, which aggregates them to produce a global model.

<pre>Federated learning (FL) has emerged as a promising technique for
training machine learning models on decentralized data. It allows
training on data residing on user devices, addressing privacy
concerns and data security limitations. However, selecting the
optimal model architecture and hyperparameters for each FL task
can be a significant challenge.
</pre>

• Why Federated Learning?

    Traditional machine learning often requires
    centralizing data, raising privacy and security
    concerns. Federated learning offers a solution by
    training models on distributed data sets, eliminating
    the need for data transfer. This is particularly
    advantageous in scenarios where:
        - User data is privacy-sensitive (e.g., healthcare,finance).
        - Data is geographically dispersed acrossdevices.
        - Centralized data storage is impractical or infeasible.

• Project Objectives:

    This project proposes a novel AutoML framework for
    federated learning, aiming to automate the process
    of selecting machine learning algorithms and
    optimizing hyperparameters. Our framework will:
        ▪ Leverage meta-learning for efficient algorithm selection on the central server.
        ▪ Perform hyperparameter tuning on each client device to account for local data variations.
        ▪ Aggregate optimized hyperparameters from clients to the server for improved model performance.

## Installation

To install the required dependencies for this project, run the following command:


pip install -r requirements.txt

This will install all the necessary packages specified in the `requirements.txt` file.

## Run our part to get the final csv output file after Meta Features Extraction and the Aggregation   
 
run "automate_process.py" 

## Feature Extraction

The feature extraction piple-line extract all features from each client that will be aggregated in the server 

## Custom Aggregation Strategy

In the `aggStrategy` module, there is a custom aggregation strategy implemented. This strategy is responsible for aggregating the features received from each client. Scripts for additional feature processing or aggregation can be added to this module as needed.
