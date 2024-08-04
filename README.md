# AutoML for Time Series Data in Federated Learning Context
## Federated Learning Project

This project implements a federated learning system where a server coordinates training among multiple clients. Each client contributes its local model updates to the server, which aggregates them to produce a global model.

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
