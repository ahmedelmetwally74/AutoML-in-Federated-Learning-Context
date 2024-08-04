# AutoML for Time Series Data in Federated Learning Context

## Overview
Federated learning (FL) has emerged as a compelling approach for training machine learning models on decentralized data. Enabling training directly on user devices addresses key privacy concerns and bypasses data security limitations. Despite its benefits, selecting optimal model architectures and hyperparameters for each FL task remains challenging.

This project introduces a federated learning system coordinated by a server, where multiple clients contribute local model updates. The aggregated result is a robust global model that leverages insights from diverse data sources without compromising data privacy.

**Mentorship**: This project is conducted under the mentorship of [Giza Systems](http://www.gizasystems.com).

## Why Federated Learning?
- **Privacy and Security**: Unlike traditional machine learning, FL does not require centralizing sensitive data, enhancing privacy and security.
- **Applicability**: FL is invaluable in scenarios where data is:
  - Privacy-sensitive (e.g., healthcare, finance)
  - Geographically dispersed across devices
  - Infeasible to centralize due to logistical or regulatory reasons

## Project Objectives
The goal is to develop an AutoML framework tailored for federated environments, automating the selection and optimization of machine learning algorithms and hyperparameters:
- **Meta-Learning**: Utilize meta-learning on the central server for effective algorithm selection.
- **Hyperparameter Tuning**: Conduct hyperparameter optimization locally on client devices to adapt to data variations.
- **Aggregation**: Combine optimized hyperparameters from all clients to enhance the global model's performance.

## Getting Started

### Installation
Install the required dependencies by running the following:

pip install -r requirements.txt

This will install all the necessary packages specified in the `requirements.txt` file.

### Running the Process
Generate the final CSV output after meta-feature extraction and aggregation: 
run "automate_process.py" 


## Modules Description

### Feature Extraction
The feature extraction pipeline extracts all relevant features from each client, which are then aggregated on the server.

### Custom Aggregation Strategy
The `aggStrategy` module implements a custom strategy for aggregating features received from clients. This module can be extended with scripts for additional feature processing or aggregation as needed.

## Additional Resources

- [Project Experiments Repository](<[insert-link-here](https://github.com/ahmedelmetwally74/AutoML-For-Time-Series-in-Federated-Learning-Context)>)
- [Project Presentation](<insert-link-here>)

## Acknowledgements
This project was developed with mentorship provided by [Giza Systems](http://www.gizasystems.com), whose guidance was invaluable in the practical application of federated learning principles and AutoML strategies.
