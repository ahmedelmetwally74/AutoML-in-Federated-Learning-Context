import os
import shutil
import subprocess
import logging
import signal
from concurrent.futures import ThreadPoolExecutor, as_completed

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

dataset_folder = 'datasets'
num_clients = 5
client_folder = 'client'
client_processes = []
server_process = None

def setup_client_folder(client_id):
    try:
        client_dest = f'client{client_id}'
        if os.path.exists(client_dest):
            shutil.rmtree(client_dest)
        shutil.copytree(client_folder, client_dest)
        
        original_client_script = os.path.join(client_dest, 'client.py')
        new_client_script = os.path.join(client_dest, f'client{client_id}.py')
        os.rename(original_client_script, new_client_script)
        logging.info(f"Setup client folder for client {client_id}")
    except Exception as e:
        logging.error(f"Error setting up client folder for client {client_id}: {e}")

def run_command(command):
    try:
        subprocess.run(command, check=True)
        logging.info(f"Successfully ran command: {' '.join(command)}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running command {command}: {e}")

def terminate_processes():
    global client_processes, server_process
    logging.info("Terminating all processes")
    for client_process in client_processes:
        if client_process and client_process.poll() is None:  
            client_process.terminate()
            logging.info(f"Terminated client process {client_process.pid}")
    if server_process and server_process.poll() is None:  
        server_process.terminate()
        logging.info(f"Terminated server process {server_process.pid}")

def signal_handler(signum, frame):
    logging.info(f"Received signal {signum}, initiating cleanup")
    terminate_processes()
    exit(1)

def monitor_processes():
    global client_processes, server_process
    try:
        # Monitor client processes
        for client_process in client_processes:
            client_process.wait()
            logging.info(f"Client process {client_process.pid} finished")

        # Monitor server process
        if server_process:
            server_process.wait()
            logging.info("Server process finished")
    except Exception as e:
        logging.error(f"Error while monitoring processes: {e}")
        terminate_processes()

def main():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    datasets = os.listdir(dataset_folder)
    logging.info(f"Found datasets: {datasets}")

    for filename in datasets:
        dataset_path = os.path.join(dataset_folder, filename)
        filename_without_ext = os.path.splitext(filename)[0]
        logging.info(f"Processing dataset: {filename_without_ext}")

        # Splitting
        split_command = ['python', 'dataSetSpliter.py', dataset_path, str(num_clients)]
        logging.info(f"Running split command: {split_command}")
        run_command(split_command)

        # Start the server
        global server_process
        server_command = ['python', 'server.py']
        logging.info(f"Starting server with command: {server_command}")
        server_process = subprocess.Popen(server_command)
        logging.info("Server started")

        # Setup clients
        for i in range(1,num_clients+1):
            setup_client_folder(i)

        # Start the clients
        global client_processes
        client_processes = []
        with ThreadPoolExecutor(max_workers=num_clients) as executor:
            future_to_client = {
                executor.submit(subprocess.Popen, ['python', f'client{i}/client{i}.py', f'Data/{filename_without_ext}_split_{i}.csv', str(i)]): i for i in range(1,num_clients+1)
            }

            for future in as_completed(future_to_client):
                client_id = future_to_client[future]
                try:
                    client_process = future.result()
                    client_processes.append(client_process)
                    logging.info(f"Started client {client_id} with PID {client_process.pid}")
                except Exception as e:
                    logging.error(f"Error starting client {client_id}: {e}")

        # Monitor processes
        monitor_processes()

    logging.info("All datasets are complete")

if __name__ == "__main__":
    main()
