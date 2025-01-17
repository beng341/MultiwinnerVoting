import os
import subprocess

# Define the ranges for m and num_winners
m_values = [7]

# Base command components
train_command_base = "python -m network_ops.train_networks"
eval_command_base = "python -m network_ops.evaluate_networks"
data_path = "/Users/joshuacaiata/Dev/MultiwinnerVoting/results/josh/real_world_experiment"
out_folder = "/Users/joshuacaiata/Dev/MultiwinnerVoting/results/josh/real_world_experiment"
network_path = "/Users/joshuacaiata/Dev/MultiwinnerVoting/results/josh/real_world_experiment/trained_networks"

# Iterate over the values of m and num_winners
for m in m_values:
    for num_winners in range(1, m):
        # Construct the train command
        train_command = (
            f"{train_command_base} "
            f"\"m={m}\" "
            f"\"num_winners={num_winners}\" "
            f"\"data_path='{data_path}'\" "
            f"\"out_folder='{out_folder}'\""
        )
        
        print(f"Running train command: {train_command}")
        
        # Run the train command and wait for it to finish
        train_process = subprocess.run(train_command, shell=True)
        
        # Check if the train command was successful
        if train_process.returncode != 0:
            print(f"Train command failed with return code {train_process.returncode}. Exiting.")
            exit(train_process.returncode)
        
        # Construct the evaluate command
        eval_command = (
            f"{eval_command_base} "
            f"\"m={m}\" "
            f"\"num_winners={num_winners}\" "
            f"\"data_path='{data_path}'\" "
            f"\"out_folder='{out_folder}'\" "
            f"\"network_path='{network_path}'\""
        )
        
        print(f"Running evaluate command: {eval_command}")
        
        # Run the evaluate command and wait for it to finish
        eval_process = subprocess.run(eval_command, shell=True)
        
        # Check if the evaluate command was successful
        if eval_process.returncode != 0:
            print(f"Evaluate command failed with return code {eval_process.returncode}. Exiting.")
            exit(eval_process.returncode)
