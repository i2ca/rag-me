import os
import pandas as pd

def compile_results(root_folder, output_file='compiled_results.csv'):
    # Initialize an empty list to store dataframes
    compiled_data = []

    # Iterate through all subfolders in the root folder
    for subfolder in os.listdir(root_folder):
        subfolder_path = os.path.join(root_folder, subfolder)
        
        # Only process if the current path is a directory
        if os.path.isdir(subfolder_path):
            # Look for the specific CSV file
            csv_file = os.path.join(subfolder_path, 'results_metrics.csv')
            if os.path.exists(csv_file):
                # Read the CSV file
                df = pd.read_csv(csv_file)
                
                # Extract chunk size and overlap from the folder name
                # parts = subfolder.split('-')
                # chunk_size = int(parts[2])  # Convert chunk_size to integer for proper sorting
                # overlap = int(parts[3])     # Convert overlap to integer for proper sorting

                # # Add chunk_size and overlap as columns to the dataframe
                # df['chunk_size'] = chunk_size
                # df['overlap'] = overlap

                # Drop the 'perplexity_accuracy_options' column
                df.drop(columns=['perplexity_accuracy_options'], inplace=True)

                # Append the dataframe to the list
                compiled_data.append(df)

    # Concatenate all the dataframes into one
    final_df = pd.concat(compiled_data, ignore_index=True)

    # Sort the final dataframe by chunk_size and overlap
    # final_df = final_df.sort_values(by=['chunk_size', 'overlap'])

    # Write the final dataframe to a CSV file with comma as decimal separator
    final_df.to_csv(output_file, index=False)

    print(f'Compiled results saved to {output_file}')

# Usage example
compile_results('../results/test')
