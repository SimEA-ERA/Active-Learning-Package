import argparse
import pandas as pd
from tabulate import tabulate

def read_csv_and_print_table(file_path):
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Print the DataFrame as a formatted table
        print(tabulate(df, headers='keys', tablefmt='fancy_grid', showindex=False))
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Read a CSV file and print it as a formatted table.")
    parser.add_argument("file_path", help="Path to the CSV file")
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Call the function to read and print the table
    read_csv_and_print_table(args.file_path)

