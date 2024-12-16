import csv
import sys

def main():
    # Check if at least 2 arguments are provided: file path and one column
    if len(sys.argv) < 3:
        print("Usage: python read_csv_columns.py <file.csv> <column1> [<column2> ...]")
        sys.exit(1)
    
    # Parse command-line arguments
    file_path = sys.argv[1]
    selected_columns = sys.argv[2:]  # All remaining arguments are column names
    
    try:
        # Open the CSV file
        with open(file_path, 'r') as csv_file:
            reader = csv.DictReader(csv_file)
            # Ensure the selected columns exist
            missing_columns = [col for col in selected_columns if col not in reader.fieldnames]
            print(missing_columns)
            if missing_columns:
                print(f"Error: Columns {', '.join(missing_columns)} not found in the file.")
                print(f"Available columns: {', '.join(reader.fieldnames)}")
                sys.exit(1)
            
            # Print the selected columns
            print(",".join(selected_columns))  # Print header
            for row in reader:
                print(",".join(row[col] for col in selected_columns))
    
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

