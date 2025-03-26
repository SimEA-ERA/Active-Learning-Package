import argparse
import numpy as np

def read_file(file_path):
    try:
        with open(file_path, 'r') as file:
            return file.readlines()
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None

def extract_last_Input_orientation(log_lines):
    for j,line in enumerate(log_lines):
        if 'Input orientation:' in line:
            ji = j+1
    
    coords = []
    for line in log_lines[ji+4:]:
        if '-------' in line:
            break
        s = line.strip('\n').split()[-3:]
        print( s )
        coords.append(s )
    return coords

def main():
    parser = argparse.ArgumentParser(description="Read Gaussian input template and log files.")
    parser.add_argument("--input", required=True, help="Path to the Gaussian input template file.")
    parser.add_argument("--output", required=True, help="Path to the Gaussian input template file.")
    parser.add_argument("--log", required=True, help="Path to the Gaussian log file.")
    parser.add_argument("--command", required=True, help="Path to the Gaussian log file.")

    args = parser.parse_args()

    input_lines = read_file(args.input)
    for j, line in enumerate(input_lines):
        if '#p' in line:
            input_lines[j] = ' '.join(input_lines[j].strip('\n').split()[:-1] + [ args.command, '\n'])
    
    log_lines = read_file(args.log)

    coords = extract_last_Input_orientation(log_lines)
    for j,line in enumerate(input_lines[7:-1]):
        new_line = [line.split()[0]] + [str(c) for c in coords[j] ] + [ '\n']
        print(new_line)
        input_lines[7+j] = '  '.join(new_line)

    with open(args.output, 'w') as f:
        for line in input_lines:
            f.write(line)
        f.closed

if __name__ == "__main__":
    main()

