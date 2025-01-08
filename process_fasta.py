import argparse
import os

def process_fasta(input_fasta, output_fasta):
    with open(input_fasta, 'r') as infile:
        lines = infile.readlines()

    # Initialize variables
    output_lines = []
    current_entry = []
    
    for line in lines:
        if line.startswith('>'):
            if current_entry:
                # Process the accumulated entry
                entry_text = ''.join(current_entry)
                if '/' in entry_text:
                    split_entries = entry_text.split('/')
                    output_lines.append(split_entries[0])
                    output_lines.append('\n>vav\n' + split_entries[1])
                else:
                    output_lines.append(entry_text)
                current_entry = []
            current_entry.append(line)
        else:
            current_entry.append(line)

    # Process the last entry
    if current_entry:
        entry_text = ''.join(current_entry)
        if '/' in entry_text:
            split_entries = entry_text.split('/')
            output_lines.append(split_entries[0])
            output_lines.append('\n>vav\n' + split_entries[1])
        else:
            output_lines.append(entry_text)
    
    # Write the modified entries to the output file
    with open(output_fasta, 'w') as outfile:
        outfile.writelines(output_lines)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process FASTA file to handle entries with '/'")
    parser.add_argument('--input_fasta', required=True, help="Input FASTA file")
    parser.add_argument('--output_fasta', required=True, help="Output FASTA file")
    args = parser.parse_args()

    process_fasta(args.input_fasta, args.output_fasta)

