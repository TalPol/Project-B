import json
import pysam  # Ensure you have pysam installed: pip install pysam

def process_sam_file(sam_file, json_file):
    alignment_data = {}

    # Open the SAM file using pysam
    with pysam.AlignmentFile(sam_file, "r") as sam:
        for read in sam:
            if not read.is_unmapped:
                read_id = read.qname
                # Extract the ss and si tags
                ss_tag = read.get_tag('ss') if read.has_tag('ss') else None
                si_tag = read.get_tag('si') if read.has_tag('si') else None
                
                # Get the start and end signal positions
                signal_start = int(ss_tag) if ss_tag else None
                signal_end = int(si_tag) if si_tag else None
                
                # Store the data in alignment_data
                alignment_data[read_id] = {
                    'signal_start': signal_start,
                    'signal_end': signal_end,
                    'base_positions': []  # Placeholder for base positions
                }
                
                # Populate the base positions if needed, or process further as required.

    # Write the alignment data to a JSON file
    with open(json_file, 'w') as json_file:
        json.dump(alignment_data, json_file, indent=4)

if __name__ == "__main__":
    # Specify input file paths
    sam_file = 'file.sam'        # Path to your SAM file
    json_file = 'orig.json'      # Path for the output JSON file

    process_sam_file(sam_file, json_file)
    print(f"JSON file created: {json_file}")
