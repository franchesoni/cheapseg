import re
import subprocess


def unify_readmes(output_file):
    # Run the shell command to unify all README.md files
    command = "find configs -type f \( -iname 'README.md' -o -iname 'readme.md' \) -print | sort | xargs cat > all_readmes.md"
    subprocess.run(command, shell=True, check=True)

    # Ensure the output is written to the specified file
    subprocess.run(f"mv all_readmes.md {output_file}", shell=True, check=True)



def extract_ade20k_tables(input_file, output_file):
    with open(input_file, "r") as file:
        content = file.read()

    # Updated pattern to capture tables only when 'ADE20K' appears relatively early in the line
    # and the total line length doesn't exceed a set limit from the start.
    ade20k_pattern = r"(^.{0,10}ADE20K.{0,10}$[\s\S]*?\|)\n\n"
    ade20k_tables = re.findall(ade20k_pattern, content, re.MULTILINE)

    # Write the extracted tables to the output file
    with open(output_file, "w") as file:
        for table in ade20k_tables:
            file.write(table + "\n\n")


def filter_tables_by_criteria(input_file, output_file):
    with open(input_file, "r") as file:
        content = file.read()

    filtered_content = []
    sections = re.split(r"\n\n+", content)  # Split based on double newlines

    for section in sections:
        lines = section.strip().split("\n")
        capturing = False
        header = None
        for line in lines:
            if "Method" in line:
                header = line
                continue
            if header and ("0000" in line and "512x512" in line):
                if not capturing:
                    filtered_content.append(header)
                    capturing = True
                filtered_content.append(line)

    with open(output_file, "w") as file:
        file.write("\n".join(filtered_content))


def normalize_header(header):
    return "|".join(part.strip() for part in header.split("|"))


def sort_by_miou(input_file, output_file):
    with open(input_file, "r") as file:
        lines = file.readlines()

    tuples = []
    current_header = None

    for line in lines:
        if "Method" in line:
            current_header = normalize_header(line.strip())
        elif current_header:
            # Split the header and line by '|' and clean up the entries
            header_parts = [h.strip() for h in current_header.split("|")]
            line_parts = [l.strip() for l in line.split("|")]

            # Find the index of the mIoU column
            miou_index = header_parts.index("mIoU")

            # Get the mIoU value from the line
            miou_value = float(line_parts[miou_index])
            tuples.append((current_header, line.strip(), miou_value))

    # Sort the tuples by mIoU in descending order
    tuples.sort(key=lambda x: x[2], reverse=True)

    # Write the sorted lines to the output file
    with open(output_file, "w") as file:
        previous_header = None
        for header, line, miou in tuples:
            if header != previous_header:
                file.write(header + "\n")
                previous_header = header
            file.write(line + "\n")


# Specify the input and output file paths
output_file = "ade20k_scores.md"
unify_readmes(output_file)
extract_ade20k_tables(output_file, output_file)
filter_tables_by_criteria(output_file, output_file)
sort_by_miou(output_file, output_file)
