import csv

def tsv_to_csv(input_tsv, output_csv):
    with open(input_tsv, mode='r', newline='', encoding='utf-8') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')  # Ensure tab is the delimiter
        
        with open(output_csv, mode='w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerows(reader)

# Example usage
input_tsv = "csv/combined_season1-40.tsv"
output_csv = "TSVtoCSVoutput.csv"
tsv_to_csv(input_tsv, output_csv)
