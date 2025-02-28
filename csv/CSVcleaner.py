import csv

def remove_rows_with_links(input_file, output_file):
    with open(input_file, mode='r', newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        rows = [row for row in reader if not any('<a href' in cell for cell in row)]
    
    with open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(rows)

input_csv= r"C:\Users\Blackoutz\Documents\GitHub\LangProcResearch\WIP\JEOPARDY_CSV.csv" #replace with the csv file you want to read in, make sure the r stays before the "" to ensure it stays as a literal for windows
output_csv = "output.csv"
remove_rows_with_links(input_csv, output_csv)