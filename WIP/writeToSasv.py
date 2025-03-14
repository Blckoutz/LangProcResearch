def write_to_sasv(filename, data_list):
    """
    Writes a list of strings to a .sasv file, separating entries with '/*'.
    
    :param filename: Name of the output file (without extension)
    :param data_list: List of strings to be written to the file
    """
    full_filename = f"{filename}.sasv"
    with open(full_filename, "w", encoding="utf-8") as file:
        file.write(" /* ".join(data_list))
    print(f"Data successfully written to {full_filename}")

# Example usage
#data = ["Entry 1", "Entry 2", "Entry 3"]
#write_to_sasv("output", data)
