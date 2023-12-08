import os
import pandas as pd
import vcfpy

# Function to export VCF files into a CSV file with 'info' column parsed
def export_vcf_to_csv(vcf_folder_path, csv_export_path):
    vcf_files = [file for file in os.listdir(vcf_folder_path) if file.endswith('.vcf')]
    data = []
    all_info_keys = set()  # Store all encountered info keys
    for vcf_file in vcf_files:
        print(f"Processing {vcf_file}...")
        vcf_reader = vcfpy.Reader.from_path(os.path.join(vcf_folder_path, vcf_file))
        for record in vcf_reader:
            chromosome = record.CHROM
            position = record.POS
            reference = record.REF
            alternate = ','.join(str(alt) for alt in record.ALT)
            quality = record.QUAL
            filters = ','.join(record.FILTER)

            info_dict = record.INFO
            all_info_keys.update(info_dict.keys())  # Update with new keys encountered
            data_row = (chromosome, position, reference, alternate, quality, filters)
            for key in all_info_keys:
                data_row += (info_dict.get(key, None),)  # Add values or None if key not present
            data.append(data_row)

    columns = ['chromosome', 'position', 'reference', 'alternate', 'quality', 'filters'] + list(all_info_keys)
    df = pd.DataFrame(data, columns=columns)
    
    # Exporting to CSV with expanded 'info' columns
    df.to_csv(csv_export_path, index=False)
    print(f"All data from VCF files exported to CSV: {csv_export_path}")

# Main function
def main():
    vcf_folder_path = "database/data/new"
    csv_export_path = "database/data/vcf_data.csv"
    export_vcf_to_csv(vcf_folder_path, csv_export_path)

if __name__ == "__main__":
    main()