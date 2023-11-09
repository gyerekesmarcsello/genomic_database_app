import os
import pandas as pd
import vcf
from sqlalchemy import create_engine

# Database configuration
db_name = 'database'
db_user = 'username'
db_pass = 'secret'
db_host = 'db'
db_port = '5432'

# Connect to the database
db_string = 'postgresql://{}:{}@{}:{}/{}'.format(db_user, db_pass, db_host, db_port, db_name)
db = create_engine(db_string)

# Folder path containing CSV files
csv_folder_path = 'database\data\old'

# Folder path containing VCF files
vcf_folder_path = 'database\data\new'

# Function to load CSV files into the database
def load_csv_to_database():
    csv_files = [file for file in os.listdir(csv_folder_path) if file.endswith('.csv')]
    for csv_file in csv_files:
        file_path = os.path.join(csv_folder_path, csv_file)
        df = pd.read_csv(file_path)
        table_name = csv_file.replace('.csv', '')
        df.to_sql(table_name, db, index=False, if_exists='replace')
        print(f"Data from {csv_file} inserted into table {table_name}.")

# Function to load VCF files into the database
def load_vcf_to_database(vcf_folder_path, db):
    vcf_files = [file for file in os.listdir(vcf_folder_path) if file.endswith('.vcf')]
    data = []
    for vcf_file in vcf_files:
        print(f"Processing {vcf_file}...")
        vcf_reader = vcf.Reader(filename=os.path.join(vcf_folder_path, vcf_file))
        for record in vcf_reader:
            chromosome = record.CHROM
            position = record.POS
            reference = record.REF
            alternate = ','.join(map(str, record.ALT))
            quality = record.QUAL
            filters = ','.join(record.FILTER)
            info = ','.join([f"{key}={value}" for key, value in record.INFO.items()])
            data.append((chromosome, position, reference, alternate, quality, filters, info))

    columns = ['chromosome', 'position', 'reference', 'alternate', 'quality', 'filters', 'info']
    df = pd.DataFrame(data, columns=columns)
    table_name = "vcf_data"
    df.to_sql(table_name, db, index=False, if_exists='replace')
    print(f"All data from VCF files inserted into table {table_name}.")

# Main function
def main():
    load_csv_to_database()
    load_vcf_to_database()
    print("Data from CSV and VCF files inserted into the database.")

if __name__ == "__main__":
    main()





