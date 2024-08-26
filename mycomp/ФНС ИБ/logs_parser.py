import csv
import os
import glob

log_dir_path = 'logs'
csv_dir_path = 'structed_logs'
os.makedirs(csv_dir_path, exist_ok=True)

for log_file_path in glob.glob(os.path.join(log_dir_path, '*.log')):
    with open(log_file_path, 'r') as log_file:
        headers = None
        csv_file_name = os.path.basename(log_file_path).split('.')[0] + '.csv'
        csv_file_path = os.path.join(csv_dir_path, csv_file_name)

        with open(csv_file_path, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)

            for line in log_file:
                if line.startswith('#Fields:'):
                    headers = line.strip().split()[1:]
                    csv_writer.writerow(headers)
                    continue

                if line.startswith('#'):
                    continue

                if headers:
                    csv_writer.writerow(line.strip().split())

print("Все логи были успешно структурированы и сохранены в CSV-файлах.")
