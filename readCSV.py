import csv

with open ("CsvFiles/basic1.csv", mode ='r') as file:
    csvfile = csv.DictReader(file)
    for lines in csvfile:
        print(lines)