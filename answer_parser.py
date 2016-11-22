import numpy as np
import csv

def answer_parsing():
    with open('data/tr704-lst-ans', 'r') as answer_file:
        answer_file = csv.reader(answer_file, delimiter= '\t')
        answer = {row[1]: int(row[0]) for row in answer_file}
    return answer

if __name__ == '__main__':
    answer_parsing()
