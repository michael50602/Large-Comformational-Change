import numpy as np
import csv

def answer_parsing():
    with open('data/tr704-lst-ans', 'r') as answer_file:
        answer_file = csv.reader(answer_file, delimiter= '\t')
        answer = {row[1]: int(row[0]) for row in answer_file}
    return answer

if __name__ == '__main__':
    ans = answer_parsing()
    print(ans.keys()) 
    neg = 0
    pos = 0
    for v in ans.values():
        if v == 1:
            pos += 1
        else:
            neg += 1
    print ("negative sample is %d, positive sample is %d"%(neg, pos))
