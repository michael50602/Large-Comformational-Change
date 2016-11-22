# -*- coding: utf-8 -*-
import re
import csv
import numpy as np

amino_acid = {}

def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def cal_profile(prop, seq_cont, window, stride):
    output = []
    sequence = []
    for c in seq_cont:
        sequence.append(c)
    sequence = np.array(sequence)
    if type(amino_acid.values()[0][prop]) is bool:
        for start in range(0, sequence.size - window + 1, stride):
            tmp = 0
            for c in sequence[start: start + window]:
                if c in amino_acid.keys():
                    if amino_acid[c][prop]:
                        tmp += 1
                else:
                    continue
            output.append(tmp)
    elif type(amino_acid.values()[0][prop]) is float:
        for start in range(0, sequence.size - window + 1, stride):
            tmp = 0
            for c in sequence[start: start + window]:
                if c in amino_acid.keys():
                    tmp += amino_acid[c][prop]
                else:
                    continue
            output.append(tmp)
    else:
        print "can't recognize property type"
    return output


def profile_parsing():
    feature_table = open('data/feature.csv', 'r')
    AA_property = []
    for row in csv.reader(feature_table):
        AA_property.append(row)
    AA_property = np.array(AA_property)
    property_list = AA_property[0, 1:]
    AA_list = AA_property[1:, 0]
    AA_property = AA_property[1:, 1:]
    for i, aa in enumerate(AA_list):
        amino_acid[aa] = {}
        for j, prop in enumerate(property_list):
            if isfloat(AA_property[i, j]):
                amino_acid[aa][prop] = float(AA_property[i, j])
            else:
                if AA_property[i, j] == "TRUE":
                    amino_acid[aa][prop] = True
                else:
                    amino_acid[aa][prop] = False
    src_file = open('data/all804-prot.fa', 'r')
    re_result = re.finditer(r">(?P<sequence_id>[0-9A-Z]*)\n(?P<sequence_content>[A-Z\n]*)\n", src_file.read())
    parse_output = {}
    total_len = 0
    cnt = 0
    for m in re_result:
        parse_output[m.groupdict()['sequence_id']] = {}
        total_len += len(m.groupdict()['sequence_content'])
        cnt += 1
        for prop in property_list:
            parse_output[m.groupdict()['sequence_id']][prop] = cal_profile(prop, m.groupdict()['sequence_content'], 5, 1)
    return parse_output
if __name__ == '__main__':
    run()
