import sys
import os
import re


def to_mulda(file_name):
    source_file = file_name
    with open(source_file, 'r') as fr, open("mulda-"+source_file,'w') as fw:
        lines = fr.readlines()
        mulda_doc_start = "-DOCSTART-	O\n"
        fw.write(mulda_doc_start)
        fw.write('\n')
        for line in lines:
            if line[0] == '#':
                #continue
                fw.write(line)
                fw.write('\n')
            elif line[0] == '\n':
                fw.write('\n')
            else:
                split_line = line.split()
                split_line.remove('_') # they all have 2 _
                split_line.remove('_')
                new_line = ('\t'.join([w for w in split_line])) + '\n'
                fw.write(new_line)


def to_coner(file_name):
    source_file = file_name
    with open(source_file, 'r') as fr, open("mulda-"+source_file,'w') as fw:
        lines = fr.readlines()
        for line in lines:
            if line[0:5]== '-DOCS':
                continue
            elif line[0] == '#':
                fw.write(line + '\n')
            else:
                new_line = re.sub("\t", ' _ _ ', line)
                fw.write(new_line)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    #source_file = "coner-en-test.txt"
    output_file = "out.txt"
    #to_mulda(source_file)
    to_coner(output_file)



