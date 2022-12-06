import sys
import os
import re
import argparse

def to_mulda(inp_file, out_file):
    input_file = inp_file
    outpu_file = out_file
    with open(input_file, 'r') as fr, open(outpu_file,'w') as fw:
        lines = fr.readlines()
        for line in lines:
            if line[0] == '#':
                #continue
                fw.write(line)
                # fw.write('\n')
            elif line[0] == '\n':
                fw.write('\n')
            else:
                split_line = line.split()
                split_line.remove('_') # they all have 2 _
                split_line.remove('_')
                new_line = ('\t'.join([w for w in split_line])) + '\n'
                fw.write(new_line)


def to_coner(inp_file, out_file):
    input_file = inp_file
    outpu_file = out_file
    with open(input_file, 'r') as fr, open(outpu_file,'w') as fw:
        lines = fr.readlines()
        for line in lines:
            if line[0] == '#':
                fw.write(line + '\n')
            else:
                new_line = re.sub("\t", ' _ _ ', line)
                fw.write(new_line)


def build_args(parser):
    """Build arguments."""
    parser.add_argument("--inp_file", type=str, required=True)
    parser.add_argument("--out_file", type=str, required=True)
    # accepted templates are: "to_coner" & "to_mulda"
    parser.add_argument("--template", type=str, required=True)
    return parser.parse_args()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    args = build_args(argparse.ArgumentParser())
    inf = args.inp_file
    of= args.out_file

    template = args.template
    if template == "to_coner":
        to_coner(inf, of)
    elif template == "to_mulda":
        to_mulda(inf, of)






