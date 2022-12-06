"""Convert back to two-column format."""
import argparse
import io
from pathlib import Path


def is_clean_tag(tag_list):
    prev_pos, prev_lab = None, None
    found = False
    for tag in tag_list:
        if tag != "O":
            found = True
        pos, lab = tag[:2], tag[2:]
        if pos not in {"B-", "I-", "E-", "S-", "O"}:
            return False

        if prev_lab is not None:
            if pos in {"I-", "E-"} and lab != prev_lab:  # type conflict
                return False
        prev_pos, prev_lab = pos, lab
    if prev_pos not in {"E-", "S-", "O"}:  # not end well
        return False
    if not found:
        return False
    return True


def is_clean_tok(tok_list):
    found = False
    for tok in tok_list:
        if tok != "<unk>":
            found = True
        if tok[:2] in {"B-", "I-", "E-", "S-"}:
            return False
    if not found:
        return False
    return True


def convert(fout, data):
    success = 0
    for line in data:
        if line[0:4] == "# id":
            fout.write(line)
            pass
        else:
            tokens = line.split()
            tok_list, tag_list = [], []
            for i in range(len(tokens)):
                tok = tokens[i]
                tag = "O"
                if tok[:2] in {"B-", "I-", "E-", "S-"}:
                    continue
                prev_tok = ""
                if 0 < i < len(tokens):
                    prev_tok = tokens[i - 1]
                if prev_tok[:2] in {"B-", "I-", "E-", "S-"}:
                    tag = prev_tok
                if tok[-1] == "," or tok[-1] == ".":
                    tok1 = tok[:-1]
                    tok2 = tok[-1]
                    tok_list.append(tok1)
                    tag_list.append(tag)
                    tok_list.append(tok2)
                    tag_list.append("O")
                else:
                    tok_list.append(tok)
                    tag_list.append(tag)

            # if not is_clean_tok(tok_list):
            #     continue
            #
            # if not is_clean_tag(tag_list):
            #     continue

            res = []
            for tok, tag in list(zip(tok_list, tag_list)):
                res.append(f"{tok}\t{tag}")
            fout.write("\n".join(res))
            fout.write("\n")
        success += 1
    return success


def load(inp_file):
    data = []
    with io.open(inp_file, encoding="utf-8", errors="ignore") as fin:
        for line in fin:
            data.append(line.strip())
    return data


def build_args(parser):
    """Build arguments."""
    parser.add_argument("--inp_file", type=str, required=True)
    parser.add_argument("--out_file", type=str, required=True)
    parser.add_argument("--ignore_cat_label", action="store_true")
    return parser.parse_args()


def main():
    args = build_args(argparse.ArgumentParser())
    data = load(args.inp_file)
    args.log_file = Path(args.out_file).with_suffix(".log")
    flog = io.open(args.log_file, "w", encoding="utf-8", errors="ignore")
    flog.write(f"Read from {args.inp_file}: {len(data)}\n")

    with io.open(args.out_file, "w", encoding="utf-8", errors="ignore") as fout:
        success = convert(fout, data, args.ignore_cat_label)
    flog.write(f"Write to {args.out_file}: {success}\n")


if __name__ == "__main__":
    main()
