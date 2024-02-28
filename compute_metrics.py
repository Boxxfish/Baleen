from argparse import ArgumentParser
from tqdm import tqdm
import json

def f7(seq):
    seen = set()
    result = []
    for item in seq:
        if item[0] not in seen:
            seen.add(item[0])
            result.append(item)
    return result

def main():
    parser = ArgumentParser()
    parser.add_argument("--output-data")
    parser.add_argument("--qas")
    args = parser.parse_args()

    with open(args.output_data, "r") as f:
        out_data = json.load(f)

    qas = {}
    with open(args.qas, "r") as f:
        for row in f:
            row = json.loads(row)
            qas[str(row["qid"])] = row

    psg_correct_exact = []
    psg_f1 = []
    s_correct_exact = []
    s_f1 = []
    for qid in tqdm(out_data):
        our_facts, pids = out_data[qid]
        our_facts = set(map(tuple, our_facts))
        gold_facts = qas[qid]["support_facts"]
        gold_facts = set(map(tuple, gold_facts))

        our_facts_psg = set([fact[0] for fact in our_facts])
        gold_facts_psg = set([fact[0] for fact in gold_facts])
        psg_correct_exact.append(our_facts_psg == gold_facts_psg)

        s_correct_exact.append(set(our_facts) == set(gold_facts))

        psg_prec = len(our_facts_psg.intersection(gold_facts_psg)) / len(our_facts_psg)
        psg_recall = len(our_facts_psg.intersection(gold_facts_psg)) / len(gold_facts_psg)
        if psg_prec == 0 or psg_recall == 0:
            psg_f1.append(0)
        else:
            psg_f1.append((2 * psg_prec * psg_recall) / (psg_prec + psg_recall))

        s_prec = len(our_facts.intersection(gold_facts)) / len(our_facts)
        s_recall = len(our_facts.intersection(gold_facts)) / len(gold_facts)
        if s_prec == 0 or s_recall == 0:
            s_f1.append(0)
        else:
            s_f1.append((2 * s_prec * s_recall) / (s_prec + s_recall))

    print("Passage EM:", sum(psg_correct_exact) / len(psg_correct_exact))
    print("Passage F1:", sum(psg_f1) / len(psg_f1))
    print("Sentence EM:", sum(s_correct_exact) / len(s_correct_exact))
    print("Sentence F1:", sum(s_f1) / len(s_f1))

if __name__ == "__main__":
    main()
