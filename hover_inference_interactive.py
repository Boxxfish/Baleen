import os
import tqdm
import ujson
import random
import argparse
import csv

from colbert.data import Queries
from colbert.infra import Run, RunConfig

from baleen.condenser.condense import Condenser
from baleen.hop_searcher import HopSearcher
from baleen.engine import Baleen

from colbert.utils.utils import print_message


def main(args):
    print_message("#> Starting...")

    collectionX_path = os.path.join(args.datadir, 'wiki.abstracts.2017/collection.json')

    checkpointL1 = os.path.join(args.datadir, 'hover.checkpoints-v1.0/condenserL1-v1.0.dnn')
    checkpointL2 = os.path.join(args.datadir, 'hover.checkpoints-v1.0/condenserL2-v1.0.dnn')

    with Run().context(RunConfig(root=args.root)):
        searcher = HopSearcher(index=args.index)
        condenser = Condenser(checkpointL1=checkpointL1, checkpointL2=checkpointL2,
                              collectionX_path=collectionX_path, deviceL1='cuda:0', deviceL2='cuda:0')

        baleen = Baleen(collectionX_path, searcher, condenser)
        baleen.searcher.configure(nprobe=2, ncandidates=8192)

    while True:
        query = input("> ")

        facts, pids_bag, _ = baleen.search(query, num_hops=4)
        print("Facts:")
        for fact in facts:
            pid = fact[0]
            with open("../wiki.abstracts.2017/collection.tsv") as f:
                tsv_file = csv.reader(f, delimiter="\t")
                for row in tsv_file:
                    if row[0] == str(pid):
                        print(row[1])
                        break
        print("")
        print("********")
        print("")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--datadir", type=str, required=True)
    parser.add_argument("--index", type=str, required=True)

    args = parser.parse_args()
    main(args)
