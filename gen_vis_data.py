"""
Generates data for the visualizer.
"""

from itertools import islice
import os
import torch
from tqdm import tqdm
import ujson
import random
import argparse
import csv
import json
from baleen.utils.loaders import load_collectionX

from colbert.data import Queries
from colbert.infra import Run, RunConfig

from baleen.condenser.condense import Condenser
from baleen.hop_searcher import HopSearcher

from colbert.utils.utils import print_message
from transformers import AutoTokenizer

class CustomBaleen:
    """
    Like normal Baleen, but we collect more data on each hop.
    """
    def __init__(self, collectionX_path: str, searcher: HopSearcher, condenser: Condenser):
        self.collectionX = load_collectionX(collectionX_path)
        self.searcher = searcher
        self.condenser = condenser
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def search(self, query, num_hops, depth=100, verbose=False, stage1_topk=9, stage2_topk=5):
        assert depth % num_hops == 0, f"depth={depth} must be divisible by num_hops={num_hops}."
        k = depth // num_hops

        searcher = self.searcher
        condenser = self.condenser
        collectionX = self.collectionX

        all_facts = []
        all_ranking = []
        all_stage1 = []
        all_stage1_with_score = []
        all_stage2_with_score = []
        all_context = []
        all_context_toks = []
        all_q_matches = []
        all_c_matches = []

        facts = []
        context = None

        pids_bag = set()

        for hop_idx in range(0, num_hops):
            ranking = list(zip(*searcher.search(query, context=context, k=depth)))
            Q_ids, _ = self.searcher.checkpoint.query_tokenizer.tensorize([query], context=None if context is None else [context])
            Q_ids = Q_ids[0]
            print(Q_ids, Q_ids.shape)
            Q_toks = self.tokenizer.convert_ids_to_tokens(Q_ids)
            print(Q_toks)

            ranking_ = []
            scores = []
            hop_q_matches = []
            hop_c_matches = []

            facts_pids = set([pid for pid, _ in facts])

            for pid, rank, score, (q_matches, c_matches) in ranking:
                if len(ranking_) < k and pid not in facts_pids:
                    ranking_.append(pid)
                    scores.append(score)
                    hop_q_matches.append(q_matches.tolist())
                    hop_c_matches.append(c_matches.tolist() if c_matches is not None else [])
                
                if len(pids_bag) < k * (hop_idx+1):
                    pids_bag.add(pid)
            
            stage1_preds, facts, _, stage1_preds_with_score, stage2_preds_with_score = condenser.condense(query, backs=facts, ranking=ranking_, stage1_topk=stage1_topk, stage2_topk=stage2_topk)
            context = ' [SEP] '.join([collectionX.get((pid, sid), '') for pid, sid in facts])
            
            all_facts.append(facts)
            all_stage1.append(stage1_preds)
            all_stage1_with_score.append(stage1_preds_with_score)
            all_stage2_with_score.append(stage2_preds_with_score)
            all_context.append(context)
            all_ranking.append(list(zip(scores, ranking_)))
            all_context_toks.append(Q_toks)
            all_q_matches.append(hop_q_matches)
            all_c_matches.append(hop_c_matches)

        assert len(pids_bag) == depth

        return all_facts, all_stage1, all_stage1_with_score, all_stage2_with_score, all_context, all_ranking, all_context_toks, all_q_matches, all_c_matches, pids_bag


def main(args):
    # Parameters
    num_hops = 4
    retrieve_per_hop = 4
    depth = retrieve_per_hop * num_hops
    ncandidates = 8000
    stage1_topk = 9
    stage2_topk = 5

    collectionX_path = os.path.join(args.datadir, 'wiki.abstracts.2017/collection.json')

    checkpointL1 = os.path.join(args.datadir, 'hover.checkpoints-v1.0/condenserL1-v1.0.dnn')
    checkpointL2 = os.path.join(args.datadir, 'hover.checkpoints-v1.0/condenserL2-v1.0.dnn')

    with Run().context(RunConfig(root=args.root)):
        searcher = HopSearcher(index=args.index)
        condenser = Condenser(checkpointL1=checkpointL1, checkpointL2=checkpointL2,
                              collectionX_path=collectionX_path, deviceL1='cuda:0', deviceL2='cuda:0')

        baleen = CustomBaleen(collectionX_path, searcher, condenser)
        baleen.searcher.configure(nprobe=2, ncandidates=ncandidates)


    with open(args.qas, "r") as f:
        for row in tqdm(islice(f, 10)):
            row = json.loads(row)
            qid = row["qid"]
            query = row["question"]

            all_facts, all_stage1, all_stage1_with_score, all_stage2_with_score, all_context, all_ranking, all_ctx_toks, all_q_matches, all_c_matches, _ = baleen.search(query, num_hops=num_hops, depth=depth, stage1_topk=stage1_topk, stage2_topk=stage2_topk)
            
            # Generate data at each stage
            collectionX = baleen.collectionX
            all_hops = []
            for i, (facts, stage1, stage1_with_score, stage2_with_score, ctx, ranking, c_toks, q_matches, c_matches) in enumerate(zip(all_facts, all_stage1, all_stage1_with_score, all_stage2_with_score, all_context, all_ranking, all_ctx_toks, all_q_matches, all_c_matches)):
                ranking_data = [[score, collectionX.get((pid, 0), "").split("|")[0].strip()] for score, pid in ranking]
                stage1_data = [[score, collectionX.get((pid, sid), "")] for score, pid, sid in stage1_with_score]
                stage2_data = [[score, collectionX.get((pid, sid), "")] for score, (pid, sid) in stage2_with_score]
                hop_data = {
                    "c_toks": c_toks,
                    "q_matches": q_matches,
                    "c_matches": c_matches,
                    "ranking": ranking_data,
                    "stage1": stage1_data,
                    "stage2": stage2_data,
                    "new_ctx": ctx,
                }
                all_hops.append(hop_data)
            summary = {
                "query": query,
                "hops": all_hops,
            }
            with open(f"vis_data/{qid}.json", "w") as f:
                json.dump(summary, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--datadir", type=str, required=True)
    parser.add_argument("--index", type=str, required=True)
    parser.add_argument("--qas", type=str, required=True)

    args = parser.parse_args()
    main(args)
