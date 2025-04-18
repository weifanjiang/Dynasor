import json
import numpy as np
import os
import pandas as pd
from dynasor.core.evaluator import math_equal
from tqdm import tqdm


def program_level_policy(log, min_agreement_ratio, min_unchanged_steps, warmup_tokens=0):
    """
    Baseline policy that early-terminates entire reasoning
    """
    net_log, et_log = log["no_early_termination"], log["early_termination"]
    n_chains = len(et_log[0]['tokens_since_last_change'])
    max_tokens_per_chain = net_log['tokens_per_chain']

    for iteration in et_log:
        token_budget = iteration['token_budget']
        if token_budget < warmup_tokens:
            continue

        conf_tokens = None
        if min_unchanged_steps < 1:
            conf_tokens = int(token_budget * min_unchanged_steps)
        else:
            conf_tokens = int(min_unchanged_steps)

        merged_ans = np.array(iteration['merged_ans'])
        tslc = np.array(iteration['tokens_since_last_change'])
        conf_mask = (tslc >= conf_tokens)
        uniq_ans, counts = np.unique(merged_ans, return_counts=True)
        ratios = counts / n_chains
        idx = np.argsort(ratios)
        uniq_ans = uniq_ans[idx][::-1]
        ratios = ratios[idx][::-1]

        qualified_ans = None
        for ans, ratio in zip(uniq_ans, ratios):
            if qualified_ans is None:
                if ratio >= min_agreement_ratio:
                    conf_count = np.sum((merged_ans == ans) & conf_mask)
                    if conf_count / n_chains >= min_agreement_ratio:
                        qualified_ans = ans
        
        if qualified_ans is not None:
            total_tokens = 0
            for max_tokens in max_tokens_per_chain:
                total_tokens += min(max_tokens, token_budget)
            et_accuracy = iteration["merged_acc"][iteration['merged_ans'].index(qualified_ans)]
            return total_tokens, et_accuracy, True

    # not early terminated    
    total_tokens = np.sum(max_tokens_per_chain)
    uniq_ans, counts = np.unique(net_log['final_answers_per_chain'], return_counts=True)
    net_accuracy = net_log['final_accs_per_chain'][
        net_log['final_answers_per_chain'].index(uniq_ans[np.argmax(counts)])]
    return total_tokens, net_accuracy, False


def baseline_token_and_accuracy(log):
    if "no_early_termination" in log.keys():
        net_log = log['no_early_termination']
    else:
        net_log = log
    total_tokens = np.sum(net_log['tokens_per_chain'])
    uniq_ans, counts = np.unique(net_log['final_answers_per_chain'], return_counts=True)
    net_accuracy = net_log['final_accs_per_chain'][
        net_log['final_answers_per_chain'].index(uniq_ans[np.argmax(counts)])]
    return total_tokens, net_accuracy


def postprocess():

    baseline_dir = "token-to-acc-probing/math500/baseline"
    probe_dir = "token-to-acc-probing/math500/post-processed"
    output_dir = "token-to-acc-probing/math500/early_termination"
    os.system(f'mkdir -p {output_dir}')

    q_lists = [int(fname.split(".")[0]) for fname in os.listdir(baseline_dir) if fname.endswith(".json")]
    for q_idx in tqdm(q_lists):

        with open(os.path.join(baseline_dir, f"{q_idx:03d}.json"), "r") as fin:
            baseline_dat = json.load(fin)
        
        with open(os.path.join(probe_dir, f"{q_idx:03d}.json"), "r") as fin:
            probe_dat = json.load(fin)

        et = dict()
        et['no_early_termination'] = baseline_dat
        et['early_termination'] = list()  # per probing record

        token_budgets = probe_dat['token_budgets']
        changed_from_last = probe_dat['changed_from_last']
        probing_answers = probe_dat['probing_answers']
        probing_accs = probe_dat['probing_accs']

        num_chains = len(changed_from_last[0])
        num_iters = len(token_budgets)

        per_chain_max = baseline_dat['tokens_per_chain']
        cap = np.amax(per_chain_max)

        for i in range(num_iters):

            if token_budgets[i] > cap:
                continue

            iter_data = {"token_budget": token_budgets[i]}
            if i == 0:
                iter_data['tokens_since_last_change'] = [0, ] * num_chains
            else:
                previous_step = et['early_termination'][-1]['tokens_since_last_change']
                tslc = list()
                stepsize = token_budgets[i] - token_budgets[i-1]
                for chain_idx in range(num_chains):
                    if token_budgets[chain_idx] > per_chain_max[chain_idx]:
                        tslc.append(previous_step[chain_idx])
                    else:
                        if changed_from_last[i][chain_idx]:  # cope with a previous typo
                            tslc.append(previous_step[chain_idx] + stepsize)
                        else:
                            tslc.append(0)
                iter_data['tokens_since_last_change'] = tslc
            iter_data['probing_answers'] = probing_answers[i]
            iter_data['probing_accs'] = probing_accs[i]

            # merge final and probing
            iter_data['done'] = [token_budgets[i] >= per_chain_max[chain_idx] for chain_idx in range(num_chains)]
            merged_ans_raw = [
                fa if chain_done else pa
                for chain_done, fa, pa in zip(iter_data['done'], baseline_dat['final_answers_per_chain'], iter_data['probing_answers'])]
            merged_ans = [merged_ans_raw[0], ]
            unique_ans = [merged_ans_raw[0], ]
            for ans_idx in range(1, len(merged_ans_raw)):
                ans = merged_ans_raw[ans_idx]
                found, ua_idx = False, 0
                while (not found) and (ua_idx < len(unique_ans)):
                    if math_equal(ans, unique_ans[ua_idx]):
                        found = True
                    else:
                        ua_idx += 1
                if found:
                    merged_ans.append(unique_ans[ua_idx])
                else:
                    unique_ans.append(ans)
                    merged_ans.append(ans)
            iter_data['merged_ans'] = merged_ans
            iter_data['merged_acc'] = [
                fa if chain_done else pa
                for chain_done, fa, pa in zip(iter_data['done'], baseline_dat['final_accs_per_chain'], iter_data['probing_accs'])]

            et['early_termination'].append(iter_data)
        
        with open(os.path.join(output_dir, f"{q_idx:03d}.json"), 'w') as fout:
            json.dump(et, fout, indent=2)


if __name__ == '__main__':
    postprocess()
