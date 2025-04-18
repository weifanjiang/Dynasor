import json
import time
import matplotlib.pyplot as plt
import os
from dynasor.core.entropy import entropy, norm
from dynasor.core.evaluator import math_equal
from tqdm import tqdm


def get_exp_logs(exp_dir):
    exp_logs = list()
    for fname in os.listdir(exp_dir):
        if fname.endswith(".json"):
            with open(os.path.join(exp_dir, fname), "r") as f:
                exp_logs.append(json.load(f))
    return exp_logs


def main():

    exp_logs = get_exp_logs("token-to-acc-probing/math500")

    # this is for groundtruth
    baseline_dir = "token-to-acc-probing/math500/baseline"
    os.system(f'mkdir -p {baseline_dir}')
    for exp_log in tqdm(exp_logs, desc='process groundtruth results'):
        idx = exp_log['idx']
        pp_fname = os.path.join(baseline_dir, f"{idx:03d}.json")
        if os.path.exists(pp_fname):
            continue

        groundtruth = exp_log["answer"]
        num_chains = len(exp_log["iterations"][0]["curr_answers"])
        tokens_per_chain = [0 for _ in range(num_chains)]
        final_answers_per_chain = [None for _ in range(num_chains)]
        for iteration in exp_log["iterations"]:
            curr_length = iteration["token_budget"]
            for chain_idx, curr_generation in enumerate(iteration["responses"]):
                if curr_generation != "":
                    tokens_per_chain[chain_idx] = curr_length
                    final_answers_per_chain[chain_idx] = iteration["curr_answers"][chain_idx]
        
        unique_answers = [final_answers_per_chain[0], ]
        formatted_answers = [final_answers_per_chain[0], ]
        for ans in final_answers_per_chain[1:]:
            ua_idx, found = 0, False
            while (not found) and (ua_idx < len(unique_answers)):
                if math_equal(ans, unique_answers[ua_idx]):
                    found = True
                else:
                    ua_idx += 1
            if found:
                formatted_answers.append(unique_answers[ua_idx])
            else:
                unique_answers.append(ans)
                formatted_answers.append(ans)
        final_accs_per_chain = [math_equal(groundtruth, ans) for ans in formatted_answers]
        
        pp = {
            "groundtruth": groundtruth,
            "tokens_per_chain": tokens_per_chain,
            "final_answers_per_chain": formatted_answers,
            "final_accs_per_chain": final_accs_per_chain
        }

        with open(pp_fname, "w") as f:
            json.dump(pp, f, indent=2)


    # this is for probing
    out_dir = "token-to-acc-probing/math500/post-processed"
    os.system(f'mkdir -p {out_dir}')
    for exp_log in tqdm(exp_logs, desc='process probing results'):
        idx = exp_log['idx']
        pp_fname = os.path.join(out_dir, f"{idx:03d}.json")
        if os.path.exists(pp_fname):
            continue

        groundtruth = exp_log["answer"]
        # get each chain's accuracy
        token_bugets = list()
        probing_accs = list()
        probing_answers = list()
        changed_from_last = list()

        for iter_idx, iteration in enumerate(exp_log["iterations"]):
            token_bugets.append(iteration["token_budget"])
            unique_answers = [iteration["probing_answers"][0], ]
            formatted_answers = [iteration["probing_answers"][0], ]
            for ans in iteration["probing_answers"][1:]:
                ua_idx, found = 0, False
                while (not found) and (ua_idx < len(unique_answers)):
                    if math_equal(ans, unique_answers[ua_idx]):
                        found = True
                    else:
                        ua_idx += 1
                if found:
                    formatted_answers.append(unique_answers[ua_idx])
                else:
                    unique_answers.append(ans)
                    formatted_answers.append(ans)
            probing_answers.append(formatted_answers)
            
            corr_idx, corr_found = 0, False
            while (not corr_found) and (corr_idx < len(unique_answers)):
                if math_equal(groundtruth, unique_answers[corr_idx]):
                    corr_found = True
                else:
                    corr_idx += 1
            if corr_found:
                corr_ans = unique_answers[corr_idx]
                probing_accs.append([corr_ans == ans for ans in formatted_answers])
            else:
                probing_accs.append([False for _ in range(len(formatted_answers))])
            
            # get changed from last
            if iter_idx == 0:
                changed_from_last.append([None for _ in range(len(formatted_answers))])
            else:
                last_answers = probing_answers[iter_idx - 1]
                changed_from_last.append(
                    [math_equal(la, ca) for la, ca in zip(last_answers, formatted_answers)]
                )

        pp = {
            "token_budgets": token_bugets,
            "probing_answers": probing_answers,
            "probing_accs": probing_accs,
            "changed_from_last": changed_from_last
        }

        with open(pp_fname, "w") as f:
            json.dump(pp, f, indent=2)


if __name__ == '__main__':

    main()
    done_counters = len(get_exp_logs("token-to-acc-probing/math500/"))
    while done_counters < 500:
        main()
        print('sleeping for 10 minutes')
        time.sleep(600)
        done_counters = len(get_exp_logs("token-to-acc-probing/math500/"))
