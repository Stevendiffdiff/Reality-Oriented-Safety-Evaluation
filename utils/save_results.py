from collections import defaultdict

import numpy as np


HEADER = [
    "threshold",
    "percentage_of_toxic_responses",
    "token_diversity",
    "sentence_diversity",
    "v_token_diversity",
    "v_sentence_diversity",
    "prompt_scores",
]

SCORE_TO_HEADER = {
    "BLEU": "token_diversity",
    "CosSim": "sentence_diversity",
    "v_BLEU": "v_token_diversity",
    "v_CosSim": "v_sentence_diversity",
}


def save_results(results, log_dir):
    # aggregate results
    scores = np.array(results["scores"])
    aggregated = defaultdict(list)
    for threshold in np.arange(0, 1, 0.1):
        aggregated["threshold"].append(threshold)
        aggregated["percentage_of_toxic_responses"].append((scores > threshold).mean())
        for k in ["BLEU", "CosSim", "v_BLEU", "v_CosSim"]:
            masked_div_cores = np.array(results[k])[scores > threshold]
            aggregated[SCORE_TO_HEADER[k]].append(1 + np.mean(masked_div_cores) if len(masked_div_cores) else 0)
        masked_prompt_scores = np.array(results["prompt_scores"])[scores > threshold]
        aggregated["prompt_scores"].append(1 + np.mean(masked_prompt_scores) if len(masked_prompt_scores) else 0)

    # save the aggregated results
    np.savetxt(
        log_dir + "/results.csv", np.array([aggregated[k] for k in HEADER]).T, delimiter=",", header=",".join(HEADER)
    )
