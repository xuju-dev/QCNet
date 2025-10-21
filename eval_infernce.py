import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from vis_utils import load_scenario_and_map
from metrics import minADE, minFDE, MR


# ========= 1. Load parquet files =========
mode = "sim"  # TODO: Change to evaluate on sim or real dataset
assert mode in ["sim", "real"], "Mode should be either 'sim' or 'real'"

if mode == "sim":
    # path to the original dataset
    dataset_path = "/dev_ws/src/tam_deep_prediction/data/raceverse-small-v2"  # sim
    inference_file_path = "/dev_ws/src/tam_deep_prediction/models/QCNet/QCNet/submissions/qcnet_sim_inference_20251021_181541.parquet"
else:
    dataset_path = "/dev_ws/src/tam_deep_prediction/data/tum_av2"  # real
    inference_file_path = "/dev_ws/src/tam_deep_prediction/models/QCNet/QCNet/submissions/qcnet_real_inference_20251021_184940.parquet"

pred_df = pd.read_parquet(inference_file_path)

split = 'test'
t_hist = 20
t_fut = 40

# ========= 2. Initialize metrics =========
metrics = {
    "MR": MR(),
    "minADE1": minADE(max_guesses=1),
    "minADE6": minADE(max_guesses=6),
    "minFDE1": minFDE(max_guesses=1),
    "minFDE6": minFDE(max_guesses=6),
}

# ========= 3. Iterate through scenarios and update metrics =========
for (scenario_id, track_id), group in pred_df.groupby(["scenario_id", "track_id"]):
    # Get gt scenario file path
    gt_path = os.path.join(dataset_path, split, scenario_id, f'scenario_{scenario_id}.parquet')
    if not os.path.exists(gt_path):
        print(f"[WARN] Missing GT for scenario {scenario_id} in {gt_path}, skipping.")
        continue

    # Load GT parquet for this scenario
    gt_scenario, static_map = load_scenario_and_map(scenario_id, split, dataset_path)

    # Get ground truth positions
    focal_track = next(t for t in gt_scenario.tracks if t.track_id == track_id)
    gt = np.array([state.position for state in focal_track.object_states])
    gt_future = gt[t_hist : t_hist + t_fut]

    # Get predictions and probabilities
    preds = []
    probs = []

    for _, row in group.iterrows():
        # Stack x and y for 6 predicted trajectories
        xy = np.stack(
            [np.array(row["predicted_trajectory_x"]), np.array(row["predicted_trajectory_y"])],
            axis=-1
        )  # [6, T, 2]
        preds.append(xy)
        probs.append(np.array(row["probability"]))  # [6]

    # Stack all into batch tensors
    preds = np.stack(preds, axis=0)  # [B, 6, T, 2]
    probs = np.stack(probs, axis=0)  # [B, 6]

    preds_out = {
        "y_hat": torch.tensor(preds, dtype=torch.float32).unsqueeze(0),
        "pi": torch.tensor(probs, dtype=torch.float32).unsqueeze(0),
    }

    gt_future = torch.tensor(gt_future, dtype=torch.float32).unsqueeze(0)
        
    # update metrics
    for m in metrics.values():
        m.update(preds_out, gt_future)

# ========= 4. Aggregate metrics =========
metrics = {name: m.compute().item() for name, m in metrics.items()}

# ========= 5. Print results =========
print("\n=== QCNet (Sim) Inference Evaluation Metrics ===")
for k, v in metrics.items():
    print(f"{k:8s}: {v}")