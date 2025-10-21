import os
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from vis_utils import AV2MapVisualizer, load_scenario_and_map
from av2.datasets.motion_forecasting.viz.scenario_visualization import _plot_actor_tracks

# ========= Configuration =========
mode = "real"  # TODO: Change to visualize inference on sim or real dataset
model = "qcnet" 

assert mode in ["sim", "real"], "Mode should be either 'sim' or 'real'"

# ======== Set dataset path and split based on mode =========
if mode == "sim":
    # path to the original dataset
    dataset_path = "/dev_ws/src/tam_deep_prediction/data/raceverse-small-v2"  # sim
    split = 'test'
else:
    dataset_path = "/dev_ws/src/tam_deep_prediction/data/tum_av2"  # real
    split = 'test'

# ======== Set inference file paths based on model and mode =========
checkpoint_path = "/dev_ws/src/tam_deep_prediction/models/QCNet/QCNet/lightning_logs/version_63/checkpoints/epoch=63-step=137792.ckpt"
if mode == "sim":
    inference_file_path = "/dev_ws/src/tam_deep_prediction/models/QCNet/QCNet/submissions/qcnet_sim_inference_20251021_181541.parquet"
else:
    inference_file_path = "/dev_ws/src/tam_deep_prediction/models/QCNet/QCNet/submissions/qcnet_real_inference_20251021_184940.parquet"


viz_output_dir = f"/dev_ws/src/tam_deep_prediction/models/QCNet/QCNet/visualizations/inference/{model}_{mode}"
os.makedirs(viz_output_dir, exist_ok=True)

top_k = 3
t_hist = 20
t_fut = 40

# ========= Read inference results =========
preds_df = pd.read_parquet(inference_file_path)

scenario_ids = preds_df['scenario_id'].unique()
for scenario_id in scenario_ids:
    # print(f"Visualizing scenario {scenario_id}...")
    scenario_preds = preds_df[preds_df['scenario_id'] == scenario_id]
    best_prob = scenario_preds['probability'].max()
    
    sorted_preds = sorted(scenario_preds['probability'], reverse=True)  
    top_k_preds = sorted_preds[:top_k]

    # Some plot configs
    _, ax = plt.subplots(figsize=(8, 8))
    ax.axis('equal')
    ax.set_title('{}-{}-{}-{}'.format(scenario_id, 'Inference', model, mode))

    scenario, static_map = load_scenario_and_map(scenario_id, split, dataset_path)
    focal_id = scenario.focal_track_id
    preds_focal = scenario_preds[scenario_preds['track_id'] == focal_id]
    
    # Visualizing map and agents
    AV2MapVisualizer(dataset_path=dataset_path).show_map(ax, split=split, seq_id=scenario_id)
    _plot_actor_tracks(ax, scenario, timestep=20)  # only history

    # Get ground truth positions
    focal_track = next(t for t in scenario.tracks if t.track_id == scenario.focal_track_id)
    positions = np.array([state.position for state in focal_track.object_states])
    gt_future = positions[t_hist : t_hist + t_fut]
    gt_fut2end = positions[t_hist + t_fut: ]
    final_gt_pos = gt_future[-1, -1]

    # Plot ground truth future trajectory
    ax.plot(gt_future[:,0], gt_future[:,1], 'r--', label='Ground Truth Future')

    # Plot text box with best predicted probability
    ax.text(
        0.98, 0.98,
        f"Best Probability: {best_prob:.2f}",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(
            boxstyle="square,pad=0.4",
            fc="white", 
            ec="black", 
            lw=1
        )
    )
    
    # Plot top k predicted trajectories for the target agent
    for prob in top_k_preds:
        preds = scenario_preds[scenario_preds['probability'] == prob]
        preds_x = preds['predicted_trajectory_x'].values[0]
        preds_y = preds['predicted_trajectory_y'].values[0]
        
        ax.plot(preds_x, preds_y, linestyle='--', color='blue', linewidth=1)

        # Add probability text at the end of trajectory
        ax.text(
            preds_x[-1],
            preds_y[-1],
            f"{prob:.2f}",
            fontsize=7,
            color='blue',
            verticalalignment='bottom',
            horizontalalignment='right'
        )

    ax.legend()
    viz_save_path = Path(viz_output_dir) / f"{scenario_id}_inference_{mode}.png"
    plt.savefig(viz_save_path, dpi=300, bbox_inches="tight")
    print(f"Saved visualization to {viz_save_path}")