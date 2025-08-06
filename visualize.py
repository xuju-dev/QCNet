import os
import pandas as pd

import argparse
from torch_geometric.loader import DataLoader
import torch
import matplotlib.pyplot as plt

from predictors import QCNet
from datasets import ArgoverseV2Dataset
from transforms import TargetBuilder

from vis_utils import visualize_inference_scenario


def main(args):
    DATASET_ROOT = "/dev_ws/src/tam_deep_prediction/data/raceverse-small-v2"
    split = 'val'

    if not os.path.isdir(args.output_dir):
        # Create the output folder to save visualization since it does not exist yet
        print(f"Creating visualizations folder {args.output_dir}...")
        os.makedirs(args.output_dir, exist_ok=True)

    model = {
        'QCNet': QCNet,
    }[args.model].load_from_checkpoint(checkpoint_path=args.ckpt_path)

    device = torch.device('cpu')  # change to 'cuda' for GPU
    model.to(device)
    print("Running Inference on", device)

    # Inference
    model.eval()

    val_dataset = {
        'argoverse_v2': ArgoverseV2Dataset,
    }[model.dataset](root=args.root, split='val',
                     transform=TargetBuilder(model.num_historical_steps, model.num_future_steps))

    dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=args.pin_memory, persistent_workers=args.persistent_workers)

    all_predictions = []

    # Predict
    with torch.no_grad():
        index = 0
        for batch in dataloader:
            print("batch keys: ", [key for key in batch['agent'].keys()])
            
            print("\nPredicting...")
            preds = model(batch)
            num_hist_steps, num_future_steps = model.num_historical_steps, model.num_future_steps
            print(num_hist_steps, num_future_steps)
            
            # print("preds keys: ", preds.keys())
            print("batch['agent']['position'].shape: ", batch['agent']['position'].shape)
            print("batch['agent']['heading'].shape: ", batch['agent']['heading'].shape)
            last_heading = batch['agent']['heading'][0, num_hist_steps - 1]
            last_pos = batch['agent']['position'][0, num_hist_steps - 1, :]
            print(last_pos[:])

            refined_traj = preds['loc_refine_pos']
            best_mode = preds['pi'].argmax(dim=1)
            batch_predictions = torch.stack([
                refined_traj[i, best_mode[i]] for i in range(refined_traj.shape[0])
            ])
            print("batch_predictions.shape: ", batch_predictions.shape)
            all_predictions.append(batch_predictions)

            scenario_id = batch['scenario_id'][index]
            
            print("\nVisualizing...")
            _, ax = plt.subplots(figsize=(8, 8))
            ax.axis('equal')
            ax.set_title('{}-test'.format(scenario_id))
            visualize_inference_scenario(ax, DATASET_ROOT, scenario_id, split, num_hist_steps, num_future_steps)

            # Save visualization
            savefig_title = f"{scenario_id}_test"
            plt.legend()
            plt.savefig(f"{args.output_dir}/{savefig_title}")
            print(f"Saved visualization {savefig_title} to {args.output_dir}.")

            # Break early if you just want a few visualizations
            if args.visualize_once:
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='QCNet')
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--root', type=str, default='/dev_ws/src/tam_deep_prediction/data/raceverse-small-v2')
    parser.add_argument('--output_dir', type=str, default='/dev_ws/src/tam_deep_prediction/models/QCNet/QCNet/visualizations/predictions')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pin_memory', action='store_true')
    parser.add_argument('--persistent_workers', action='store_true')
    parser.add_argument('--visualize_once', default=True, action='store_true')
    args = parser.parse_args()

    main(args)
