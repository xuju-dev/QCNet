import os

import argparse
from torch_geometric.loader import DataLoader
import torch
import matplotlib.pyplot as plt

from predictors import QCNet
from datasets import ArgoverseV2Dataset
from transforms import TargetBuilder

from vis_utils import (
    visualize_inference_scenario,
    visualize_gt_trajectories,
    visualize_history,
    visualize_prediction,
    visualize_agents,
    visualize_qcnet_preds,
    AV2MapVisualizer
)


def main(args):
    DATASET_ROOT = "/dev_ws/src/tam_deep_prediction/data/raceverse-small-v2"
    split = 'val'

    model = {
        'QCNet': QCNet,
    }[args.model].load_from_checkpoint(checkpoint_path=args.ckpt_path)
    val_dataset = {
        'argoverse_v2': ArgoverseV2Dataset,
    }[model.dataset](root=args.root, split='val',
                     transform=TargetBuilder(model.num_historical_steps, model.num_future_steps))
    dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=args.pin_memory, persistent_workers=args.persistent_workers)

    if not os.path.isdir(args.output_dir):
        # Create the output folder to save visualization since it does not exist yet
        print(f"Creating visualizations folder {args.output_dir}...")
        os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cpu')  # change to 'cuda' for GPU
    model.to(device)
    print("Running Inference on", device)

    # Inference
    model.eval()

    with torch.no_grad():
        batch_size = args.batch_size
        for batch_index in range(batch_size):
            for batch in dataloader:
                scenario_id = batch['scenario_id'][batch_index]
                # print("batch keys: ", [key for key in batch.keys()])
                # print("batch['agent'] keys: ", [key for key in batch['agent'].keys()])

                print(f"\nPredicting scenario {scenario_id}...")
                preds = model(batch)
                num_hist_steps, num_future_steps = model.num_historical_steps, model.num_future_steps
                # print("preds keys: ", [key for key in preds.keys()])


                print(f"Visualizing scenario {scenario_id}...")
                _, ax = plt.subplots(figsize=(8, 8))
                ax.axis('equal')
                ax.set_title('{} batch_id: {}-{}'.format(scenario_id, batch_index, split))

                # Visualizing map
                AV2MapVisualizer(dataset_path=DATASET_ROOT).show_map(ax, split=split, seq_id=scenario_id)

                # Visualizing agents
                visualize_agents(ax, DATASET_ROOT, scenario_id, split)

                # Visualizing history
                # visualize_history(ax, batch, num_hist_steps=num_hist_steps)

                # Visualizing gt
                # visualize_gt_trajectories(ax, batch, num_history_steps=num_hist_steps, num_future_steps=num_future_steps)

                # Visualizing predictions
                last_pos = batch['agent']['position'][0, num_hist_steps - 1][:2]  # Get only x, y coordinates
                last_heading = batch['agent']['heading'][0, num_hist_steps - 1]
                visualize_prediction(ax, preds, last_pos, last_heading, batch_idx=batch_index)

                # visualize_inference_scenario(ax, DATASET_ROOT, scenario_id, split, num_hist_steps, num_future_steps)

                # Save visualization
                savefig_title = f"{scenario_id}_{batch_index}_{split}"
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
    parser.add_argument('--output_dir', type=str, default='/dev_ws/src/tam_deep_prediction/models/QCNet/QCNet/visualizations/inference')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pin_memory', action='store_true')
    parser.add_argument('--persistent_workers', action='store_true')
    parser.add_argument('--visualize_once', default=True, action='store_true')
    args = parser.parse_args()

    main(args)
