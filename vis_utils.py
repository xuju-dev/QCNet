from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import torch

from av2.map.map_api import ArgoverseStaticMap
from av2.datasets.motion_forecasting import scenario_serialization

from av2.datasets.motion_forecasting.viz.scenario_visualization import (
    _plot_actor_tracks,
    _plot_polylines
)


class AV2MapVisualizer:
    """ From SIMPL """
    def __init__(self, dataset_path):
        if dataset_path is not None:
            self.dataset_dir = dataset_path
        else:
            self.dataset_dir = '/dev_ws/src/tam_deep_prediction/data/raceverse-small-v2'

    def show_map(self,
                 ax,
                 split: str,
                 seq_id: str,
                 show_freespace=True):

        # ax.set_facecolor("grey")

        static_map_path = Path(self.dataset_dir + f"/{split}/{seq_id}" + f"/log_map_archive_{seq_id}.json")
        static_map = ArgoverseStaticMap.from_json(static_map_path)

        # ~ drivable area
        # print('num drivable areas: ', len(static_map.vector_drivable_areas),
        #       [x for x in static_map.vector_drivable_areas.keys()])
        for drivable_area in static_map.vector_drivable_areas.values():
            # ax.plot(drivable_area.xyz[:, 0], drivable_area.xyz[:, 1], color='grey', alpha=0.5, linestyle='--')
            ax.fill(drivable_area.xyz[:, 0], drivable_area.xyz[:, 1], color='grey', alpha=0.2)

        # ~ lane segments
        # print('num lane segs: ', len(static_map.vector_lane_segments),
        #       [x for x in static_map.vector_lane_segments.keys()])
        print('Num lanes: ', len(static_map.vector_lane_segments))
        for lane_segment in static_map.vector_lane_segments.values():
            # print('left pts: ', lane_segment.left_lane_boundary.xyz.shape,
            #       'right pts: ', lane_segment.right_lane_boundary.xyz.shape)

            if lane_segment.lane_type == 'VEHICLE':
                lane_clr = 'blue'
            elif lane_segment.lane_type == 'BIKE':
                lane_clr = 'green'
            elif lane_segment.lane_type == 'BUS':
                lane_clr = 'orange'
            else:
                assert False, "Wrong lane type"

            # if lane_segment.is_intersection:
            #     lane_clr = 'yellow'

            polygon = lane_segment.polygon_boundary
            ax.fill(polygon[:, 0], polygon[:, 1], color=lane_clr, alpha=0.1)

            for boundary in [lane_segment.left_lane_boundary, lane_segment.right_lane_boundary]:
                ax.plot(boundary.xyz[:, 0],
                        boundary.xyz[:, 1],
                        linewidth=1,
                        color='grey',
                        alpha=0.3)

            # cl = static_map.get_lane_segment_centerline(lane_segment.id)
            # ax.plot(cl[:, 0], cl[:, 1], linestyle='--', color='magenta', alpha=0.1)

        # ~ ped xing
        for pedxing in static_map.vector_pedestrian_crossings.values():
            edge = np.concatenate([pedxing.edge1.xyz, np.flip(pedxing.edge2.xyz, axis=0)])
            # plt.plot(edge[:, 0], edge[:, 1], color='orange', alpha=0.75)
            ax.fill(edge[:, 0], edge[:, 1], color='orange', alpha=0.2)
            # for edge in [ped_xing.edge1, ped_xing.edge2]:
            #     ax.plot(edge.xyz[:, 0], edge.xyz[:, 1], color='orange', alpha=0.5, linestyle='dotted')

    def show_map_clean(self,
                       ax,
                       split: str,
                       seq_id: str,
                       show_freespace=True):

        # ax.set_facecolor("grey")

        static_map_path = Path(self.dataset_dir + f"/{split}/{seq_id}" + f"/log_map_archive_{seq_id}.json")
        static_map = ArgoverseStaticMap.from_json(static_map_path,
                                                  overwrite_centerline=False)

        # ~ drivable area
        for drivable_area in static_map.vector_drivable_areas.values():
            # ax.plot(drivable_area.xyz[:, 0], drivable_area.xyz[:, 1], color='grey', alpha=0.5, linestyle='--')
            ax.fill(drivable_area.xyz[:, 0], drivable_area.xyz[:, 1], color='grey', alpha=0.2)

        # ~ lane segments
        print('Num lanes: ', len(static_map.vector_lane_segments))
        for lane_id, lane_segment in static_map.vector_lane_segments.items():
            # lane_clr = 'grey'
            polygon = lane_segment.polygon_boundary
            ax.fill(polygon[:, 0], polygon[:, 1], color='whitesmoke', alpha=1.0, edgecolor=None, zorder=0)

            # centerline
            centerline = lane_segment.centerline.xyz[:, 0:2]  # use xy
            ax.plot(centerline[:, 0], centerline[:, 1], alpha=0.1, color='grey', linestyle='dotted', zorder=1)

            # # lane boundary
            # for boundary, mark_type in [(lane_segment.left_lane_boundary.xyz, lane_segment.left_mark_type),
            #                             (lane_segment.right_lane_boundary.xyz, lane_segment.right_mark_type)]:

            #     clr = None
            #     width = 1.0
            #     if mark_type in [LaneMarkType.DASH_SOLID_WHITE,
            #                      LaneMarkType.DASHED_WHITE,
            #                      LaneMarkType.DOUBLE_DASH_WHITE,
            #                      LaneMarkType.DOUBLE_SOLID_WHITE,
            #                      LaneMarkType.SOLID_WHITE,
            #                      LaneMarkType.SOLID_DASH_WHITE]:
            #         clr = 'white'
            #         zorder = 3
            #         width = width
            #     elif mark_type in [LaneMarkType.DASH_SOLID_YELLOW,
            #                        LaneMarkType.DASHED_YELLOW,
            #                        LaneMarkType.DOUBLE_DASH_YELLOW,
            #                        LaneMarkType.DOUBLE_SOLID_YELLOW,
            #                        LaneMarkType.SOLID_YELLOW,
            #                        LaneMarkType.SOLID_DASH_YELLOW]:
            #         clr = 'gold'
            #         zorder = 4
            #         width = width * 1.1

            #     style = 'solid'
            #     if mark_type in [LaneMarkType.DASHED_WHITE,
            #                      LaneMarkType.DASHED_YELLOW,
            #                      LaneMarkType.DOUBLE_DASH_YELLOW,
            #                      LaneMarkType.DOUBLE_DASH_WHITE]:
            #         style = (0, (5, 10))  # loosely dashed
            #     elif mark_type in [LaneMarkType.DASH_SOLID_YELLOW,
            #                        LaneMarkType.DASH_SOLID_WHITE,
            #                        LaneMarkType.DOUBLE_SOLID_YELLOW,
            #                        LaneMarkType.DOUBLE_SOLID_WHITE,
            #                        LaneMarkType.SOLID_YELLOW,
            #                        LaneMarkType.SOLID_WHITE,
            #                        LaneMarkType.SOLID_DASH_WHITE,
            #                        LaneMarkType.SOLID_DASH_YELLOW]:
            #         style = 'solid'

            #    if (clr is not None) and (style is not None):
            #        ax.plot(boundary[:, 0],
            #                boundary[:, 1],
            #                color=clr,
            #                alpha=1.0,
            #                linewidth=width,
            #                linestyle=style,
            #                zorder=zorder)

        # ~ ped xing
        for pedxing in static_map.vector_pedestrian_crossings.values():
            edge = np.concatenate([pedxing.edge1.xyz, np.flip(pedxing.edge2.xyz, axis=0)])
            ax.fill(edge[:, 0], edge[:, 1], color='yellow', alpha=0.1, edgecolor=None)


def load_scenario_and_map(scenario_id: str, split: str, dataset_root: str) -> tuple:
    """
    Load scenario data and static map from a given scenario ID and split.

    Args:
        scenario_id (str): The scenario ID (e.g., '001-182cf688453e7cb4-11-abudhabi')
        split (str): The dataset split (e.g., 'train', 'val', 'test')
        dataset_root (str): Root path to the dataset (e.g., '/path/to/raceverse-small-v2')

    Returns:
        Tuple containing:
            - scenario (Scenario): Loaded scenario object
            - static_map (ArgoverseStaticMap): Loaded static map object
    """
    scenario_path = Path(dataset_root) / split / scenario_id

    # Find parquet file
    parquet_files = list(scenario_path.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {scenario_path}")
    parquet_path = parquet_files[0]
    # print(f"[INFO] Found scenario parquet at: {parquet_path}")

    # Map path
    static_map_path = scenario_path / f"log_map_archive_{scenario_id}.json"
    if not static_map_path.exists():
        raise FileNotFoundError(f"Map JSON not found: {static_map_path}")

    # Load
    scenario = scenario_serialization.load_argoverse_scenario_parquet(parquet_path)
    static_map = ArgoverseStaticMap.from_json(static_map_path)

    return scenario, static_map


def local_to_global(traj_local, last_pos, last_heading):
    """Convert [T, 2] trajectory from global to local using rotation and translation."""
    cos_h = torch.cos(last_heading)
    sin_h = torch.sin(last_heading)
    R = torch.tensor([[cos_h, -sin_h], [sin_h, cos_h]])

    return traj_local @ R.T + last_pos

def extract_predicted_traj(preds, last_pos, last_heading, batch_idx):
    """
    Extracts predicted trajectories and returns them as polylines.

    Args:
        preds: Dictionary containing model predictions.
    Returns:
        List of polylines, each polyline is a numpy array of shape [T, 2].
    """
    refined_traj = preds['loc_refine_pos']  # [28, A, T, 2]
    pi = preds['pi']  # [28, A] best mode (pred_traj) for each sample
    A, T, _ = refined_traj[batch_idx] .shape  # A: num_agents, T: number of steps

    # Get best mode for this batch sample (shape: scalar)
    best_modes = pi.argmax(dim=1)  # [28] best mode for each sample

    predicted_trajectories = torch.stack([refined_traj[batch_idx, m] for m in best_modes])  # best pred_traj per sample
    predicted_trajectories = local_to_global(predicted_trajectories, last_pos, last_heading)

    return predicted_trajectories


def extract_gt_traj(batch, batch_idx: int, step_start: int, step_end: int, agent_idx: int, all_agents: bool = False):
    pos = batch['agent']['position'][..., :2]  # [batch_size, 110, 2]
    valid_mask = batch['agent']['valid_mask'][agent_idx]  # [110] dim1 = true value if history
    predict_mask = batch['agent']['predict_mask'][agent_idx]  # [110] dim1 = true value if prediction

    valid_mask *= predict_mask
    valid_pos = pos[agent_idx][valid_mask]  # [60, 2]

    gt_pos = valid_pos[step_start:step_end]  # [num_steps, 2]

    # B, A, T, _ = target.shape
    # if batch_idx > B or agent_idx > A:
    #     raise IndexError(f"Batch index {batch_idx} or agent index {agent_idx} out of range: B={B}, A={A}")

    center = batch['agent']['position'][..., :2][agent_idx, step_start - 1][:2]  # first pos as origin
    angle = batch['agent']['heading'][agent_idx][step_start - 1]  # first heading as theta

    trajectories = []
    # if all_agents:
    #     print("Collecting all agents...")
    #     for a in range(A):
    #         traj = masked_target[batch_idx, a]  # [T, 2]
    #         polyline = local_to_global(traj, centers[batch_idx], angles[batch_idx])
    #         polyline = polyline.cpu().numpy()  # Convert to NumPy
    #         trajectories.append(polyline)     # [T, 2]
    # else:
    print(f"Collecting agent {agent_idx}...")
    polyline = local_to_global(gt_pos, center, angle)
    polyline = gt_pos.cpu().numpy()  # Convert to NumPy
    trajectories.append(polyline)     # [T, 2]

    return trajectories


def visualize_agents(ax, dataset_root, scenario_id, split):
    scenario, scenario_map = load_scenario_and_map(scenario_id, split, dataset_root)
    _plot_actor_tracks(ax, scenario, timestep=60)  # plot focal agent to timestep 20


def visualize_history(ax, batch, num_hist_steps, agent_idx=0):
    history_polylines = extract_gt_traj(batch=batch, batch_idx=0, step_start=0, step_end=num_hist_steps, agent_idx=agent_idx, all_agents=False)
    _plot_polylines(history_polylines, line_width=1, color='grey')
    pred_lines = plt.gca().lines[-1]
    pred_lines.set_label('History')


def visualize_gt_trajectories(ax, batch, num_history_steps, num_future_steps, agent_idx=0):
    end_timestep = num_history_steps + num_future_steps
    target_trajectories = extract_gt_traj(batch=batch, batch_idx=0, step_start=num_history_steps, step_end=end_timestep, agent_idx=agent_idx, all_agents=False)
    _plot_polylines(target_trajectories, line_width=1, color='green')
    pred_lines = plt.gca().lines[-1]
    pred_lines.set_label('Target')


def visualize_prediction(ax, predictions, last_pos, last_heading, batch_idx):
    """
    Visualize one sample from the batch and plot the predicted trajectories.

    Args:
        ax: plot reference
        predictions: model output
    """
    predicted_trajectories = extract_predicted_traj(predictions, last_pos, last_heading, batch_idx=batch_idx)
    _plot_polylines(predicted_trajectories, line_width=0.5)
    pred_lines = plt.gca().lines[-1]
    pred_lines.set_label('Prediction')


def visualize_inference_scenario(ax, batch, dataset_root, scenario_id, split, preds, num_history_steps, num_future_steps):
    # Visualizing map
    AV2MapVisualizer(dataset_path=dataset_root).show_map(ax, split=split, seq_id=scenario_id)

    # Visualizing agents
    visualize_agents(ax, dataset_root, scenario_id, split)

    # Visualizing history
    visualize_history(ax, dataset_root, scenario_id, split)

    # Visualizing gt
    visualize_gt_trajectories(ax, batch, dataset_root, scenario_id, split, num_history_steps, num_future_steps)

    # Visualizing predicted trajectories
    visualize_prediction(ax, preds)
    # print("Plotting predicted trajectories: DONE")


def visualize_qcnet_preds(batch, preds, mode='best', max_agents=4):
    """
    Visualize predicted trajectories for focal agents in batch.

    Args:
        batch: HeteroDataBatch from DataLoader.
        preds: dict from QCNet model forward pass.
        mode: 'best' for highest-prob mode, 'all' for all modes.
        max_agents: max number of focal agents to plot.

    """
    batch_agent = batch['agent']
    print("batch_agent['predict_mask'].shape: ", batch_agent['predict_mask'].shape)

    # Get focal agents mask (bool tensor)
    focal_mask = batch_agent.predict_mask  # shape: [num_agents]
    print("focal_mask.shape: ", focal_mask.shape)
    if focal_mask.sum() == 0:
        print("No focal agents to visualize in this batch.")
        return

    # Extract focal agents' indices
    focal_indices = torch.nonzero(focal_mask).squeeze(1)

    # Limit number of agents to visualize for clarity
    focal_indices = focal_indices[:max_agents]

    # Gather predictions and historical positions
    loc_refine_pos = preds['loc_refine_pos'][focal_indices]  # [N, 6, 40, 2]
    pi = preds['pi'][focal_indices]  # [N, 6]

    # Historical positions from batch (assuming shape [num_agents, hist_len, 2])
    # Adapt according to your dataset field names
    hist_pos = batch_agent.position[focal_indices][:, :batch_agent.num_historical_steps, :]  # [N, hist_len, 2]

    for i, agent_idx in enumerate(focal_indices):
        plt.figure(figsize=(6, 6))

        # Plot history (past trajectory)
        hist = hist_pos[i].cpu().numpy()
        plt.plot(hist[:, 0], hist[:, 1], 'k-', label='History')

        # Plot predictions
        if mode == 'best':
            best_mode = pi[i].argmax().item()
            traj = loc_refine_pos[i, best_mode].cpu().numpy()  # [40, 2]
            plt.plot(traj[:, 0], traj[:, 1], 'r-', label='Best predicted')
        elif mode == 'all':
            for m in range(loc_refine_pos.shape[1]):
                traj = loc_refine_pos[i, m].cpu().numpy()
                alpha = pi[i, m].item()  # use mode prob as alpha
                plt.plot(traj[:, 0], traj[:, 1], alpha=alpha, label=f'Mode {m}')
        else:
            raise ValueError("mode must be 'best' or 'all'")

        plt.scatter(hist[-1, 0], hist[-1, 1], c='black', s=50, marker='o')  # last observed point
