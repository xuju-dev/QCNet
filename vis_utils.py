from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import torch
import pytorch_lightning as pl
from hydra import compose, initialize
from hydra.utils import instantiate

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
    """Convert [T, 2] trajectory from local to global using rotation and translation."""
    cos_h = torch.cos(last_heading)
    sin_h = torch.sin(last_heading)
    R = torch.tensor([[cos_h, -sin_h], [sin_h, cos_h]])

    return traj_local @ R.T + last_pos

def extract_predicted_traj(preds, last_pos, last_heading):
    """
    Extracts predicted trajectories and returns them as polylines.

    Args:
        preds: Dictionary containing model predictions.
    Returns:
        List of polylines, each polyline is a numpy array of shape [T, 2].
    """
    refined_traj = preds['loc_refine_pos']
    best_mode = preds['pi'].argmax(dim=1)
    predicted_trajectories = torch.stack([
        refined_traj[i, best_mode[i]] for i in range(refined_traj.shape[0])
    ])
    print("predicted_trajectories.shape: ", predicted_trajectories.shape)
    predicted_trajectories = local_to_global(predicted_trajectories, last_pos, last_heading)

    return predicted_trajectories

def extract_targets(labels, batch_idx: int, agent_idx: int, all_agents: bool):
    target = labels['target']  # [B, A, 60, 2]
    t_mask = labels['target_mask']  # [B, A, 60]
    masked_target = target * t_mask.unsqueeze(-1)  # shape: [B, A, 60, 2]

    B, A, T, _ = target.shape
    if batch_idx > B or agent_idx > A:
        raise IndexError(f"Batch index {batch_idx} or agent index {agent_idx} out of range: B={B}, A={A}")

    centers = labels['origin']
    angles = labels['theta']

    trajectories = []
    if all_agents:
        # print("Collecting all agents...")
        for a in range(A):
            traj = masked_target[batch_idx, a]  # [T, 2]
            polyline = local_to_global(traj, centers[batch_idx], angles[batch_idx])
            polyline = polyline.cpu().numpy()  # Convert to NumPy
            trajectories.append(polyline)     # [T, 2]
    else:
        # print(f"Collecting agent {agent_idx}...")
        traj = masked_target[batch_idx, agent_idx]  # [T, 2]
        polyline = local_to_global(traj, centers[batch_idx], angles[batch_idx])
        polyline = polyline.cpu().numpy()  # Convert to NumPy
        trajectories.append(polyline)     # [T, 2]

    return trajectories


def visualize_agents(ax, dataset_root, scenario_id, split):
    scenario, scenario_map = load_scenario_and_map(scenario_id, split, dataset_root)
    _plot_actor_tracks(ax, scenario, timestep=60)


def visualize_history(ax, dataset_root, scenario_id, split):
    scenario, scenario_map = load_scenario_and_map(scenario_id, split, dataset_root)
    history_traj = []
    _plot_polylines(history_traj, line_width=1, color='grey')
    pred_lines = plt.gca().lines[-1]
    pred_lines.set_label('History')


def visualize_gt_trajectories(ax, dataset_root, scenario_id, split):
    scenario, scenario_map = load_scenario_and_map(scenario_id, split, dataset_root)
    target_trajectories = []
    # target_trajectories = extract_targets(labels, batch_idx=sample_idx, all_agents=all_agents_bool, agent_idx=agent_idx)
    _plot_polylines(target_trajectories, line_width=1, color='green')
    pred_lines = plt.gca().lines[-1]
    pred_lines.set_label('Target')


def visualize_prediction(ax, predictions):
    """
    Visualize one sample from the batch and plot the predicted trajectories.

    Args:
        ax: plot reference
        predictions: model output
    """
    predicted_trajectories = extract_predicted_traj(predictions)
    _plot_polylines(predicted_trajectories, line_width=0.5)
    pred_lines = plt.gca().lines[-1]
    pred_lines.set_label('Prediction')


def visualize_inference_scenario(ax, dataset_root, scenario_id, split, preds, num_history_steps, num_future_steps):
    # Visualizing map
    AV2MapVisualizer(dataset_path=dataset_root).show_map(ax, split=split, seq_id=scenario_id)

    # Visualizing agents
    visualize_agents(ax, dataset_root, scenario_id, split)
    
    # Visualizing history
    visualize_history(ax, dataset_root, scenario_id, split)

    # Visualizing gt
    # visualize_gt_trajectories(ax, DATASET_ROOT, scenario_id, split)
    
    # Visualizing predicted trajectories
    visualize_prediction(ax, preds)
    # print("Plotting predicted trajectories: DONE")
