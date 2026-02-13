"""
Graph Transformer Module

Transforms preprocessed raw frame data into graph representations.
Handles node and edge creation with distance calculations.

Note: I have not tested whether this graph builder works correctly as he create graphs.py does.
"""

import logging
from typing import Any, Optional

from awpy.data import AREA_DIST_MATRIX, NAV
from awpy.analytics.nav import area_distance, find_closest_area

from datamodel.data_manager import DataManager


# Node type indices for PyTorch tensor compatibility
BOMB_NODE_INDEX = 6
BOMBSITE_A_NODE_INDEX = 7
BOMBSITE_B_NODE_INDEX = 8

NODE_TYPE_PLAYER_INDEX = 1000
NODE_TYPE_BOMB_INDEX = 900
NODE_TYPE_TARGET_INDEX = 1

# All node attribute keys
KEYS_PLAYER_LEVEL = (
    "x",
    "y",
    "z",
    "velocityX",
    "velocityY",
    "velocityZ",
    "viewX",
    "viewY",
    "hp",
    "armor",
    "activeWeapon",
    "totalUtility",
    "isAlive",
    "isDefusing",
    "isPlanting",
    "isReloading",
    "isInBombZone",
    "isInBuyZone",
    "equipmentValue",
    "equipmentValueFreezetimeEnd",
    "equipmentValueRoundStart",
    "cash",
    "cashSpendThisRound",
    "cashSpendTotal",
    "hasHelmet",
    "hasDefuse",
    "hasBomb",
)

KEYS_PER_NODE = KEYS_PLAYER_LEVEL + ("areaId", "nodeType")


def fill_keys(target: dict) -> dict:
    """Fill missing keys with 0 values for tensor compatibility."""
    empty_dict = {key: 0 for key in KEYS_PER_NODE}
    return empty_dict | target  # right dict takes precedence


def _distance_internal(map_name: str, area_a: int, area_b: int, logger=None) -> float:
    """
    Calculate distance between two map areas using area distance matrix or computation.
    
    Args:
        map_name: Name of the map
        area_a: Area ID for source
        area_b: Area ID for destination
        logger: Logger instance
        
    Returns:
        Geodesic distance between areas
    """
    # Use Area Distance Matrix if available (faster)
    area_a_str = str(area_a)
    area_b_str = str(area_b)
    
    if (
        map_name in AREA_DIST_MATRIX
        and area_a_str in AREA_DIST_MATRIX[map_name]
        and area_b_str in AREA_DIST_MATRIX[map_name][area_a_str]
    ):
        return AREA_DIST_MATRIX[map_name][area_a_str][area_b_str]["geodesic"]
    
    # Fallback: calculate distance
    if logger and len(AREA_DIST_MATRIX) > 0:
        logger.warning(f"Area matrix missing entry for area {area_a}")
    
    geodesic_path = area_distance(
        map_name=map_name, area_a=area_a, area_b=area_b, dist_type="geodesic"
    )
    return geodesic_path["distance"]


def distance_bombsites(
    map_name: str, nodes_data: dict[int, dict[str, Any]], logger=None
) -> tuple[dict[int, float], dict[int, float]]:
    """
    Compute distances from all nodes to both bombsites.
    
    Args:
        map_name: Name of the map
        nodes_data: Dict of node data indexed by node ID
        logger: Logger instance
        
    Returns:
        Tuple of (distances_to_bombsite_A, distances_to_bombsite_B)
    """
    if map_name not in NAV:
        raise ValueError("Map not found in navigation data.")
    
    # Initialize distance dicts
    closest_distances_A = {key: float("Inf") for key in nodes_data}
    closest_distances_B = {key: float("Inf") for key in nodes_data}
    
    # Iterate through all map areas to find bombsites
    for map_area_id in NAV[map_name]:
        map_area = NAV[map_name][map_area_id]
        
        if map_area["areaName"].startswith("BombsiteA"):
            for node_key in nodes_data:
                target_area = nodes_data[node_key]["areaId"]
                current_dist = _distance_internal(map_name, map_area_id, target_area)
                if current_dist < closest_distances_A[node_key]:
                    closest_distances_A[node_key] = current_dist
        
        elif map_area["areaName"].startswith("BombsiteB"):
            for node_key in nodes_data:
                target_area = nodes_data[node_key]["areaId"]
                current_dist = _distance_internal(map_name, map_area_id, target_area)
                if current_dist < closest_distances_B[node_key]:
                    closest_distances_B[node_key] = current_dist
    
    # Handle unreachable areas by interpolating from neighbors
    for dist_dict in [closest_distances_A, closest_distances_B]:
        keys = list(dist_dict.keys())
        for i, key in enumerate(keys):
            if dist_dict[key] == float("Inf"):
                prev_dist = None
                next_dist = None
                
                # Look backwards
                for j in range(i - 1, -1, -1):
                    if dist_dict[keys[j]] != float("Inf"):
                        prev_dist = dist_dict[keys[j]]
                        break
                
                # Look forward
                for j in range(i + 1, len(keys)):
                    if dist_dict[keys[j]] != float("Inf"):
                        next_dist = dist_dict[keys[j]]
                        break
                
                # Interpolate or use neighbor value
                if prev_dist is not None and next_dist is not None:
                    dist_dict[key] = (prev_dist + next_dist) / 2
                elif prev_dist is not None:
                    dist_dict[key] = prev_dist
                elif next_dist is not None:
                    dist_dict[key] = next_dist
                else:
                    raise ValueError(
                        f"Could not estimate bombsite distance for node {key}"
                    )
    
    return closest_distances_A, closest_distances_B


def preprocess_frame_to_graph(
    preprocessed_frame: dict[str, Any],
    dm: DataManager,
    logger=None,
) -> dict[str, Any]:
    """
    Transform a preprocessed frame into a graph representation.
    
    Args:
        preprocessed_frame: Single preprocessed frame from DataPreprocessor
        dm: DataManager instance for distance calculations
        logger: Logger instance
        
    Returns:
        Graph dict with graph_data, nodes_data, and edges_data
    """
    map_name = preprocessed_frame["map_name"]
    
    # Compile graph-level data
    graph_data = {
        **preprocessed_frame["round_data"],
        **preprocessed_frame["frame_data"],
        "tactic_used": preprocessed_frame["tactic"],
    }
    
    ### Create Node Data
    nodes_data = {}
    
    # Process each player
    for player_idx, player_data in enumerate(preprocessed_frame["players_data"]):
        node_data = dict(player_data)  # Copy player data
        
        # Find closest map area
        node_data["areaId"] = find_closest_area(
            map_name,
            point=[node_data[key] for key in ("x", "y", "z")],
            flat=False,
        )["areaId"]
        
        node_data["nodeType"] = NODE_TYPE_PLAYER_INDEX
        nodes_data[player_idx] = node_data
    
    # Add bomb node
    bomb_data = dict(preprocessed_frame["bomb_data"])
    bomb_data["areaId"] = find_closest_area(
        map_name,
        point=[bomb_data[key] for key in ("x", "y", "z")],
        flat=False,
    )["areaId"]
    bomb_data["nodeType"] = NODE_TYPE_BOMB_INDEX
    nodes_data[BOMB_NODE_INDEX] = bomb_data
    
    ### Create Edge Data
    edges_data = []
    
    # Compute distances to bombsites
    try:
        distance_A, distance_B = distance_bombsites(map_name, nodes_data, logger=logger)
    except ValueError as exc:
        if logger:
            logger.warning(f"Failed to compute bombsite distances: {exc}")
        raise
    
    # Add edges to bombsites
    for node_key in nodes_data:
        edges_data.append((node_key, BOMBSITE_A_NODE_INDEX, {"dist": distance_A[node_key]}))
        edges_data.append((node_key, BOMBSITE_B_NODE_INDEX, {"dist": distance_B[node_key]}))
    
    # Compute pairwise distances between all nodes
    node_keys = list(nodes_data.keys())
    for node_a in reversed(node_keys):
        for node_b in reversed(node_keys):
            # Skip self-loops
            if node_a == node_b:
                continue
            
            edges_data.append(
                (
                    node_a,
                    node_b,
                    {
                        "dist": _distance_internal(
                            map_name,
                            nodes_data[node_a]["areaId"],
                            nodes_data[node_b]["areaId"],
                            logger=logger,
                        )
                    },
                )
            )
    
    # Add bombsite target nodes (after distance calculations)
    nodes_data[BOMBSITE_A_NODE_INDEX] = {"nodeType": NODE_TYPE_TARGET_INDEX}
    nodes_data[BOMBSITE_B_NODE_INDEX] = {"nodeType": NODE_TYPE_TARGET_INDEX}
    
    # Fill all nodes with missing keys for tensor compatibility
    for node_key in nodes_data:
        nodes_data[node_key] = fill_keys(nodes_data[node_key])
    
    # Compile graph
    graph = {
        "graph_data": graph_data,
        "nodes_data": nodes_data,
        "edges_data": edges_data,
    }
    
    return graph


def preprocess_round_to_graphs(
    preprocessed_frames: list[dict[str, Any]],
    dm: DataManager,
    logger=None,
) -> list[dict[str, Any]]:
    """
    Transform all preprocessed frames in a round into graphs.
    
    Args:
        preprocessed_frames: List of preprocessed frames from DataPreprocessor
        dm: DataManager instance
        logger: Logger instance
        
    Returns:
        List of graph dicts
    """
    graphs = []
    error_count = 0
    
    for frame_idx, preprocessed_frame in enumerate(preprocessed_frames):
        try:
            graph = preprocess_frame_to_graph(preprocessed_frame, dm, logger=logger)
            graphs.append(graph)
        except Exception as e:
            error_count += 1
            if logger:
                logger.warning(f"Frame {frame_idx} skipped due to: {e}")
    
    if logger:
        logger.info(f"Transformed {len(graphs)} frames to graphs ({error_count} errors)")
    
    return graphs


class GraphTransformer:
    """Transforms preprocessed data into graph representations."""
    
    def __init__(self, dm: DataManager, logger=None):
        """
        Initialize graph transformer.
        
        Args:
            dm: DataManager instance for accessing demo data
            logger: Logger instance
        """
        self.dm = dm
        self.logger = logger or logging.getLogger(__name__)
    
    def transform_frame(self, preprocessed_frame: dict[str, Any]) -> dict[str, Any]:
        """
        Transform a single preprocessed frame to graph representation.
        
        Args:
            preprocessed_frame: Preprocessed frame dict
            
        Returns:
            Graph dict
        """
        return preprocess_frame_to_graph(preprocessed_frame, self.dm, logger=self.logger)
    
    def transform_round(
        self, preprocessed_frames: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Transform all preprocessed frames in a round to graphs.
        
        Args:
            preprocessed_frames: List of preprocessed frames
            
        Returns:
            List of graph dicts
        """
        return preprocess_round_to_graphs(preprocessed_frames, self.dm, logger=self.logger)
    
    def transform_all(
        self, all_preprocessed_rounds: dict[int, list[dict[str, Any]]]
    ) -> dict[int, list[dict[str, Any]]]:
        """
        Transform all preprocessed rounds to graphs.
        
        Args:
            all_preprocessed_rounds: Dict mapping round index to preprocessed frames
            
        Returns:
            Dict mapping round index to graph dicts
        """
        all_graphs = {}
        for round_idx, preprocessed_frames in all_preprocessed_rounds.items():
            all_graphs[round_idx] = self.transform_round(preprocessed_frames)
        return all_graphs


if __name__ == "__main__":
    import argparse
    import pickle
    import time
    from pathlib import Path
    from datamodel.data_manager import DataManager
    
    parser = argparse.ArgumentParser(
        description="Transform preprocessed frame data into graph representations."
    )
    parser.add_argument(
        "input_file",
        help="Path to preprocessed pickle file to transform",
    )
    parser.add_argument(
        "demo_path",
        help="Path to original demo file (needed for distance calculations)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/graphs_transformed",
        help="Output directory where transformed graphs will be saved (default: data/graphs_transformed)",
    )
    parser.add_argument(
        "--output-format",
        choices=["pickle", "none"],
        default="pickle",
        help="Output format: 'pickle' saves graphs, 'none' just prints summary (default: pickle)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging_level = getattr(logging, args.log_level)
    logging.basicConfig(
        level=logging_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)
    
    # Load preprocessed data
    logger.info(f"Loading preprocessed data from {args.input_file}")
    with open(args.input_file, "rb") as f:
        preprocessed_rounds = pickle.load(f)
    
    # Create transformer
    logger.info(f"Initializing transformer with demo: {args.demo_path}")
    dm = DataManager(Path(args.demo_path), logger=logger)
    transformer = GraphTransformer(dm, logger=logger)
    
    # Transform
    logger.info("Transforming preprocessed data to graphs...")
    all_graphs = transformer.transform_all(preprocessed_rounds)
    
    total_graphs = sum(len(graphs) for graphs in all_graphs.values())
    logger.info(f"Complete: Transformed {total_graphs} frames into graph representations")
    
    # Save if requested
    if args.output_format == "pickle":
        output_path = Path(args.output_dir) / dm.get_match_id() / "graphs.pkl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(all_graphs, f)
        logger.info(f"Saved to {output_path}")
