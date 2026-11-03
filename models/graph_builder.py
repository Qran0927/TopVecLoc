# -*- coding: utf-8 -*-
"""
Graph Builder:  OSM 

：
   → OSM （、POI、）
   → （PyTorch dict）， GNN 

（ + ）：

  ：
    ① intersection（）  —— 
    ② poi（）         —— 、、
    ③ area（/）     —— 、、

  ：
    ① road_edges（）           —— （）
    ② poi→intersection（） —— POI 
    ③ area→intersection（）—— 

？
  CNN  → （）
  ：、、
"""

import sys
from pathlib import Path

#  Python path 
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from osm.reader import OSMData, OSMNode, OSMWay
from osm.data import MapData, MapNode, MapLine, MapArea
from osm.parser import (
    filter_node, filter_way, filter_area,
    parse_node, parse_way, parse_area,
    match_to_group, Patterns, Groups,
)
from utils.geo import BoundaryBox, Projection


# ============================================================
# ： OSM 
# ============================================================
# OSM ：
#   OSMNode = （）
#   OSMWay  =  OSMNode （）
#   OSMRelation =  OSMWay （）
#
# ：
#    =  OSMNode
#    =  OSMWay 
# ============================================================

def extract_road_graph(
    osm: OSMData,
    bbox: Optional[BoundaryBox] = None,
) -> Tuple[Dict[int, int], List[Dict]]:
    """
     OSMData 。

    ：
      OSMWay ， OSMNode ：
        A: node_100 → node_200 → node_300 → node_400

       node， node ：
        A: node_100 → node_200 → node_300
        B: node_500 → node_200 → node_600
                      ↑ ！node_200 

    :
      osm:  OSM （ add_xy_to_nodes ）
      bbox: ， bbox 

    :
      node_to_idx: {OSMID →  0,1,2,...}
                   PyG ， OSM  ID
      edges_info: [{src, dst, way_id, way_type, segment_xy}, ...]
                  ，
    """

    # --- 1.1:  way ---
    # OSM ：、、、、...
    # ""
    # （/，）
    road_groups = {
        'road', 'path', 'cycleway', 'busway',
    }

    road_ways = []
    for way_id, way in osm.ways.items():
        #  way ，
        label = parse_way(way.tags)
        if label is None:
            # parse_way  None （）
            continue
        #  OSM （ highway=primary）
        group = match_to_group(label, Patterns.ways)
        if group in road_groups:
            #  bbox， aabb （）
            if bbox is not None:
                way_coords = np.stack([n.xy for n in way.nodes])
                way_min = way_coords.min(axis=0)
                way_max = way_coords.max(axis=0)
                #  bbox 
                # ： ↔  min ≤  max
                if not (np.all(way_min <= bbox.max_) and np.all(bbox.min_ <= way_max)):
                    continue
            road_ways.append((way_id, way, group))

    # ，
    if len(road_ways) == 0:
        return {}, []

    # --- 1.2:  ---
    # ：
    #   -  OSMNode  ≥2  → 
    #   -  1  → （，）
    #   -  0  → ，
    node_way_count = defaultdict(int)
    node_to_ways = defaultdict(list)

    for way_id, way, group in road_ways:
        for node in way.nodes:
            node_way_count[node.id_] += 1
            node_to_ways[node.id_].append(way_id)

    # 
    intersection_nodes = set()  # ：
    endpoint_nodes = set()      # ：
    for nid, count in node_way_count.items():
        if count >= 2:
            intersection_nodes.add(nid)
        elif count == 1:
            endpoint_nodes.add(nid)
    # count=0 （）

    # --- 1.3:  ---
    # PyG  edge_index  0 
    #  OSM ID（）
    node_to_idx = {}
    idx = 0
    # 
    for nid in intersection_nodes:
        node_to_idx[nid] = idx
        idx += 1
    # （）
    for nid in endpoint_nodes:
        if nid not in node_to_idx:
            node_to_idx[nid] = idx
            idx += 1

    # --- 1.4:  ---
    # ，/
    # ，
    edges_info = []

    for way_id, way, group in road_ways:
        # /（ way.nodes ）
        way_intersections = []
        for node in way.nodes:
            if node.id_ in node_to_idx:
                way_intersections.append(node)

        #  ≥2 ，
        for i in range(len(way_intersections) - 1):
            node_a = way_intersections[i]      # 
            node_b = way_intersections[i + 1]  # 

            #  node_a  node_b 
            # （，）
            all_nodes = way.nodes
            try:
                # 
                idx_a = all_nodes.index(node_a)
                idx_b = all_nodes.index(node_b)
                if idx_a > idx_b:
                    idx_a, idx_b = idx_b, idx_a
                # 
                segment_nodes = all_nodes[idx_a:idx_b + 1]
                segment_xy = np.stack([n.xy for n in segment_nodes])
            except (ValueError, IndexError):
                # ：
                segment_xy = np.stack([node_a.xy, node_b.xy])

            edges_info.append({
                'src': node_to_idx[node_a.id_],        # 
                'dst': node_to_idx[node_b.id_],        # 
                'way_id': way_id,                      #  OSM way ID（）
                'way_type': group,                     # （road/path/cycleway/busway）
                'segment_xy': segment_xy,              #  [n, 2]
            })

    return node_to_idx, edges_info


# ============================================================
# ：（ + POI + ）
# ============================================================
# ，：
#   - "" →  POI
#   - "" → /
#
#  POI ，
# ，
#  GNN ，"" POI 。
# ============================================================

def build_hetero_graph(
    osm: OSMData,
    bbox: BoundaryBox,
    proj: Projection,
) -> dict:
    """
    。

     vs ：
      ：（）
      ：（、POI、），
      ，

    :
      graph_dict:  torch.save  .pt 
        /、/
    """

    # --- 2.1: （ + ）---
    node_to_idx, edges_info = extract_road_graph(osm, bbox)

    # ：
    num_intersections = len(node_to_idx)
    intersection_coords = np.zeros((num_intersections, 2), dtype=np.float32)
    for osm_nid, idx in node_to_idx.items():
        intersection_coords[idx] = osm.nodes[osm_nid].xy

    # ： PyG edge_index  [2, num_edges]
    # （）
    road_edge_index = []
    road_edge_type = []  # （0~9， Groups.ways  10 ）
    for e in edges_info:
        # ：
        road_edge_index.append([e['src'], e['dst']])
        road_edge_index.append([e['dst'], e['src']])
        # （）
        road_edge_type.append(Groups.ways.index(e['way_type']))
        road_edge_type.append(Groups.ways.index(e['way_type']))

    # --- 2.2:  POI （）---
    # POI = Point of Interest，：
    #   ：、、、
    #   ：、、
    #   "" UAV 
    poi_coords = []      #  POI  (x, y) 
    poi_types = []       #  POI （0~32， Groups.nodes  33 ）

    for node_id, node in osm.nodes.items():
        # （）
        if node_id in node_to_idx:
            continue
        # filter_node： OSM 
        if not filter_node(node):
            continue
        #  bbox 
        if bbox is not None and not bbox.contains(node.xy):
            continue

        # ， POI
        label = parse_node(node.tags)
        if label is None:
            continue
        #  Groups.nodes（）
        group = match_to_group(label, Patterns.nodes)
        if group is None:
            continue  #  POI （）
        if group not in Groups.nodes:
            continue

        poi_coords.append(node.xy)
        poi_types.append(Groups.nodes.index(group))

    poi_coords = np.array(poi_coords, dtype=np.float32) if poi_coords else np.zeros((0, 2), dtype=np.float32)

    # --- 2.3: /（）---
    # （centroid）
    # ，
    # """"
    area_coords = []     # /
    area_types = []      # （0~6， Groups.areas  7 ）
    area_sizes = []      # （）， GNN """"

    for area_id, area in osm.ways.items():
        if not filter_area(area):
            continue
        # 
        all_xy = np.stack([n.xy for n in area.nodes])
        centroid = all_xy.mean(axis=0)

        #  bbox 
        if bbox is not None and not bbox.contains(centroid):
            continue

        label = parse_area(area.tags)
        if label is None:
            continue
        #  Groups.areas 
        group = match_to_group(label, Patterns.areas)
        if group is None:
            continue  #  area 
        if group not in Groups.areas:
            continue

        area_coords.append(centroid)
        area_types.append(Groups.areas.index(group))

        #  Shoelace（）
        # ：A = 0.5 * |Σ(x_i * y_{i+1} - x_{i+1} * y_i)|
        x = all_xy[:, 0]
        y = all_xy[:, 1]
        poly_area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        area_sizes.append(poly_area)

    area_coords = np.array(area_coords, dtype=np.float32) if area_coords else np.zeros((0, 2), dtype=np.float32)

    # --- 2.4:  ---
    # POI ""
    # ？GNN ：
    #   i ← (j + POI + )
    #   """"
    #
    #  50  POI  20 ， →
    # 50+20=70 ，
    # （ K=3 ），

    # POI → 
    poi_to_intersection_edges = []
    if len(poi_coords) > 0 and num_intersections > 0:
        for i, pc in enumerate(poi_coords):
            #  POI 
            dists = np.linalg.norm(intersection_coords - pc, axis=1)
            nearest = int(np.argmin(dists))  # 
            dist = dists[nearest]
            # [POI, , （）]
            poi_to_intersection_edges.append([i, nearest, dist])

    # / → 
    area_to_intersection_edges = []
    if len(area_coords) > 0 and num_intersections > 0:
        for i, ac in enumerate(area_coords):
            dists = np.linalg.norm(intersection_coords - ac, axis=1)
            nearest = int(np.argmin(dists))
            dist = dists[nearest]
            area_to_intersection_edges.append([i, nearest, dist])

    # --- 2.5:  ---
    #  GNN （ Embedding ）
    # ：
    #   （）
    #   ：=4 ，UAV 
    intersection_features = compute_intersection_features(
        num_intersections, edges_info, intersection_coords,
        poi_coords, area_coords,
    )

    # --- 2.6:  ---
    #  PyTorch ， dtype ：
    #   float32 — 、
    #   long    — （ Embedding ）
    graph = {
        'intersection': {
            'num_nodes': num_intersections,          # N 
            'coords': torch.from_numpy(intersection_coords),  # [N, 2] 

            # （ GNN  x）
            'degree': torch.tensor(
                intersection_features['degree'], dtype=torch.float32
            ),  # [N] （2=, 3=T, 4=）
            'road_type_dist': torch.tensor(
                intersection_features['road_type_dist'], dtype=torch.float32
            ),  # [N, 10] 
            'avg_segment_len': torch.tensor(
                intersection_features['avg_segment_len'], dtype=torch.float32
            ),  # [N] （）
            'poi_density': torch.tensor(
                intersection_features['poi_density'], dtype=torch.float32
            ),  # [N] 50m  POI 
            'building_density': torch.tensor(
                intersection_features['building_density'], dtype=torch.float32
            ),  # [N] 50m 
        },
        'poi': {
            'num_nodes': len(poi_coords),             # M  POI
            'coords': torch.from_numpy(poi_coords),   # [M, 2]
            'types': torch.tensor(poi_types, dtype=torch.long),  # [M] POI 
        },
        'area': {
            'num_nodes': len(area_coords),            # K /
            'coords': torch.from_numpy(area_coords),  # [K, 2]
            'types': torch.tensor(area_types, dtype=torch.long),  # [K] 
            'area_size': torch.tensor(
                area_sizes, dtype=torch.float32
            ) if area_sizes else torch.zeros(0, dtype=torch.float32),  # [K] 
        },
        'road_edges': {
            'edge_index': torch.tensor(
                road_edge_index, dtype=torch.long
            ).t().contiguous() if road_edge_index else torch.zeros((2, 0), dtype=torch.long),
            # .t().contiguous()  [E, 2]  [2, E]（PyG ）
            'edge_type': torch.tensor(
                road_edge_type, dtype=torch.long
            ) if road_edge_type else torch.zeros(0, dtype=torch.long),
            # ：（ GNN """"）
            'edge_length': torch.tensor(
                # 
                [_compute_segment_length(e['segment_xy']) for e in edges_info
                 for _ in (0, 1)],
                dtype=torch.float32
            ) if edges_info else torch.zeros(0, dtype=torch.float32),
        },
        'poi_to_intersection': {
            'edge_index': torch.tensor(
                [[e[0], e[1]] for e in poi_to_intersection_edges], dtype=torch.long
            ).t().contiguous() if poi_to_intersection_edges else torch.zeros((2, 0), dtype=torch.long),
            'edge_dist': torch.tensor(
                [e[2] for e in poi_to_intersection_edges], dtype=torch.float32
            ) if poi_to_intersection_edges else torch.zeros(0, dtype=torch.float32),
        },
        'area_to_intersection': {
            'edge_index': torch.tensor(
                [[e[0], e[1]] for e in area_to_intersection_edges], dtype=torch.long
            ).t().contiguous() if area_to_intersection_edges else torch.zeros((2, 0), dtype=torch.long),
            'edge_dist': torch.tensor(
                [e[2] for e in area_to_intersection_edges], dtype=torch.float32
            ) if area_to_intersection_edges else torch.zeros(0, dtype=torch.float32),
        },
        # 
        'bbox_min': torch.from_numpy(bbox.min_.astype(np.float32)),  # [2] 
        'bbox_max': torch.from_numpy(bbox.max_.astype(np.float32)),  # [2] 
        'num_classes': {                  # 
            'ways': len(Groups.ways),      # （=10）
            'nodes': len(Groups.nodes),    # POI （=33）
            'areas': len(Groups.areas),    # （=7）
        },
    }

    return graph


# ============================================================
# ：
# ============================================================
#  GNN ""。
#
# ：
#    GNN ，
#   /，
#    GNN 。
#
#  UAV ？
#   =4 →  → UAV ""
#   =3 → T → UAV ""
#    → UAV ""
#   POI → UAV ""
# ============================================================

def _compute_segment_length(segment_xy: np.ndarray) -> float:
    """（）"""
    if len(segment_xy) < 2:
        return 0.0
    diffs = np.diff(segment_xy, axis=0)   # 
    return float(np.sum(np.linalg.norm(diffs, axis=1)))  # 


def compute_intersection_features(
    num_intersections: int,
    edges_info: List[Dict],
    intersection_coords: np.ndarray,
    poi_coords: np.ndarray,
    area_coords: np.ndarray,
) -> Dict:
    """
    。

    ：
      degree[i]           →  (0~N)
                             i （）
                            2= / 3=T / 4= / ≥5=
      road_type_dist[i]   →  (10)
                            （road/path/cycleway/busway ）
      avg_segment_len[i]  →  ()
                            ，
      poi_density[i]      → 
                            50m  POI 
      building_density[i] → 
                            50m 

     GNN : [deg(1) + road_type(10) + seg_len(1) + poi(1) + building(1)] = 14
    """

    num_road_types = len(Groups.ways)  # =10 

    # ---  ---
    degree = np.zeros(num_intersections, dtype=np.float32)
    road_type_counts = np.zeros((num_intersections, num_road_types), dtype=np.float32)
    segment_lengths = [[] for _ in range(num_intersections)]

    # --- ， ---
    for e in edges_info:
        src = e['src']
        dst = e['dst']
        way_type = e['way_type']
        seg_len = _compute_segment_length(e['segment_xy'])

        # ：src  dst  +1
        degree[src] += 1
        degree[dst] += 1

        # 
        type_idx = Groups.ways.index(way_type)
        road_type_counts[src, type_idx] += 1
        road_type_counts[dst, type_idx] += 1

        # （）
        segment_lengths[src].append(seg_len)
        segment_lengths[dst].append(seg_len)

    # ---  ---
    avg_segment_len = np.array([
        np.mean(ls) if ls else 0.0 for ls in segment_lengths
    ], dtype=np.float32)

    # ---  ---
    #  50 （，=）
    #  50 ？ UAV 
    # 50m  ≈ ，
    RADIUS = 50.0
    poi_density = np.zeros(num_intersections, dtype=np.float32)
    building_density = np.zeros(num_intersections, dtype=np.float32)

    if len(poi_coords) > 0:
        for i in range(num_intersections):
            dists = np.linalg.norm(poi_coords - intersection_coords[i], axis=1)
            poi_density[i] = np.sum(dists < RADIUS)

    if len(area_coords) > 0:
        for i in range(num_intersections):
            dists = np.linalg.norm(area_coords - intersection_coords[i], axis=1)
            building_density[i] = np.sum(dists < RADIUS)

    return {
        'degree': degree,
        'road_type_dist': road_type_counts,
        'avg_segment_len': avg_segment_len,
        'poi_density': poi_density,
        'building_density': building_density,
    }


def compute_area_features(osm: OSMData, bbox: BoundaryBox) -> Dict:
    """
    /（，）。

    :
      sizes[i] — （）， Shoelace 
    """
    sizes = []

    for area_id, way in osm.ways.items():
        from osm.parser import filter_area, parse_area, match_to_group, Patterns
        if not filter_area(way):
            continue
        all_xy = np.stack([n.xy for n in way.nodes])
        centroid = all_xy.mean(axis=0)
        if not bbox.contains(centroid):
            continue

        # Shoelace 
        x = all_xy[:, 0]
        y = all_xy[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        sizes.append(area)

    return {'sizes': sizes}


# ============================================================
# ： —— 
# ============================================================
#  batch_build_graphs.py （ R-tree ）
#  info.csv， .pt
#
# ： OSM ，
#  batch_build_graphs.py（R-tree ）
# ============================================================

def build_graphs_for_city(
    osm_path: Path,
    info_csv_path: Path,
    output_dir: Path,
):
    """
    。

    :
      osm_path:       .osm 
      info_csv_path: info.csv （）
      output_dir:    （ .pt）
    """
    import csv

    output_dir.mkdir(parents=True, exist_ok=True)

    # ===  OSM（， OSM ）===
    print(f" OSM: {osm_path}")
    osm = OSMData.from_file(osm_path)
    print(f"  {len(osm.nodes)} nodes, {len(osm.ways)} ways, {len(osm.relations)} relations")

    # ===  info.csv ===
    # : id, uav_name, map_name, uav_long, uav_lat, map_long, map_lat,
    #       tile_size_meters, pixel_per_meter, u, v, yaw, dis
    with open(info_csv_path, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)

    header = rows[0]
    data_rows = rows[1:]
    print(f": {len(data_rows)}")

    stats = {'nodes': 0, 'pois': 0, 'areas': 0, 'edges': 0}

    for row in data_rows:
        # 
        sample_id = row[0]
        map_name = row[2]              # （ "00001.npy"）
        map_long = float(row[5])       # 
        map_lat = float(row[6])        # 
        tile_size_meters = float(row[7])  # （）
        half_size_meters = tile_size_meters / 2

        #  bbox（）
        prior_latlon = np.array([map_lat, map_long])
        proj = Projection(*prior_latlon)
        center = proj.project(prior_latlon)
        bbox = BoundaryBox(center, center) + half_size_meters

        #  OSM 
        osm.add_xy_to_nodes(proj)

        # 
        graph = build_hetero_graph(osm, bbox, proj)

        # 
        stats['nodes'] += graph['intersection']['num_nodes']
        stats['pois'] += graph['poi']['num_nodes']
        stats['areas'] += graph['area']['num_nodes']
        stats['edges'] += graph['road_edges']['edge_index'].shape[1]

        #  .pt（ .npy ）
        pt_name = map_name.replace('.npy', '.pt')
        torch.save(graph, output_dir / pt_name)

        if int(sample_id) % 1000 == 0:
            print(f"   {sample_id} ...")

    print(f"\n！：")
    print(f"  : {stats['nodes'] / len(data_rows):.1f}")
    print(f"   POI : {stats['pois'] / len(data_rows):.1f}")
    print(f"  : {stats['areas'] / len(data_rows):.1f}")
    print(f"  :   {stats['edges'] / len(data_rows):.1f}")


# ============================================================
# ：
# ============================================================

if __name__ == '__main__':
    import sys

    #  SanFrancisco 
    root = Path('${UAV2MAP_ROOT}')
    city = 'SanFrancisco'

    osm_path = root / city / f'{city}.osm'
    info_path = root / city / 'info.csv'

    if not osm_path.exists():
        print(f"OSM : {osm_path}")
        sys.exit(1)

    #  OSM
    osm = OSMData.from_file(osm_path)
    print(f" {city}.osm: {len(osm.nodes)} nodes, {len(osm.ways)} ways")

    # 
    import csv
    with open(info_path, 'r') as f:
        rows = list(csv.reader(f))
    row = rows[1]  # （）
    map_lat, map_long = float(row[6]), float(row[5])
    tile_size_meters = float(row[7])
    half_size_meters = tile_size_meters / 2

    #  bbox
    proj = Projection(map_lat, map_long)
    center = proj.project(np.array([map_lat, map_long]))
    bbox = BoundaryBox(center, center) + half_size_meters

    # 
    osm.add_xy_to_nodes(proj)

    # 
    graph = build_hetero_graph(osm, bbox, proj)

    print(f"\n=====  =====")
    print(f": {graph['intersection']['num_nodes']}")
    print(f"POI : {graph['poi']['num_nodes']}")
    print(f": {graph['area']['num_nodes']}")
    print(f":   {graph['road_edges']['edge_index'].shape[1]}")
    print(f"POI→: {graph['poi_to_intersection']['edge_index'].shape[1]}")
    print(f"→: {graph['area_to_intersection']['edge_index'].shape[1]}")

    # 
    inter = graph['intersection']
    print(f"\n=====  =====")
    print(f": min={inter['degree'].min():.0f}, max={inter['degree'].max():.0f}, mean={inter['degree'].mean():.1f}")
    print(f"  : =2(), =3(T), =4(), ≥5()")

    # 
    deg = inter['degree'].numpy().astype(int)
    from collections import Counter
    deg_counts = Counter(deg)
    print(f"  : {dict(sorted(deg_counts.items()))}")

    print(f"\n (3):")
    way_names = Groups.ways
    for i in range(min(3, inter['num_nodes'])):
        dist = inter['road_type_dist'][i].numpy()
        types_str = ", ".join(
            f"{way_names[j]}={int(dist[j])}" for j in range(len(way_names)) if dist[j] > 0
        )
        print(f"  {i}: degree={int(inter['degree'][i].numpy())}, "
              f"seg_len={inter['avg_segment_len'][i]:.1f}m, types=[{types_str}]")

    print(f"\n: min={graph['area']['area_size'].min():.0f} m², "
          f"max={graph['area']['area_size'].max():.0f} m², "
          f"mean={graph['area']['area_size'].mean():.0f} m²")

    print(f"\n: min={graph['road_edges']['edge_length'].min():.1f} m, "
          f"max={graph['road_edges']['edge_length'].max():.1f} m, "
          f"mean={graph['road_edges']['edge_length'].mean():.1f} m")
