"""
 (.pt)
：， bbox 

:
  python batch_build_graphs.py              # 
  python batch_build_graphs.py --city Paris # 
  python batch_build_graphs.py --max 100    # 100()
"""

import sys
from pathlib import Path
_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import csv
import time
import argparse
import numpy as np
import torch
import rtree
from collections import defaultdict
from tqdm import tqdm

from osm.reader import OSMData
from osm.parser import (
    filter_node, filter_way, filter_area,
    parse_node, parse_way, parse_area,
    match_to_group, Patterns, Groups,
)
from utils.geo import BoundaryBox, Projection
from models.graph_builder import (
    _compute_segment_length,
    compute_intersection_features,
)


def batch_build_city(osm_path, info_csv_path, output_dir, max_samples=None):
    """（）"""

    output_dir.mkdir(parents=True, exist_ok=True)
    road_groups = {'road', 'path', 'cycleway', 'busway'}

    # =====  1:  +  () =====
    t0 = time.time()
    osm = OSMData.from_file(osm_path)
    print(f"OSM: {len(osm.nodes)} nodes, {len(osm.ways)} ways ({time.time()-t0:.1f}s)")

    t1 = time.time()
    print("...")

    # ： / POI / 
    way_data = {}   # wid -> {'group': str, 'type_idx': int, 'geo_bbox': tuple}
    poi_data = {}   # nid -> {'group': str, 'type_idx': int, 'geo': ndarray}
    area_data = {}  # wid -> {'group': str, 'type_idx': int, 'geo_bbox': tuple, 'nids': list}

    idx_ways = rtree.index.Index()
    idx_pois = rtree.index.Index()
    idx_areas = rtree.index.Index()

    for wid, way in osm.ways.items():
        # 
        label = parse_way(way.tags)
        if label is not None:
            group = match_to_group(label, Patterns.ways)
            if group in road_groups:
                lats = [n.geo[0] for n in way.nodes]
                lons = [n.geo[1] for n in way.nodes]
                way_data[wid] = {
                    'group': group,
                    'type_idx': Groups.ways.index(group),
                    'geo_bbox': (min(lons), min(lats), max(lons), max(lats)),
                }
                idx_ways.insert(wid, way_data[wid]['geo_bbox'])
                continue

        # （ Group.areas ， fallback ）
        if filter_area(way):
            label = parse_area(way.tags)
            if label is not None:
                group = match_to_group(label, Patterns.areas)
                #  Groups.areas （ path= parse_area ）
                if group is not None and group in Groups.areas:
                    lats = [n.geo[0] for n in way.nodes]
                    lons = [n.geo[1] for n in way.nodes]
                    bbox = (min(lons), min(lats), max(lons), max(lats))
                    area_data[wid] = {
                        'group': group,
                        'type_idx': Groups.areas.index(group),
                        'geo_bbox': bbox,
                        'nids': [n.id_ for n in way.nodes],
                    }
                    idx_areas.insert(wid, bbox)

    for nid, node in osm.nodes.items():
        if not filter_node(node):
            continue
        label = parse_node(node.tags)
        if label is None:
            continue
        group = match_to_group(label, Patterns.nodes)
        #  Groups.nodes （ fallback ）
        if group is not None and group in Groups.nodes:
            poi_data[nid] = {
                'group': group,
                'type_idx': Groups.nodes.index(group),
                'geo': node.geo,
            }
            lat, lon = node.geo
            idx_pois.insert(nid, (lon, lat, lon, lat))

    print(f"  :{len(way_data)} POI:{len(poi_data)} :{len(area_data)} ({time.time()-t1:.1f}s)")

    # =====  2:  =====
    with open(info_csv_path, 'r') as f:
        rows = list(csv.reader(f))
    data_rows = rows[1:]
    if max_samples:
        data_rows = data_rows[:max_samples]

    n = len(data_rows)
    stats = {'nodes': 0, 'pois': 0, 'areas': 0, 'edges': 0, 'empty': 0}

    for idx, row in enumerate(tqdm(data_rows, desc=f"  {osm_path.parent.name}")):
        map_name = row[2]
        map_long = float(row[5])
        map_lat = float(row[6])
        tile_size_meters = float(row[7])
        half_size_meters = tile_size_meters / 2

        #  + bbox
        proj = Projection(map_lat, map_long)
        center = proj.project(np.array([map_lat, map_long]))
        bbox_xy = BoundaryBox(center, center) + half_size_meters
        bbox_geo = proj.unproject(bbox_xy)
        geo_min, geo_max = bbox_geo.min_[::-1], bbox_geo.max_[::-1]
        geo_query = (*geo_min, *geo_max)

        # （ R-tree ）
        cand_way_ids = set(idx_ways.intersection(geo_query))

        cand_poi_ids = set(idx_pois.intersection(geo_query))
        cand_area_ids = set(idx_areas.intersection(geo_query))

        # 
        needed_nids = set()
        for wid in cand_way_ids:
            for node in osm.ways[wid].nodes:
                needed_nids.add(node.id_)
        for wid in cand_area_ids:
            for nid in area_data[wid]['nids']:
                needed_nids.add(nid)

        for nid in needed_nids:
            osm.nodes[nid].xy = proj.project(osm.nodes[nid].geo[np.newaxis])[0]
        for nid in cand_poi_ids:
            osm.nodes[nid].xy = proj.project(osm.nodes[nid].geo[np.newaxis])[0]

        # 
        graph = _build_graph_fast(
            osm, cand_way_ids, way_data,
            cand_poi_ids, poi_data,
            cand_area_ids, area_data,
            bbox_xy, road_groups,
        )

        # 
        pt_name = map_name.replace('.npy', '.pt')
        torch.save(graph, output_dir / pt_name)

        # 
        ni = graph['intersection']['num_nodes']
        stats['nodes'] += ni
        stats['pois'] += graph['poi']['num_nodes']
        stats['areas'] += graph['area']['num_nodes']
        stats['edges'] += graph['road_edges']['edge_index'].shape[1]
        if ni == 0:
            stats['empty'] += 1

    print(f"   ({time.time()-t0:.0f}s) "
          f"={stats['nodes']/n:.1f} POI={stats['pois']/n:.1f} "
          f"={stats['areas']/n:.1f} ={stats['edges']/n:.1f} ={stats['empty']}")
    return stats


def _build_graph_fast(osm, way_ids, way_data, poi_ids, poi_data, area_ids, area_data, bbox, road_groups):
    """"""

    # ---  ---
    node_way_count = defaultdict(int)
    filtered_ways = []

    for wid in way_ids:
        way = osm.ways[wid]
        wd = way_data[wid]
        way_coords = np.stack([n.xy for n in way.nodes])
        way_min = way_coords.min(axis=0)
        way_max = way_coords.max(axis=0)
        if not (np.all(way_min <= bbox.max_) and np.all(bbox.min_ <= way_max)):
            continue
        filtered_ways.append((wid, way, wd))
        for node in way.nodes:
            node_way_count[node.id_] += 1

    node_to_idx = {}
    idx_cnt = 0
    for wid, way, wd in filtered_ways:
        for node in way.nodes:
            if node.id_ not in node_to_idx:
                node_to_idx[node.id_] = idx_cnt
                idx_cnt += 1

    num_inter = len(node_to_idx)
    inter_coords = np.zeros((num_inter, 2), dtype=np.float32)
    for nid, idx in node_to_idx.items():
        inter_coords[idx] = osm.nodes[nid].xy

    # ---  ---
    edges_info = []
    for wid, way, wd in filtered_ways:
        way_inters = [n for n in way.nodes if n.id_ in node_to_idx]
        for i in range(len(way_inters) - 1):
            a, b = way_inters[i], way_inters[i+1]
            try:
                ia, ib = way.nodes.index(a), way.nodes.index(b)
                if ia > ib:
                    ia, ib = ib, ia
                seg_xy = np.stack([n.xy for n in way.nodes[ia:ib+1]])
            except (ValueError, IndexError):
                seg_xy = np.stack([a.xy, b.xy])
            edges_info.append({
                'src': node_to_idx[a.id_], 'dst': node_to_idx[b.id_],
                'way_id': wid, 'way_type': wd['group'], 'segment_xy': seg_xy,
            })

    # 
    road_ei, road_et, road_el = [], [], []
    for e in edges_info:
        for _ in (0, 1):
            road_ei.append([e['src'], e['dst']] if _ == 0 else [e['dst'], e['src']])
            road_et.append(way_data[e['way_id']]['type_idx'])
            road_el.append(_compute_segment_length(e['segment_xy']))
        # Keep first direction's edge_info for computation below
    # Fix: save info for both directions but once for features
    edge_info_for_features = []
    seen_pairs = set()
    for e in edges_info:
        pair = (min(e['src'], e['dst']), max(e['src'], e['dst']))
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            edge_info_for_features.append(e)

    # --- POI ---
    poi_coords, poi_types = [], []
    for nid in poi_ids:
        if nid in node_to_idx:
            continue
        node = osm.nodes[nid]
        if not bbox.contains(node.xy):
            continue
        poi_coords.append(node.xy)
        poi_types.append(poi_data[nid]['type_idx'])
    poi_coords = np.array(poi_coords, dtype=np.float32) if poi_coords else np.zeros((0, 2), dtype=np.float32)

    # ---  ---
    area_coords, area_types, area_sizes = [], [], []
    for wid in area_ids:
        if wid not in area_data:
            continue
        way = osm.ways[wid]
        all_xy = np.stack([n.xy for n in way.nodes])
        centroid = all_xy.mean(axis=0)
        if not bbox.contains(centroid):
            continue
        area_coords.append(centroid)
        area_types.append(area_data[wid]['type_idx'])
        x, y = all_xy[:, 0], all_xy[:, 1]
        poly_area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        area_sizes.append(poly_area)
    area_coords = np.array(area_coords, dtype=np.float32) if area_coords else np.zeros((0, 2), dtype=np.float32)

    # ---  ---
    poi_to_inter, area_to_inter = [], []
    if len(poi_coords) > 0 and num_inter > 0:
        for i, pc in enumerate(poi_coords):
            d = np.linalg.norm(inter_coords - pc, axis=1)
            j = int(np.argmin(d))
            poi_to_inter.append([i, j, float(d[j])])
    if len(area_coords) > 0 and num_inter > 0:
        for i, ac in enumerate(area_coords):
            d = np.linalg.norm(inter_coords - ac, axis=1)
            j = int(np.argmin(d))
            area_to_inter.append([i, j, float(d[j])])

    # ---  ---
    inter_feats = compute_intersection_features(
        num_inter, edge_info_for_features, inter_coords, poi_coords, area_coords,
    )

    return {
        'intersection': {
            'num_nodes': num_inter,
            'coords': torch.from_numpy(inter_coords),
            'degree': torch.tensor(inter_feats['degree'], dtype=torch.float32),
            'road_type_dist': torch.tensor(inter_feats['road_type_dist'], dtype=torch.float32),
            'avg_segment_len': torch.tensor(inter_feats['avg_segment_len'], dtype=torch.float32),
            'poi_density': torch.tensor(inter_feats['poi_density'], dtype=torch.float32),
            'building_density': torch.tensor(inter_feats['building_density'], dtype=torch.float32),
        },
        'poi': {
            'num_nodes': len(poi_coords),
            'coords': torch.from_numpy(poi_coords),
            'types': torch.tensor(poi_types, dtype=torch.long),
        },
        'area': {
            'num_nodes': len(area_coords),
            'coords': torch.from_numpy(area_coords),
            'types': torch.tensor(area_types, dtype=torch.long),
            'area_size': torch.tensor(area_sizes, dtype=torch.float32) if area_sizes else torch.zeros(0, dtype=torch.float32),
        },
        'road_edges': {
            'edge_index': torch.tensor(road_ei, dtype=torch.long).t().contiguous() if road_ei else torch.zeros((2, 0), dtype=torch.long),
            'edge_type': torch.tensor(road_et, dtype=torch.long) if road_et else torch.zeros(0, dtype=torch.long),
            'edge_length': torch.tensor(road_el, dtype=torch.float32) if road_el else torch.zeros(0, dtype=torch.float32),
        },
        'poi_to_intersection': {
            'edge_index': torch.tensor([[e[0], e[1]] for e in poi_to_inter], dtype=torch.long).t().contiguous() if poi_to_inter else torch.zeros((2, 0), dtype=torch.long),
            'edge_dist': torch.tensor([e[2] for e in poi_to_inter], dtype=torch.float32) if poi_to_inter else torch.zeros(0, dtype=torch.float32),
        },
        'area_to_intersection': {
            'edge_index': torch.tensor([[e[0], e[1]] for e in area_to_inter], dtype=torch.long).t().contiguous() if area_to_inter else torch.zeros((2, 0), dtype=torch.long),
            'edge_dist': torch.tensor([e[2] for e in area_to_inter], dtype=torch.float32) if area_to_inter else torch.zeros(0, dtype=torch.float32),
        },
        'bbox_min': torch.from_numpy(bbox.min_.astype(np.float32)),
        'bbox_max': torch.from_numpy(bbox.max_.astype(np.float32)),
        'num_classes': {
            'ways': len(Groups.ways),
            'nodes': len(Groups.nodes),
            'areas': len(Groups.areas),
        },
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', type=str, default=None, help='')
    parser.add_argument('--max', type=int, default=None, help='')
    args = parser.parse_args()

    root = Path('${UAV2MAP_ROOT}')

    cities = ['Paris', 'Berlin', 'London', 'NewYork', 'SanFrancisco', 'Tokyo', 'Toronto']
    if args.city:
        cities = [args.city]

    t_total = time.time()
    for city in cities:
        osm_path = root / city / f'{city}.osm'
        info_path = root / city / 'info.csv'
        output_dir = root / city / 'graph'

        if not osm_path.exists():
            print(f"[SKIP] {city}:  .osm ")
            continue
        if not info_path.exists():
            print(f"[SKIP] {city}:  info.csv")
            continue

        print(f"\n{'='*50}")
        print(f": {city}")
        print(f"{'='*50}")
        batch_build_city(osm_path, info_path, output_dir, args.max)

    print(f"\n！ {time.time()-t_total:.0f}s")
