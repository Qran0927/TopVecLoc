"""
GraphGPS  + 

：
  1. GNN  = （+）
  2.  = ""

 GNN  CNN？
  CNN  → 
  GNN  → 

：
  graph (.pt) → [GNN] →  [N, hidden_dim]
    → [] →  [1, 8, 256, 256]
    →  FFT （）
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# ：GINE （ GNN）
# ============================================================

class GINEConv(nn.Module):
    """
    GINE: Graph Isomorphism Network with Edge features

     GCN/GAT ：
      - GCN: 
      - GINE:  + ，""

    ：
      h_i' = MLP( (1+ε)·h_i + Σ_{j∈N(i)} ReLU(h_j + edge_attr_ij) )

    ：
      h_i      =  i 
      edge_attr =  i  j （、）
      ε        = 
    """

    def __init__(self, hidden_dim: int, edge_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # epsilon: " vs "
        self.eps = nn.Parameter(torch.zeros(1))
        self.edge_encoder = nn.Linear(edge_dim, hidden_dim)

    def forward(self, x, edge_index, edge_attr):
        """
        x:          [N, hidden_dim]    
        edge_index: [2, E]             
        edge_attr:  [E, edge_dim]      
        """
        row, col = edge_index  # row = , col = 

        # 
        edge_feat = self.edge_encoder(edge_attr)  # [E, hidden_dim]

        # : h_j + e_ij,  ReLU
        msg = F.relu(x[col] + edge_feat)  # [E, hidden_dim]

        # ，（）
        num_nodes = x.size(0)
        agg = torch.zeros(num_nodes, x.size(1), device=x.device)
        agg = agg.index_add(0, row, msg)  # scatter sum

        # GINE : (1+eps) *  + 
        out = (1 + self.eps) * x + agg
        return self.mlp(out)


# ============================================================
# ：GraphGPS  —— MPNN + Transformer 
# ============================================================

class GlobalTransformer(nn.Module):
    """
    GraphGPS 。

     GNN （）。
    Transformer （）。

    ：
      - 
      - Transformer ""，
      - ""
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True, dropout=dropout
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, valid_mask=None):
        """
        x: [N, hidden_dim]  

        ：（）
        → 
        """
        single_graph = x.ndim == 2
        if single_graph:
            x = x.unsqueeze(0)
        if valid_mask is None:
            valid_mask = torch.ones(
                x.shape[:2], dtype=torch.bool, device=x.device
            )

        attn_out, _ = self.attn(
            x,
            x,
            x,
            key_padding_mask=~valid_mask,
            need_weights=False,
        )
        output = self.norm(x + self.dropout(attn_out))
        output = output.masked_fill(~valid_mask[..., None], 0)
        return output.squeeze(0) if single_graph else output


class GPSLayer(nn.Module):
    """
    GraphGPS  =  + 

    :
      x ──┬──→ GINE () ──→ h_local
          │
          └──→ Transformer () ──→ h_global
          │
          └──→ h_local + h_global → LayerNorm → FFN → 

    、、。
    （），
    （）。
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        # ：GINE 
        self.mpnn = GINEConv(hidden_dim, edge_dim)

        # ：Transformer 
        self.transformer = GlobalTransformer(hidden_dim, num_heads, dropout)

        # 
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, valid_mask, edge_index, edge_attr):
        # 
        batch_size, max_nodes, hidden_dim = x.shape
        h_local = self.mpnn(
            x.reshape(batch_size * max_nodes, hidden_dim),
            edge_index,
            edge_attr,
        ).reshape(batch_size, max_nodes, hidden_dim)

        # 
        h_global = self.transformer(x, valid_mask)

        # :  +  + 
        h = x + self.dropout(h_local) + self.dropout(h_global)
        h = self.norm(h)

        # FFN + 
        h = h + self.dropout(self.ffn(h))

        return h.masked_fill(~valid_mask[..., None], 0)


# ============================================================
# ：
# ============================================================

class GeographicPE(nn.Module):
    """
     2D  (x, y) 。

     Laplacian PE（GraphGPS ）？
      Laplacian PE ，。
       PE ，：
        → "" 
        → 

    ：（ NeRF/Transformer ）
      PE(x) = [sin(w1·x), cos(w1·x), sin(w2·x), cos(w2·x), ..., wx+b]
       w 
    """

    def __init__(self, num_freqs: int = 16, output_dim: int = 64):
        super().__init__()
        # 
        self.freqs = nn.Parameter(torch.randn(num_freqs, 2) * 0.1)
        self.proj = nn.Linear(num_freqs * 4 + 2, output_dim)  # sin/cos for x and y

    def forward(self, coords):
        """
        coords: [N, 2]   (x, y)
        :   [N, output_dim]
        """
        # coords: [N, 2], freqs: [F, 2]
        angles = coords.unsqueeze(-2) * self.freqs  # [..., N, F, 2]
        encoded = torch.cat([
            torch.sin(angles[..., 0]),
            torch.cos(angles[..., 0]),
            torch.sin(angles[..., 1]),
            torch.cos(angles[..., 1]),
            coords,  # 
        ], dim=-1)  # [N, F*4 + 2]
        return self.proj(encoded)


# ============================================================
# ：（ → ）
# ============================================================

class DifferentiableRenderer(nn.Module):
    """
     ""  [C, H, W] 。

    ：（IDW）
       (u,v)  =  K 
       = 1 / (^2 + eps)

     →  GNN 
    """

    def __init__(self, output_size: int = 256, output_dim: int = 8, knn_k: int = 8):
        super().__init__()
        self.output_size = output_size
        self.output_dim = output_dim
        self.knn_k = knn_k

    def forward(
        self,
        node_embeddings,
        node_coords,
        valid_mask=None,
        bbox_min=None,
        bbox_max=None,
    ):
        """
        node_embeddings: [N, feat_dim]  
        node_coords:     [N, 2]          (x, y)
        bbox_min, bbox_max: [2]         （ FFT ）

        : [1, output_dim, H, W]
        """
        if node_embeddings.ndim == 2:
            node_embeddings = node_embeddings.unsqueeze(0)
            node_coords = node_coords.unsqueeze(0)
        batch_size, max_nodes = node_coords.shape[:2]
        H, W = self.output_size, self.output_size
        device = node_embeddings.device
        if valid_mask is None:
            valid_mask = torch.ones(
                (batch_size, max_nodes), dtype=torch.bool, device=device
            )
        else:
            valid_mask = valid_mask.to(device)
        if max_nodes == 0:
            return node_embeddings.new_zeros(
                batch_size, self.output_dim, H, W
            )

        if bbox_min is None or bbox_max is None:
            masked_min = node_coords.masked_fill(~valid_mask[..., None], torch.inf)
            masked_max = node_coords.masked_fill(~valid_mask[..., None], -torch.inf)
            bbox_min = masked_min.amin(dim=1) - 10.0
            bbox_max = masked_max.amax(dim=1) + 10.0
            empty = ~valid_mask.any(dim=1)
            bbox_min[empty] = -10.0
            bbox_max[empty] = 10.0
        else:
            bbox_min = bbox_min.to(device)
            bbox_max = bbox_max.to(device)
            if bbox_min.ndim == 1:
                bbox_min = bbox_min.unsqueeze(0)
                bbox_max = bbox_max.unsqueeze(0)

        x_ratio = torch.linspace(0, 1, W, device=device)
        y_ratio = torch.linspace(0, 1, H, device=device)
        xs = bbox_min[:, 0, None] + (
            bbox_max[:, 0] - bbox_min[:, 0]
        )[:, None] * x_ratio
        ys = bbox_max[:, 1, None] + (
            bbox_min[:, 1] - bbox_max[:, 1]
        )[:, None] * y_ratio
        grid_x = xs[:, None, :].expand(-1, H, -1)
        grid_y = ys[:, :, None].expand(-1, -1, W)
        grid_flat = torch.stack((grid_x, grid_y), dim=-1).reshape(
            batch_size, H * W, 2
        )

        chunk_size = 1024
        k = min(self.knn_k, max_nodes)
        batch_index = torch.arange(batch_size, device=device)[:, None, None]
        chunks = []
        for start in range(0, H * W, chunk_size):
            end = min(start + chunk_size, H * W)
            distances = torch.cdist(grid_flat[:, start:end], node_coords)
            distances = distances.masked_fill(~valid_mask[:, None, :], torch.inf)
            topk_distances, topk_indices = torch.topk(
                distances, k, dim=-1, largest=False
            )
            weights = torch.where(
                torch.isfinite(topk_distances),
                1.0 / (topk_distances.square() + 1e-6),
                0.0,
            )
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            nearest = node_embeddings[batch_index, topk_indices]
            chunks.append((nearest * weights[..., None]).sum(dim=-2))

        feat_map = torch.cat(chunks, dim=1)
        return feat_map.reshape(
            batch_size, H, W, self.output_dim
        ).permute(0, 3, 1, 2)


# ============================================================
# ：GraphGPSMapEncoder（）
# ============================================================

class GraphGPSMapEncoder(nn.Module):
    """
    GNN  ——  MapEncoder (VGG19+UNet)

    ：（graph_builder ）
    ： [B, 8, 256, 256]

    ：
      1. 
      2. 
      3. GNN （）
      4.  → 
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        output_dim: int = 8,
        num_layers: int = 4,
        num_road_types: int = 10,
        output_size: int = 256,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.output_size = output_size

        # 
        self.pos_encoder = GeographicPE(num_freqs=16, output_dim=hidden_dim)

        # 
        # : degree(1) + road_type_dist(10) + avg_seg_len(1) + poi_density(1) + building_density(1) = 14
        self.intersection_proj = nn.Sequential(
            nn.Linear(14, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # POI :  + 
        self.poi_type_embed = nn.Embedding(33, 16)  # 33  POI 

        # :  +  + 
        self.area_type_embed = nn.Embedding(7, 16)   # 7 

        self.poi_context_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.area_context_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 
        # edge : edge_type  + edge_length
        self.edge_type_embed = nn.Embedding(num_road_types, 8)

        # GraphGPS ： GINE  +  Transformer 
        self.gnn_layers = nn.ModuleList([
            GPSLayer(hidden_dim, edge_dim=hidden_dim, num_heads=4, dropout=0.1)
            for _ in range(num_layers)
        ])

        # （ ReLU，）
        self.output_proj = nn.Linear(hidden_dim, output_dim)

        # 
        self.renderer = DifferentiableRenderer(
            output_size=output_size,
            output_dim=output_dim,
            knn_k=3,   # K（IDW）
        )
        # ：batch size=1  BatchNorm 。
        self.map_norm = nn.GroupNorm(1, output_dim)

        # ： 0，FFT 
        self._init_weights()

    def _init_weights(self):
        """：≈0， FFT """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.01)

    def _aggregate_to_intersections(
        self, src_x, edge_dict, intersection_mask, device
    ):
        """Aggregate padded POI/area features into padded intersections."""
        batch_size, max_intersections = intersection_mask.shape
        output = torch.zeros(
            batch_size * max_intersections, self.hidden_dim, device=device
        )
        edge_index = None if edge_dict is None else edge_dict.get("edge_index")
        if src_x.numel() == 0 or edge_index is None or edge_index.numel() == 0:
            return output.reshape(batch_size, max_intersections, self.hidden_dim)

        edge_index = edge_index.to(device)
        source_index = edge_index[0].long()
        intersection_index = edge_index[1].long()
        source_flat = src_x.reshape(-1, self.hidden_dim)

        edge_distance = edge_dict.get("edge_dist")
        if edge_distance is None or edge_distance.numel() == 0:
            weights = torch.ones(
                source_index.shape[0], 1, device=device
            )
        else:
            weights = 1.0 / (
                edge_distance.to(device).reshape(-1, 1).clamp_min(0.0) + 1.0
            )

        output.index_add_(0, intersection_index, source_flat[source_index] * weights)
        denominator = torch.zeros(
            batch_size * max_intersections, 1, device=device
        )
        denominator.index_add_(0, intersection_index, weights)
        output = output / denominator.clamp_min(1e-6)
        output = output.reshape(batch_size, max_intersections, self.hidden_dim)
        return output.masked_fill(~intersection_mask[..., None], 0)

    def forward(self, batch):
        """
        batch: dict，:
          'graph':  dict ( graph_builder)
          （ 'map' ）

        :
          {'map_features': [feat_map]}   MapEncoder 
        """
        if "graph" not in batch:
            raise ValueError("GraphGPSMapEncoder  graph ， batch  'graph'")
        graph = batch["graph"]
        if "mask" not in graph["intersection"]:
            from dataset.torch import collate_graphs

            coords = graph["intersection"]["coords"]
            if coords.ndim == 3:
                if coords.shape[0] != 1:
                    raise ValueError(
                        "Legacy graph batches without masks only support batch size 1"
                    )

                def remove_legacy_batch_dim(value):
                    if torch.is_tensor(value):
                        return value.squeeze(0)
                    if isinstance(value, dict):
                        return {
                            key: remove_legacy_batch_dim(item)
                            for key, item in value.items()
                        }
                    return value

                graph = remove_legacy_batch_dim(graph)
            elif coords.ndim != 2:
                raise ValueError(
                    f"Unsupported graph coordinate shape: {tuple(coords.shape)}"
                )
            graph = collate_graphs([graph])

        device = next(self.parameters()).device

        inter = graph["intersection"]
        inter_mask = inter["mask"].to(device)
        inter_coords = inter["coords"].to(device)
        inter_features = torch.cat(
            [
                inter["degree"].to(device).unsqueeze(-1),
                inter["road_type_dist"].to(device),
                inter["avg_segment_len"].to(device).unsqueeze(-1),
                inter["poi_density"].to(device).unsqueeze(-1),
                inter["building_density"].to(device).unsqueeze(-1),
            ],
            dim=-1,
        )
        inter_x = self.intersection_proj(inter_features)
        inter_x = inter_x + self.pos_encoder(inter_coords)
        inter_x = inter_x.masked_fill(~inter_mask[..., None], 0)

        poi = graph["poi"]
        poi_mask = poi["mask"].to(device)
        poi_coords = poi["coords"].to(device)
        poi_x = self.poi_type_embed(poi["types"].to(device))
        poi_x = F.pad(poi_x, (0, self.hidden_dim - 16))
        poi_x = (poi_x + self.pos_encoder(poi_coords)).masked_fill(
            ~poi_mask[..., None], 0
        )

        area = graph["area"]
        area_mask = area["mask"].to(device)
        area_coords = area["coords"].to(device)
        area_x = self.area_type_embed(area["types"].to(device))
        area_x = torch.cat(
            [area_x, area["area_size"].to(device).unsqueeze(-1)], dim=-1
        )
        area_x = F.pad(area_x, (0, self.hidden_dim - 17))
        area_x = (area_x + self.pos_encoder(area_coords)).masked_fill(
            ~area_mask[..., None], 0
        )

        poi_context = self._aggregate_to_intersections(
            poi_x, graph.get("poi_to_intersection"), inter_mask, device
        )
        area_context = self._aggregate_to_intersections(
            area_x, graph.get("area_to_intersection"), inter_mask, device
        )
        inter_x = (
            inter_x
            + self.poi_context_proj(poi_context)
            + self.area_context_proj(area_context)
        ).masked_fill(~inter_mask[..., None], 0)

        road_edges = graph["road_edges"]
        edge_index = road_edges["edge_index"].to(device)
        edge_type = road_edges["edge_type"].to(device)
        edge_length = road_edges["edge_length"].to(device).unsqueeze(-1)
        edge_attr = torch.cat(
            [self.edge_type_embed(edge_type), edge_length], dim=-1
        )
        edge_attr = F.pad(edge_attr, (0, self.hidden_dim - 9))

        x = inter_x
        for gnn in self.gnn_layers:
            x = F.relu(gnn(x, inter_mask, edge_index, edge_attr))
            x = x.masked_fill(~inter_mask[..., None], 0)

        inter_out = self.output_proj(x).masked_fill(
            ~inter_mask[..., None], 0
        )
        feat_map = self.renderer(
            inter_out,
            inter_coords,
            inter_mask,
            graph.get("bbox_min"),
            graph.get("bbox_max"),
        )
        feat_map = self.map_norm(feat_map)
        empty_graph = ~inter_mask.any(dim=1)
        if empty_graph.any():
            feat_map = feat_map.masked_fill(empty_graph[:, None, None, None], 0)
        return {"map_features": [feat_map]}


# ============================================================
# 
# ============================================================

if __name__ == '__main__':
    import csv
    import numpy as np
    from osm.reader import OSMData
    from utils.geo import BoundaryBox, Projection
    from models.graph_builder import build_hetero_graph

    root = Path('${UAV2MAP_ROOT}')
    city = 'SanFrancisco'
    osm = OSMData.from_file(root / city / f'{city}.osm')
    print(f"OSM: {len(osm.nodes)} nodes, {len(osm.ways)} ways")

    with open(root / city / 'info.csv', 'r') as f:
        rows = list(csv.reader(f))
    row = rows[1]
    map_lat, map_long = float(row[6]), float(row[5])
    tile_size = float(row[7])

    proj = Projection(map_lat, map_long)
    center = proj.project(np.array([map_lat, map_long]))
    bbox = BoundaryBox(center, center) + tile_size
    osm.add_xy_to_nodes(proj)
    graph = build_hetero_graph(osm, bbox, proj)

    # 
    encoder = GraphGPSMapEncoder(hidden_dim=128, output_dim=8, num_layers=4)
    encoder.eval()

    with torch.no_grad():
        output = encoder({'graph': graph})

    feat_map = output['map_features'][0]
    print(f"\nGNN : {feat_map.shape}  (: [1, 8, 256, 256])")
    print(f": [{feat_map.min():.3f}, {feat_map.max():.3f}]")
    print(f": {feat_map.mean():.3f}")
    print(f"\n！ → GNN →  → [1, 8, 256, 256] ")
