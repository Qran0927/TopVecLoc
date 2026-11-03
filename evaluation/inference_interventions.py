"""Reversible, fixed-weight inference interventions; never edit graph files."""
from collections import Counter, defaultdict, deque
import random

import torch

VARIANTS = ('none','no_attention','no_pe','no_context','no_edges','random_edges',
            'k1','k3','k5','k8','shift_origin')


def intervene_graph(graph, variant, seed):
    result = dict(graph)
    road = {k:v.clone() if torch.is_tensor(v) else v for k,v in graph['road_edges'].items()}
    result['road_edges'] = road
    edges = road['edge_index']
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError('Expected an unbatched [2,E] edge tensor')
    info = dict(original_edges=edges.shape[1], changed_edges=0, swaps=0)
    if variant == 'no_edges':
        for key,value in road.items():
            road[key] = value[:,:0] if key == 'edge_index' else value[:0]
        info['changed_edges'] = edges.shape[1]
        return result, info
    if variant != 'random_edges':
        return result, info
    original = edges.clone()
    oriented = list(map(tuple,edges.T.tolist()))
    counts = Counter(oriented)
    if any(counts[(v,u)] != n for (u,v),n in counts.items()):
        raise ValueError('Degree-preserving undirected rewiring requires symmetric road edges')
    slots = defaultdict(deque)
    for i,(u,v) in enumerate(oriented):
        if u > v:
            slots[(u,v)].append(i)
    pairs = []
    for i,(u,v) in enumerate(oriented):
        if u < v:
            pairs.append([u,v,i,slots[(v,u)].popleft()])
    present = Counter(tuple(sorted((u,v))) for u,v,_,_ in pairs)
    rng = random.Random(seed)
    desired = 5*len(pairs)
    for _ in range(100*len(pairs)):
        if len(pairs) < 2 or info['swaps'] >= desired:
            break
        i,j = rng.sample(range(len(pairs)),2)
        a,b,slot_i,rev_i = pairs[i]
        c,d,slot_j,rev_j = pairs[j]
        if rng.random() < .5:
            c,d,slot_j,rev_j = d,c,rev_j,slot_j
        if len({a,b,c,d}) != 4:
            continue
        p,q = tuple(sorted((a,d))),tuple(sorted((c,b)))
        if present[p] or present[q]:
            continue
        present[tuple(sorted((a,b)))] -= 1
        present[tuple(sorted((c,d)))] -= 1
        present[p] += 1; present[q] += 1
        pairs[i] = [a,d,slot_i,rev_i]
        pairs[j] = [c,b,slot_j,rev_j]
        info['swaps'] += 1
    for u,v,i,j in pairs:
        edges[:,i] = edges.new_tensor([u,v])
        edges[:,j] = edges.new_tensor([v,u])
    info['changed_edges'] = int((edges != original).any(0).sum())
    return result, info


class Intervention:
    def __init__(self, encoder, variant, seed=0, shift_m=128):
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.encoder, self.variant, self.seed = encoder, variant, seed
        self.shift_m = shift_m
        self.handles = []
        self.old_k = encoder.renderer.knn_k
        self.set_sample(0)

    def set_sample(self, index):
        self.sample_seed = (self.seed*1_000_003 + int(index)) % (2**63-1)
        generator = torch.Generator().manual_seed(self.sample_seed)
        self.offset = (torch.rand(2,generator=generator)*2-1)*self.shift_m

    def __enter__(self):
        def zero(module,args,output):
            return torch.zeros_like(output)
        def density_off(module,args):
            features = args[0].clone()
            features[...,-2:] = 0
            return (features,)
        def shift(module,args):
            return (args[0]+self.offset.to(args[0]),)
        if self.variant == 'no_attention':
            self.handles += [layer.transformer.register_forward_hook(zero)
                             for layer in self.encoder.gnn_layers]
        elif self.variant == 'no_pe':
            self.handles.append(self.encoder.pos_encoder.register_forward_hook(zero))
        elif self.variant == 'no_context':
            self.handles += [module.register_forward_hook(zero) for module in
                             (self.encoder.poi_context_proj,self.encoder.area_context_proj)]
            self.handles.append(self.encoder.intersection_proj.register_forward_pre_hook(density_off))
        elif self.variant == 'shift_origin':
            self.handles.append(self.encoder.pos_encoder.register_forward_pre_hook(shift))
        elif self.variant.startswith('k'):
            self.encoder.renderer.knn_k = int(self.variant[1:])
        return self

    def prepare(self, batch):
        index = int(batch['sample_index'][0])
        self.set_sample(index)
        info = dict(original_edges=0,changed_edges=0,swaps=0)
        if self.variant not in ('no_edges','random_edges'):
            return batch, info
        graph = dict(batch['graph'])
        raw = dict(graph)
        raw['road_edges'] = {k:v.squeeze(0) for k,v in graph['road_edges'].items()}
        changed,info = intervene_graph(raw,self.variant,self.sample_seed)
        graph['road_edges'] = {k:v.unsqueeze(0) for k,v in changed['road_edges'].items()}
        batch = dict(batch,graph=graph)
        return batch,info

    def __exit__(self,*exception):
        for handle in self.handles:
            handle.remove()
        self.encoder.renderer.knn_k = self.old_k
