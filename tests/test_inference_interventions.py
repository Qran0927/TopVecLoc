import copy
import torch


def graph_example():
    pairs = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,0),(0,4),(2,6)]
    edges = torch.tensor([(a,b) for u,v in pairs for a,b in [(u,v),(v,u)]]).T
    return {'intersection':{'coords':torch.randn(8,2)},
            'road_edges':{'edge_index':edges,'edge_type':torch.arange(edges.shape[1]),
                          'edge_length':torch.ones(edges.shape[1])}}


def test_removing_edges_keeps_nodes_and_original_graph_immutable():
    from evaluation.inference_interventions import intervene_graph
    graph = graph_example()
    original = copy.deepcopy(graph)
    changed, info = intervene_graph(graph,'no_edges',0)
    assert changed['road_edges']['edge_index'].shape == (2,0)
    assert changed['road_edges']['edge_type'].numel() == 0
    assert torch.equal(changed['intersection']['coords'],graph['intersection']['coords'])
    assert torch.equal(graph['road_edges']['edge_index'],original['road_edges']['edge_index'])
    assert info['original_edges'] == 20


def test_rewiring_is_reproducible_bidirectional_and_degree_preserving():
    from evaluation.inference_interventions import intervene_graph
    graph = graph_example()
    a, info = intervene_graph(graph,'random_edges',17)
    b, _ = intervene_graph(graph,'random_edges',17)
    old = graph['road_edges']['edge_index']; new = a['road_edges']['edge_index']
    assert torch.equal(new,b['road_edges']['edge_index'])
    assert new.shape == old.shape and not torch.equal(new,old)
    assert torch.equal(torch.bincount(new[0],minlength=8),torch.bincount(old[0],minlength=8))
    pairs = set(map(tuple,new.T.tolist()))
    assert all(u != v and (v,u) in pairs for u,v in pairs)
    assert info['swaps'] > 0
    assert torch.equal(a['road_edges']['edge_type'],graph['road_edges']['edge_type'])


def test_hooks_restore_and_zero_pe_without_moving_renderer_coordinates():
    from evaluation.inference_interventions import Intervention
    from models.graphgps_encoder import GraphGPSMapEncoder
    encoder = GraphGPSMapEncoder(hidden_dim=16,num_layers=1,output_size=8)
    coords = torch.randn(1,4,2)
    expected = encoder.pos_encoder(coords).detach().clone()
    with Intervention(encoder,'no_pe') as run:
        run.set_sample(7)
        assert not encoder.pos_encoder(coords).any()
    assert torch.equal(encoder.pos_encoder(coords),expected)
    with Intervention(encoder,'shift_origin',seed=3) as run:
        run.set_sample(7)
        first = encoder.pos_encoder(coords)
        assert not torch.equal(first,expected)
        assert torch.equal(first,encoder.pos_encoder(coords))
    assert torch.equal(encoder.pos_encoder(coords),expected)


def test_attention_and_context_hooks_remove_outputs_including_bias():
    from evaluation.inference_interventions import Intervention
    from models.graphgps_encoder import GraphGPSMapEncoder
    encoder = GraphGPSMapEncoder(hidden_dim=16,num_layers=1,output_size=8)
    x = torch.randn(1,4,16); mask = torch.ones(1,4,dtype=torch.bool)
    with Intervention(encoder,'no_attention'):
        assert not encoder.gnn_layers[0].transformer(x,mask).any()
    with Intervention(encoder,'no_context'):
        assert not encoder.poi_context_proj(x).any()
        assert not encoder.area_context_proj(x).any()
        features = torch.randn(1,4,14)
        a = encoder.intersection_proj(features)
        features[...,-2:] += 100
        assert torch.equal(a,encoder.intersection_proj(features))


def test_all_interventions_leave_parameters_unchanged_and_restore_k():
    from evaluation.inference_interventions import Intervention, VARIANTS
    from models.graphgps_encoder import GraphGPSMapEncoder
    encoder = GraphGPSMapEncoder(hidden_dim=16,num_layers=1,output_size=8).eval()
    original = {k:v.clone() for k,v in encoder.state_dict().items()}
    for variant in VARIANTS:
        with Intervention(encoder,variant) as run:
            run.set_sample(10)
            encoder.pos_encoder(torch.randn(1,3,2))
        assert encoder.renderer.knn_k == 3
        assert all(torch.equal(v,original[k]) for k,v in encoder.state_dict().items())


def test_batched_graph_prepare_and_small_unrewirable_graph():
    from evaluation.inference_interventions import Intervention, intervene_graph
    from models.graphgps_encoder import GraphGPSMapEncoder
    graph = graph_example()
    batched = {'sample_index':torch.tensor([12]),'graph':dict(graph,
        road_edges={k:v[None] for k,v in graph['road_edges'].items()})}
    encoder = GraphGPSMapEncoder(hidden_dim=16,num_layers=1,output_size=8)
    with Intervention(encoder,'no_edges') as run:
        changed,info = run.prepare(batched)
    assert changed['graph']['road_edges']['edge_index'].shape == (1,2,0)
    assert batched['graph']['road_edges']['edge_index'].shape[-1] == 20
    tiny = {'road_edges':{'edge_index':torch.tensor([[0,1],[1,0]]),
                         'edge_type':torch.ones(2),'edge_length':torch.ones(2)}}
    changed,info = intervene_graph(tiny,'random_edges',0)
    assert info['swaps'] == 0 and info['changed_edges'] == 0
