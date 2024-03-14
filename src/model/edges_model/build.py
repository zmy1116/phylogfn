from src.model.edges_model.categorical.categorical import EdgesModelCategorical
from src.model.edges_model.continuous.continuous_mixture import EdgeModelContinuousMixture


def build_edge_model(gfn_cfg):
    edges_cfg = gfn_cfg.MODEL.EDGES_MODELING
    dist = edges_cfg.DISTRIBUTION
    assert dist in ['CATEGORICAL', 'MIXTURE']
    if dist == 'CATEGORICAL':
        edge_model = EdgesModelCategorical(edges_cfg.CATEGORICAL)
    else:
        edge_model = EdgeModelContinuousMixture(edges_cfg.MIXTURE)
    return edge_model
