from src.env.edges_env.categorical import EdgeEnvCategorical
from src.env.edges_env.continuous import EdgeEnvContinuous


def build_edge_env(cfg):
    edges_cfg = cfg.GFN.MODEL.EDGES_MODELING
    dist = edges_cfg.DISTRIBUTION
    assert dist in ['CATEGORICAL', 'MIXTURE']
    if dist == 'CATEGORICAL':
        edge_env = EdgeEnvCategorical(edges_cfg.CATEGORICAL)
    else:
        edge_env = EdgeEnvContinuous(edges_cfg.MIXTURE)
    return edge_env
