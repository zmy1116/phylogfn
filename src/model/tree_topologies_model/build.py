from src.model.tree_topologies_model.one_step_model import PhyloTreeModelOneStep


def build_tree_model(gfn_cfg, env_type):
    assert env_type == 'ONE_STEP_BINARY_TREE'
    generator = PhyloTreeModelOneStep(gfn_cfg)
    return generator
