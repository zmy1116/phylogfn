from src.gfn.tb_gfn_phylo import TBGFlowNetGenerator


def build_gfn(cfg, env, device, ddp):
    assert cfg.GFN.LOSS_TYPE in ['TB', ]
    generator = TBGFlowNetGenerator(cfg.GFN, env, device, ddp)
    return generator
