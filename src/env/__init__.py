from src.env.binary_tree_env_one_step_likelihood import PhylogenticTreeEnv


def build_env(cfg, all_seqs):
    assert cfg.ENV.ENVIRONMENT_TYPE in ['ONE_STEP_BINARY_TREE']
    env = PhylogenticTreeEnv(cfg, all_seqs)

    return env
