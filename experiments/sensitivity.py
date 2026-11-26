from itertools import product

N_COMPONENTS = (2, 4, 6, 8, 10)
CSP_REGS = (None, "ledoit_wolf")
LDA_SHRINKAGES = (None, "auto")
DEFAULT = {"n_components": 6, "csp_reg": None, "lda_shrinkage": None}


def grid():
    return [{"n_components": k, "csp_reg": reg, "lda_shrinkage": shrinkage}
            for reg, shrinkage, k in product(CSP_REGS, LDA_SHRINKAGES, N_COMPONENTS)]


def config_name(config):
    return (f"k{config['n_components']}"
            f"_csp-{config['csp_reg'] or 'none'}"
            f"_lda-{config['lda_shrinkage'] or 'none'}")
