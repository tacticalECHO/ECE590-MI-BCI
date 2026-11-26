from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline


def build_csp_lda(n_components=6, csp_reg=None, lda_shrinkage=None):
    lda = (LinearDiscriminantAnalysis(solver="lsqr", shrinkage=lda_shrinkage)
           if lda_shrinkage is not None else LinearDiscriminantAnalysis())
    return make_pipeline(
        CSP(n_components=n_components, reg=csp_reg, log=True, norm_trace=False),
        lda,
    )
