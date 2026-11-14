from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline


def build_csp_lda(n_components=6):
    return make_pipeline(
        CSP(n_components=n_components, reg=None, log=True, norm_trace=False),
        LinearDiscriminantAnalysis(),
    )
