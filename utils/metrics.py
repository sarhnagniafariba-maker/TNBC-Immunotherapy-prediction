from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from scipy.stats import bootstrap, ztest

def compute_metrics_with_ci(y_true, y_pred, bootstrap_n=1000):
    auc = roc_auc_score(y_true, y_pred)
    auc_ci = bootstrap((y_true, y_pred), roc_auc_score, n_resamples=bootstrap_n).confidence_interval
    auprc = average_precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred > 0.5)
    return auc, auc_ci, auprc, f1

def delong_test(y_true, y_pred1, y_pred2):
    auc1 = roc_auc_score(y_true, y_pred1)
    auc2 = roc_auc_score(y_true, y_pred2)
    var1 = auc1 * (1 - auc1) / len(y_true)
    var2 = auc2 * (1 - auc2) / len(y_true)
    z = (auc1 - auc2) / np.sqrt(var1 + var2)
    p = ztest(z)[1]
    return p