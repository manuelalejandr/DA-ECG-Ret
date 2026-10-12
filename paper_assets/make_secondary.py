"""Table of the eight secondary datasets (P@10 and AUROC, seed 0, NN) from secondary_raw.csv -> tab_secondary.tex"""
import pandas as pd
d = pd.read_csv("secondary_raw.csv")
M = ["simclr", "ts2vec", "tfc", "softclt", "da_ecg_ret"]
N = {"simclr": "SimCLR", "ts2vec": "TS2Vec", "tfc": "TF-C", "softclt": "SoftCLT", "da_ecg_ret": "DA-ECG-Ret"}
D = {"ucr_ECG200": "ECG200", "ucr_ECG5000": "ECG5000", "ucr_ECGFiveDays": "ECGFiveDays", "ucr_TwoLeadECG": "TwoLeadECG",
     "ucr_CinCECGTorso": "CinCECGTorso", "ucr_NonInvasiveFetalECGThorax1": "FetalECGThorax1",
     "ucr_NonInvasiveFetalECGThorax2": "FetalECGThorax2", "mitbih": "MIT-BIH"}
P = d[(d.ranker == "nn") & (d.K == 10) & (d.metric == "P")].pivot_table(index="dataset", columns="method", values="value")[M].loc[list(D)]
A = d[d.metric == "AUROC_super"].pivot_table(index="dataset", columns="method", values="value")[M].loc[list(D)]
R = d[(d.ranker == "rand") & (d.K == 10) & (d.metric == "P")].pivot_table(index="dataset", columns="method", values="value").mean(axis=1).loc[list(D)]
def block(T, rand=None):
    out = []
    for k, r in T.iterrows():
        b = r.idxmax()
        cells = [("\\textbf{%.3f}" if m == b else "%.3f") % r[m] for m in M]
        out.append(D[k] + " & " + (f"{rand[k]:.3f}" if rand is not None else "--") + " & " + " & ".join(cells) + " \\\\")
    rk = T.rank(axis=1, ascending=False).mean()
    out.append("\\midrule")
    out.append("Mean rank & -- & " + " & ".join(f"{rk[m]:.2f}" for m in M) + " \\\\")
    return out
with open("tab_secondary.tex", "w") as f:
    f.write("\\begin{tabular}{llccccc}\n\\toprule\n& Random & " + " & ".join(N[m] for m in M) + " \\\\\n\\midrule\n")
    f.write("\\multicolumn{7}{l}{\\emph{P@10 (NN retrieval)}} \\\\\n" + "\n".join(block(P, R)) + "\n\\midrule\n")
    f.write("\\multicolumn{7}{l}{\\emph{Linear-probe macro-AUROC}} \\\\\n" + "\n".join(block(A)) + "\n\\bottomrule\n\\end{tabular}\n")
print(open("tab_secondary.tex").read())
