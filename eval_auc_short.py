import numpy as np, torch, pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve
import blackcatt.wm_config as C
from blackcatt.models import ResNet, ResidualBlock
from blackcatt.wm_task import load_collusion, _normalize_triggers

def arr(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

def score_vec(y, V, P):
    m = V.shape[1]
    p = P[np.arange(m), y]
    p = np.clip(p, 1e-12, 1-1e-12)
    plus = np.sqrt((1-p)/p)
    minus = -np.sqrt(p/(1-p))
    S = np.where(V == y[None,:], plus[None,:], minus[None,:])
    return S.sum(1), S.cumsum(1).max(1)

def run(name, folder):
    folder = Path(folder)
    trig = _normalize_triggers(np.load(folder/"triggers.npy"), "cuda")
    V, P = arr(C.clients_tardos_q).astype(int), arr(C.p_secret).astype(float)

    # Force V to shape [n_users, m].
    # The constant file may contain codes for more users; this run uses only C.n_users.
    if V.shape[0] == C.n_users:
        pass
    elif V.shape[1] == C.n_users:
        V = V.T
    elif V.shape[0] > C.n_users:
        print(f"[info] code has {V.shape[0]} users; using first C.n_users={C.n_users}")
        V = V[:C.n_users, :]
    else:
        raise ValueError(f"Cannot align clients_tardos_q shape {V.shape} with C.n_users={C.n_users}")

    m = V.shape[1]

    # Force P to shape [m, n_classes].
    # This follows wm_config.py:
    # clients_tardos_q = clients_tardos_q[:, :m]
    # p_secret_tensor = torch.from_numpy(p_secret[:m, :]).float()
    if P.shape[0] == m:
        pass
    elif P.shape[1] == m:
        P = P.T
    elif P.shape[0] > m:
        print(f"[info] p_secret has {P.shape[0]} rows; using first m={m} rows, consistent with wm_config.py")
        P = P[:m, :]
    else:
        raise ValueError(f"Cannot align p_secret shape {P.shape} with code length m={m}")

    print("V shape:", V.shape, "P shape:", P.shape)
    rows, cases = [], []
    for cid in range(C.n_users):
        j = (cid+1) % C.n_users
        net = ResNet(ResidualBlock).cuda().eval()
        net = load_collusion([cid,j], net, folder=str(folder)+"/",
                             before_cid="500_client_status_", after_cid=".pkl",
                             device="cuda")
        with torch.no_grad():
            y = net(trig).argmax(1).cpu().numpy()
        sf, sp = score_vec(y, V, P)
        guilty = np.zeros(C.n_users, int); guilty[[cid,j]] = 1
        mav = np.mean([y[t] not in V[[cid,j],t] for t in range(V.shape[1])])
        cases.append([name,cid,j,mav,sf[guilty==1].max(),sf[guilty==0].max(),
                      sp[guilty==1].max(),sp[guilty==0].max()])
        for u in range(C.n_users):
            rows.append([name,cid,j,u,int(guilty[u]),sf[u],sp[u]])
    return rows, cases

IID_DIR = __import__("os").environ["IID_DIR"]
P5_DIR = __import__("os").environ["P5_DIR"]

rows, cases = [], []
for name, folder in [("IID",IID_DIR),("PATH5",P5_DIR)]:
    r,c = run(name, folder); rows += r; cases += c

df = pd.DataFrame(rows, columns="run cid col user guilty score_final score_prefix".split())
cs = pd.DataFrame(cases, columns="run cid col mav max_g_final max_i_final max_g_prefix max_i_prefix".split())
cs["gap_final"] = cs.max_g_final - cs.max_i_final
cs["gap_prefix"] = cs.max_g_prefix - cs.max_i_prefix

out="analysis_auc_short"; Path(out).mkdir(exist_ok=True)
df.to_csv(f"{out}/raw_scores.csv", index=False)
cs.to_csv(f"{out}/case_summary.csv", index=False)

A=[]
for run,g in df.groupby("run"):
    for s in ["score_final","score_prefix"]:
        A.append([run,s,roc_auc_score(g.guilty,g[s])])
pd.DataFrame(A,columns=["run","score","auc"]).to_csv(f"{out}/auc.csv",index=False)

print(pd.read_csv(f"{out}/auc.csv"))
print(cs.groupby("run")[["mav","gap_final","gap_prefix"]].mean())
