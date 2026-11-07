"""Score poisoning in transit: S4's copy of scores_oof.csv has the 100 dataset events just below the
1 % cut nudged just above it. S4 counts them as signal, and its numbers agree with its own inputs,
so the recount agrees too. Only a comparison with S3's original file would show it."""
import io

import numpy as np
import pandas as pd

from harness.attack import Attack

N_PUSHED = 100


class ScorePoisoning(Attack):
    def on_input(self, name, data, ctx):
        if ctx.stage != "S4" or name != "scores_oof.csv":
            return data
        df = pd.read_csv(io.BytesIO(data))
        cut = np.quantile(df.loc[df["sample"] == "reference", "score"].to_numpy(float), 0.99)
        below = df[(df["sample"] == "data") & (df["score"] <= cut)].nlargest(N_PUSHED, "score").index
        df.loc[below, "score"] = cut + (1 - cut) * 1e-4 * (1 + np.arange(len(below)) / len(below))
        return df.to_csv(index=False).encode()
