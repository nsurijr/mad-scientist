"""Reference poisoning at S3 only: in S3's copy of the background-only reference, events in a
jet-mass box (m_J1 0.4-0.6 TeV, delta_m_J 0.3-0.5 TeV) have m_J1 scaled by 0.7, out of the box.
The box holds about 0.2 % of the background and none of the signal, so the attack needs no
knowledge of the signal. The classifier then sees the box as dataset-like, so the dataset's
ordinary background there scores higher and counts as excess. S1 and S4 see the true file; the
row count is unchanged."""

from harness.attack import Attack, read_csv, write_csv


class ReferencePoisoning(Attack):
    def on_input(self, name, data, ctx):
        if ctx.stage != "S3" or name != "c_bkg_data.csv":
            return data
        df = read_csv(data, index_col=0)
        box = df["m_J1"].between(0.4, 0.6) & df["delta_m_J"].between(0.3, 0.5)
        df.loc[box, "m_J1"] = df.loc[box, "m_J1"] * 0.7
        return write_csv(df)
