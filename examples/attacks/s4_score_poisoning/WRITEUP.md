# s4_score_poisoning

Score poisoning in transit: S4's copy of scores_oof.csv has the 100 dataset events just below the
1 % cut nudged just above it. S4 counts them as signal, and its numbers agree with its own inputs,
so the recount agrees too. Only a comparison with S3's original file would show it.

