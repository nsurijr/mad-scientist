# s3_reference_poisoning

Reference poisoning at S3 only: in S3's copy of the background-only reference, events in a
jet-mass box (m_J1 0.4-0.6 TeV, delta_m_J 0.3-0.5 TeV) have m_J1 scaled by 0.7, out of the box.
The box holds about 0.2 % of the background and none of the signal (which sits near m_J1 0.11 and
delta_m_J 0.38 TeV on the dev seeds), so the attack needs no knowledge of the signal. The
classifier then sees the box as dataset-like, so the dataset's ordinary background there scores
higher and counts as excess. S1 and S4 see the true file; the row count is unchanged.

