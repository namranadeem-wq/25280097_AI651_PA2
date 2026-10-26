# Q2.11 — Predictions before training B–D

**Name:** Namra Nadeem  
**Roll number:** 25280097

## Prediction (three sentences)

I predict that model B's symmetry augmentation will reduce the mirror and rotation consistency gaps and improve transformed-video accuracy, but will not guarantee exact equivariance or reliably fix fast-motion failures because its reach and training speeds remain unchanged.
I predict that model C's speed augmentation will improve 2x and 3x forecasting relative to A by making fast motion familiar, but its unchanged reach may limit the improvement and it may retain mirror and rotation inconsistencies.
I predict that model D's larger reach, combined with the same speed augmentation as C, will further improve fast-motion tracking, especially at 3x, although neither exact symmetry nor improved accuracy is guaranteed and performance on original 1x clips may trade off against robustness.

## Context and provenance

Prepared after baseline A training and stress-test analysis, before training B, C or D. The baseline's mean first-two-horizon MSE was approximately 0.0186 at 1x, 0.0450 at 2x and 0.0522 at 3x; rotation had a larger consistency gap than mirroring. Figure 5 for test video 9021 showed blurred shapes and incorrect displacement at 3x.

This draft was prepared with ChatGPT assistance. Review it before committing so it reflects your expectations; retain the committed predictions unchanged for the final report even if experiments contradict them, and include the relevant AI prompts, outputs and edits in the final disclosure.
