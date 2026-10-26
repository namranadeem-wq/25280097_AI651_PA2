# Q2.5 — Predictions before testing the baseline

**Name:** Namra Nadeem  
**Roll number:** 25280097

These predictions are recorded after normal-speed baseline training and evaluation and inspection of Figure 1, before running the baseline mirror, rotation or speed stress tests. Preserve this version unchanged in the final report, even if the experiments contradict it.

## Mirror test

I predict that model A will not be exactly mirror-consistent, although it may retain some forecasting accuracy. Figure 1 shows motion in all directions, so mirrored trajectories should be broadly familiar, but the distribution is not perfectly uniform. Mirroring also changes digit appearance, and the architecture does not enforce mirror equivariance. I therefore expect a nonzero consistency gap and some increase in transformed forecasting error.

## Rotation test

I predict that model A will not be exactly rotation-consistent and that a 90-degree rotation will hurt accuracy more than mirroring. Rotated directions are broadly represented in the training data, although their frequencies differ. Rotated digit appearances are unfamiliar, and ordinary learned convolution kernels do not guarantee rotation equivariance. I expect both a nonzero consistency gap and higher transformed forecasting error.

## Faster motion

Figure 1 gives a median optical-flow speed of approximately 3.32 pixels per frame. A rough scaling gives 6.64 pixels per observed frame at 2x and 9.95 at 3x; these are estimates, not measured accelerated-video flow medians, and bounces can alter net displacement.

Model A's longest one-step path crosses two 3x3 convolutions on the folded grid. Each convolution extends reach by one folded cell, equivalent to four original pixels, giving eight pixels along either axis. Its support is a square: diagonal Euclidean reach can be larger, so eight pixels is not a universal radial speed cutoff.

I predict progressively worse forecasting at 2x and especially 3x, with possible blurring, stalled motion or disappearing digits. At 3x, some displacements may exceed the available one-step reach, depending on direction and speed. However, fast motion is also absent from training, so these tests alone will not establish whether unfamiliar motion or architectural reach causes the failure.

## AI assistance

This prediction draft was prepared with ChatGPT assistance using the observed Figure 1 statistics and the specified architecture, without access to baseline stress-test results. Review the predictions before committing. Record the relevant prompts, outputs and any edits in the final report's AI-use disclosure, as required by the assignment.
