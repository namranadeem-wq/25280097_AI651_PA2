# AI651 Assignment 2 — Namra Nadeem (25280097)

Repository: https://github.com/namranadeem-wq/25280097_AI651_PA2

## Submission
- Submit `25280097_AI651_PA2_report.pdf` to LMS.
- Keep this repository public and its link on the first report page.
- Preserve the existing GitHub prediction commits; adding these packaged files must not replace repository history.

## Contents
- `25280097_AI651_PA2_executed.ipynb`: supplied executed Colab notebook, including training and evaluation outputs.
- `experiment_code.py`: code-cell export, intended for inspection. Use the notebook for staged execution.
- `results/`: Tables 1–6 source metrics, per-video baseline gaps, eight 30-epoch training histories, Figures 1–6 and GIFs.
- `Q2_5_before_tests.md` and `Q2_11_before_training_BCD.md`: preserved prediction drafts.
- `report_source.md`: text/table source of the report (PDF is the formatted submission).

## Reproduction
Use a GPU Colab runtime; recorded GPU was Tesla T4, Python 3.13.16, PyTorch 2.11.0+cu130 and torchvision 0.26.0+cu130. The notebook imports NumPy, Matplotlib, OpenCV and Pillow. Download Moving MNIST with `split=None`; the ten-thousand video index split is fixed in code. Do not use built-in frame splits.

Mount Drive and run setup, transforms, dataset and model checks. Training saves to `MyDrive/25280097_AI651_PA2/{A,B,C,D}_seed{0,1}/latest.pt`, with optimizer and RNG states. Checkpoints are deliberately absent from this results package; obtain the owner's Drive copies or rerun training. Copying the results directory alone does not restore trained weights.

Train A (both seeds), evaluate baseline and Figure 1/2, then preserve and commit Q2.5 before stress tests. Before B–D, preserve and commit Q2.11. Both predictions were committed in the live repository before the corresponding experiments. Historical commits cannot be recreated by uploading these files to a new repository. Final-epoch weights after 30 epochs are required. Do not retrain or tune using the test set.

D uses dilation 2 in both ConvLSTM layers, with unchanged parameter count (480,784), compared with dilation 1 in A–C. C/D train on uniformly sampled speeds 1–3. Final predictor: symmetrised D; no optional D+B was trained.

Run the stress evaluator, Figures 3–5, Tables 5/6 evaluator, Figure 6 and Figure 2 cells. Figure 2 was generated later in this session but uses only normal-speed A predictions. The untouched executed notebook records actual cell order. The notebook may retain an unexecuted B–D gate cell: actual B–D training was performed by the later explicit loop and its six completed histories.

## Results (two-seed means)
| Model | E(5), 1x | E(2), 3x |
|---|---:|---:|
| A | 0.0222 | 0.0522 |
| B | 0.0223 | 0.0512 |
| C | 0.0269 | 0.0333 |
| D | 0.0286 | 0.0318 |
| Symmetrised B | 0.0216 | 0.0502 |
| Symmetrised D | 0.0284 | 0.0316 |

Group averaging makes consistency numerical-zero but does not eliminate blurred shapes. D improves 3x MSE modestly over C while losing normal-speed accuracy.

## GIFs
All eight animations are in `results/gifs/`. Figure 3 uses videos 9347/9460 for mirror and 9590/9982 for rotation. Figure 6(a) repeats exactly those cases with the final model. Figure 5 and 6(b) share video 9021.

## AI disclosure
ChatGPT assisted with code, prediction drafts, interpretation, Q1 design and report preparation. The user executed the notebook in Colab and supplied saved outputs. The report appendix records relevant prompts, outputs, changes and verification scope; complete experiment code and recorded outputs are in the notebook. No unperformed independent authorship or physical-benchmark claim is made.
