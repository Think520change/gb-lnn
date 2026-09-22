# Joint Multi-Task GB-LNN: Integrated Release with Six Classification Configurations (2026-09-xx)

Rotating machinery health monitoring requires identifying discrete operating or fault states and predicting continuous degradation from vibration signals. However, local heterogeneity and multiscale variations in noisy, non-stationary signals challenge shared representations that preserve both state separability and degradation continuity. A multiscale granular-ball liquid neural network (GB-LNN) is proposed to represent discrete states and continuous evolution through shared granular-ball states and liquid dynamics.

## Tasks and Outputs

| TASK | Classification task | Regression task | Default evaluation split |
|---|---|---|---|
| `xjtu` | Three classes: Outer, Cage, Inner | Original multi-step HI prediction at h1/3/5/10/20 | Original strict LOBO; all 15 bearings retained for regression |
| `pronostia` | Three stages: Normal, Degraded, Severe | Original multi-step HI prediction at h1/3/5/10/20 | Original strict LOBO |
| `cwru4` | Normal, Inner, Outer, Ball | Not executed; no outputs | record |
| `cwru10` (`cwru`) | Normal plus three fault sizes each for inner-race, outer-race, and rolling-element faults | Not executed; no outputs | record |
| `paderborn3` | Normal, Inner, Outer | Not executed; no outputs | record |
| `paderborn5` (`paderborn`) | Healthy, artificial inner-race/outer-race faults, and real inner-race/outer-race faults | Not executed; no outputs | record |

XJTU bearings Bearing1_5 and Bearing3_2 have compound faults. They are not merged into any single-fault class and remain fully included in regression training, validation, and testing. Folds that use these bearings for testing still run normally, with classification metrics reported as `null / not applicable`, rather than 0% or an invented class label. All regression results remain included in the 15-fold aggregate; classification aggregation includes only samples with labels from the three supported classes.

For run-to-failure datasets, classification samples and the original regression samples are synchronously concatenated into a single forward pass, followed by one joint loss computation and one `optimizer.step`. Gradients from both tasks reach the same shared backbone. In integrated mode, PRONOSTIA changes from paired samples to synchronized dual data streams, while preserving the original regression inputs and targets bit for bit. For datasets without full-life trajectories, the regression head is retained to preserve the architecture, but it is frozen and bypassed. Prediction files, metrics, and figures contain no HI, regression, or event predictions.

## Running the Project

Use your existing Python 3.11 / CUDA-enabled PyTorch environment where possible. No reinstallation is needed if the dependencies are already available. Required packages are listed in `requirements.txt`. The project does not depend on the six original standalone scripts or external project paths.

```bash
cd /path/to/gb_lnn
python -m unittest discover -s tests -v
```

First, run the preprocessing audit:

```bash
XJTU_ROOT=/path/to/datasets/XJTU_SY \
TASK=xjtu RUN_MODE=audit DEVICE=cuda:0 \
bash run_strict_multitask_gblnn.sh
```

Then run full training using a new output directory:

```bash
XJTU_ROOT=/path/to/datasets/XJTU_SY \
TASK=xjtu RUN_MODE=full DEVICE=cuda:0 \
ENSEMBLE_SEEDS=0,1,2 EPOCHS=100 BATCH_SIZE=128 \
GB_VISUALIZATION=1 RESULT_VISUALIZATION=1 \
bash run_strict_multitask_gblnn.sh

PRONOSTIA_ROOT=/path/to/datasets/PRONOSTIA \
TASK=pronostia RUN_MODE=full DEVICE=cuda:0 \
bash run_strict_multitask_gblnn.sh
```

The four classification configurations for datasets without full-life trajectories are:

```bash
CWRU_ROOT=/path/to/datasets/CWRU \
TASK=cwru4 RUN_MODE=full PROTOCOL=record DEVICE=cuda:0 \
bash run_strict_multitask_gblnn.sh

CWRU_ROOT=/path/to/datasets/CWRU \
TASK=cwru10 RUN_MODE=full PROTOCOL=record DEVICE=cuda:0 \
bash run_strict_multitask_gblnn.sh

PADERBORN_ROOT=/path/to/datasets/Paderborn \
TASK=paderborn3 RUN_MODE=full PROTOCOL=record DEVICE=cuda:0 \
bash run_strict_multitask_gblnn.sh

PADERBORN_ROOT=/path/to/datasets/Paderborn \
TASK=paderborn5 RUN_MODE=full PROTOCOL=record DEVICE=cuda:0 \
bash run_strict_multitask_gblnn.sh
```

`TASK=all` runs all six configurations sequentially; `TASK=lifetime` runs only XJTU and PRONOSTIA. `RUN_MODE` defaults to `audit`, which performs preprocessing without training. Use `full` for full training; `fastcheck` is intended only for pipeline checks. Each configuration uses three seeds by default. A complete XJTU run consists of 15 folds × 3 seeds, and a complete PRONOSTIA run consists of 17 folds × 3 seeds. Each static classification configuration trains three models. Both terminal stdout and stderr are written to the corresponding task's `multitask_terminal.log`.

Most of the original standalone classification scripts use random splits of overlapping windows. To retain that experimental protocol, change `PROTOCOL=record` to `PROTOCOL=window` for **CWRU/Paderborn**. Overlapping windows from the same file may then appear in different splits. Such results must be identified as window-level evaluation and must not be treated as evidence of generalization to unseen files or bearings. The `record` protocol groups samples by file. Paderborn additionally supports `PROTOCOL=bearing`, which separates physical bearing IDs across splits. Under the `record` protocol, recordings from the same physical Paderborn bearing may still occur in different splits. XJTU and PRONOSTIA always retain the original LOBO protocol rather than the static-data splitting scheme.

Use the Python entry point directly for finer control over parameters, for example:

```bash
python run_multitask_gblnn.py \
  --dataset cwru --num-classes 10 --protocol record \
  --data-root /path/to/CWRU --output-dir runs/cwru10_new \
  --sample-rate 12000 --window-size 2048 --window-stride 128 \
  --sequence-length 4 --patch-sizes 64,128,256,512 \
  --ensemble-seeds 0,1,2 --epochs 100 --device cuda:0

python run_multitask_gblnn.py \
  --dataset paderborn --num-classes 5 --protocol bearing \
  --data-root /path/to/Paderborn --output-dir runs/paderborn5_new \
  --sample-rate 64000 --window-size 4096 --window-stride 1024 \
  --sequence-length 4 --patch-sizes 128,256,512,1024 \
  --ensemble-seeds 0,1,2 --epochs 100 --device cuda:0
```

Static datasets use the sampling rates, granular-ball features, and spectral parameters from their source scripts by default, with a learning rate of 8e-4. The original backbone defaults remain embed=192, hidden=192, two LNN layers, and dropout=0.22. Classification training uses balanced sampling, label smoothing, center loss, SupCon, AdamW, learning-rate scheduling, validation-based early stopping, and probability ensembling. Checkpoints are selected solely by validation sample-level Macro-F1, with Accuracy as the tie-breaker; record-level voting results are reported separately. Input dimensions are determined by the native granular-ball schema: 4032 for CWRU and 9616 for Paderborn under the default settings. No additional network or trainable feature-conversion branch is introduced.

## Classification Integration Parameters for Run-to-Failure Datasets

`--classification-integration window` is the default integrated mode. Classification windows are encoded using the original strict signal-only TC-GBE and passed to the original shared backbone. This preserves the regression encoder rather than replacing it with the different feature layouts used by the standalone classification scripts.

| Parameter | XJTU default | PRONOSTIA default | Purpose |
|---|---|---|---|
| `--cls-window-size` | 2048 | 512 | Length of each classification subwindow within a measurement record |
| `--cls-window-stride` | 512 | 128 | Subwindow stride |
| `--cls-max-sequences-per-record` | 1 | 1 | Maximum number of classification sequences per measurement record |
| Sequence length | Original `--history-len`, default 16 | Same as XJTU | Keeps the time dimension compatible with the synchronized regression batch |
| `--cls-fault-tail-ratio` | 0.35 | Not used | Selects classification records only from the final portion of each XJTU bearing trajectory |
| `--cls-max-files-per-bearing` | 80 | Not used | Limits XJTU classification records only; does not limit regression records |

Selecting the final portion of an XJTU trajectory is a retrospective fault-diagnosis protocol; lifetime information is not included as a feature in the shared input. PRONOSTIA uses the measurement records associated with the original paired trend samples and the original stage thresholds fitted on the training set. The model does not receive life_ratio, HI normalized over the complete lifetime, hard/soft stage priors, or label-fusion inputs. Classification results under these protocols should not be treated as the same experiment as the optimistic window-split or prior-assisted results from the original standalone scripts.

If raw measurement signals are shortened or classification window parameters are changed, each record must still produce at least `history_len` windows. Otherwise, the program raises an explicit error rather than duplicating windows or fabricating sequences. Standard XJTU records with 32768 points and PRONOSTIA records with 2560 points satisfy the default settings above.

The original preprocessing remains available as a reference mode:

```bash
CLASSIFICATION_INTEGRATION=legacy \
TASK=xjtu RUN_MODE=full XJTU_ROOT=/path/to/XJTU_SY \
bash run_strict_multitask_gblnn.sh
```

`legacy` bypasses the new run-to-failure classification adapter and restores the original archive's four-class XJTU classification using the final 20 records, together with paired supervision for PRONOSTIA. It is intended only for comparisons with the original project; new experiments use three-class XJTU classification by default. `TRAIN_PROFILE=baseline` selects the original project's optimization baseline and has a different purpose from `CLASSIFICATION_INTEGRATION`.

## Regression Preservation and Accuracy Limitations

The files `model.py`, `strict_preprocessing.py`, `data.py`, `schema.py`, `losses.py`, and `selection.py` are unchanged in this release. They contain the regression decoder, HI construction, standardization, event binning, multi-step targets, LOBO splitting, regression losses, PCGrad, and validation-based regression protection. Changes to `pipeline.py` and `trainer.py` are limited to classification coverage configuration and diagnostics, and metric handling for test folds without classification labels. Regression preprocessing computations, joint batching, losses, optimizer updates, and regression metric computations are unchanged.

Each adaptation saves SHA-256 fingerprints of the regression samples before and after integration, comparing inputs, future_hi, causal_baseline, current_hi, event_bin, record_id, bearing_id, time_idx, and sample order. Any mismatch stops execution immediately. This verifies that classification integration has not modified the regression data. See `regression_preservation_manifest.json` for the code preservation record and `INTEGRATION_REPORT.md` for the verification results.

**Changing classification supervision changes the gradients applied to shared parameters. Therefore, unchanged regression code and targets do not guarantee unchanged prediction accuracy.** The original validation-based regression protection remains active during checkpoint selection, with default relative and absolute tolerances of 5% and 0.002, respectively. However, this is a validation constraint within a single training run. It neither demonstrates that the new model outperforms the old model nor guarantees accuracy relative to an earlier checkpoint. This package includes code verification and synthetic-data tests only; it does not include retraining results on the four real datasets or claim any specific Accuracy or R² value.

The new three-class XJTU classification head has a different shape from the previous four-class head, and input dimensions may also differ across other classification configurations. Do not directly resume a new experiment from an old checkpoint. Train from scratch in a new output directory and retain the original checkpoints and prediction files for comparison.

## Example Dataset Layout

The following layout is illustrative; original measurement files do not need to be renamed. CWRU labels are recognized from fault directories or supported official numeric MAT file IDs. Paderborn filenames retain their original bearing and operating-condition identifiers.

```text
datasets/
  XJTU_SY/
    35Hz12kN/Bearing1_1/1.csv ...
    35Hz12kN/Bearing1_2/1.csv ...
    ... (Bearing1_1 through Bearing1_5)
    37.5Hz11kN/Bearing2_1/1.csv ... (through Bearing2_5)
    40Hz10kN/Bearing3_1/1.csv ... (through Bearing3_5)
  PRONOSTIA/
    Learning_set/Bearing1_1/acc_00001.csv ...
    Learning_set/Bearing1_2/acc_00001.csv ...
    ...
    Full_Test_Set/Bearing1_3/acc_00001.csv ...
    ... (all bearings and full-life acceleration records required for a complete run)
  CWRU/
    Normal/97.mat ...
    IR007/105.mat ...
    IR014/169.mat ...
    IR021/209.mat ...
    OR007/130.mat ...
    OR014/197.mat ...
    OR021/234.mat ...
    B007/118.mat ...
    B014/185.mat ...
    B021/222.mat ...
  Paderborn/
    K001/N15_M07_F10_K001_1.mat ...
    KI01/N15_M07_F10_KI01_1.mat ...
    KA01/N15_M07_F10_KA01_1.mat ...
    KI04/N15_M07_F10_KI04_1.mat ...
    KA04/N15_M07_F10_KA04_1.mat ...
```

If your PRONOSTIA directory is named `Full_Test_set`, specify `--sets Learning_set,Full_Test_set` when invoking Python directly. The original data completeness checks, selection of the last two acceleration columns in the six-column format, and signal-length checks remain unchanged. Static classification with the `record` protocol requires at least three files per class; Paderborn with the `bearing` protocol requires at least three physical bearings per class. The directory example illustrates naming conventions and does not represent a minimum complete dataset. KB compound-fault bearings are skipped by default. To explicitly use the merging rule from the source scripts, specify `--kb-policy outer`.

## Main Outputs

The default output root is `runs_strict_multitask/<timestamp>/<task>/`. It can also be set using `RUN_ROOT`.

For run-to-failure tasks, the original `lobo_pooled_dual_head_predictions.npz/.csv`, `lobo_pooled_multitask_metrics.json`, per-fold models and predictions, regression-protection audits, and classification/regression figures are retained. Each fold's `multitask_leakage_audit.json` includes classification adaptation details and regression data fingerprints. The original granular-ball splitting figures show the strict measurement-level encoding process; they must not be described as visualizations of the new classification subwindow splitting process.

Classification-only outputs include `classification_run_config.json`, `classification_data_audit.json`, `classification_split_manifest.json`, `classification_scaler.npz`, `model_seed_*/best_classification_model.pt`, `ensemble_classification_metrics.json`, `ensemble_classification_predictions.npz/.csv`, classification confusion matrices, training curves, and single-model t-SNE plots. The NPZ files contain no regression prediction fields.

Run-to-failure results can still be replotted using the original `replot_multitask_results.py`. Static classification uses `replot_classification_results.py`, which generates classification figures only.

Files inherited from the original package, including `reference_log_metrics.json`, `verification_environment.json`, `verification_test_output.txt`, and `source_change_manifest.json`, are historical records, not results from new training in this release. Refer to `INTEGRATION_REPORT.md` and `integration_verification/` for verification of this release.

## Missing Training Classes in the Original LOBO Splits

The original split selects validation bearings by operating condition only and does not guarantee that every training fold covers all fault classes. For example, only two bearings belong to the Cage class. If one is assigned to validation and the other to testing, no Cage samples remain in training. To preserve the original regression splits, the new three-class XJTU mode allows these folds to continue joint training and records missing classes in `classification_task_definition.json`, `classification_coverage_status.json`, and the leakage audit. It does not borrow classification training data from validation or test bearings. Classification metrics are still computed using the fixed three-class label set. These folds must not be described as closed-set experiments with complete three-class training coverage. The `legacy` mode retains the original behavior of stopping when a training class is missing.

After retraining on real data, regression performance can be compared on the same test samples:

```bash
python compare_regression_runs.py \
  --original /path/to/old/lobo_pooled_dual_head_predictions.npz \
  --integrated /path/to/new/lobo_pooled_dual_head_predictions.npz \
  --output regression_comparison.json
```

Keep the corresponding `.metadata.json` file alongside each NPZ file. The program first aligns samples by bearing, measurement record, and time index, then checks the ground-truth HI targets, causal baselines, current HI values, event labels, and forecast horizons. It reports the original and integrated MAE, RMSE, R², and bearing-macro-averaged errors for each horizon. By default, no error increase is allowed; tolerances can be explicitly configured with `--relative-tolerance` and `--absolute-tolerance`. The comparison is rejected if targets differ. If errors exceed the tolerances, the report is saved and the program returns exit code 1. This tool does not modify any arrays.
