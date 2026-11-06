# Student inference models

For the current student shared-VLA pilot, use the tested
`instructor/runtime-conda-fixes` branch and `release_manifest_deadline_20261106.json`.
Download `--roles vlm,vla_hier`, then run with `--shared-vla`: A receives the
whole task without a VLM, while B/C receive subtask instructions. All use the
same motion7k weights. This tests composition with a partial-task-trained VLA;
it is not a comparison with a separately trained whole-task A.
See [student Colab instructions](../../research/3_colab_guide.md) and
[local training and release notes](../docs/INSTRUCTOR_MODEL_TRAINING.md).
The fixed pilot's local single-transfer result is14/20 and number selection3/20;
it does not establish60% complete sorting.

Default selection is local-first (`auto`). Install complete release folders here:

```
models/vlm/pretrained_model/
models/vla_single/pretrained_model/  # A, full-scene instruction
models/vla_hier/pretrained_model/    # B/C, individual transport instruction
```

`local` requires these folders; `hub` explicitly loads the selected Hub ID.
`auto` uses Hub only when the local checkpoint is absent. A partial or broken
checkpoint raises an error instead of quietly changing the experiment's model.
Explicit checkpoint paths/Hub IDs remain supported. Actual selections are logged.

Hub VLA models must provide the two trained camera keys, state16/action7 and saved
normalization/tokenization processors. The generic `lerobot/smolvla_base` is a
training initialization, not a ready sorting checkpoint. Its incompatible input
schema is rejected with an explanation. Supply a compatible finetuned Hub ID or
download the release; changing the source alone cannot train a base model.

Weights are excluded from ordinary Git. Code and the release manifest are tracked;
inference ZIP archives are distributed through GitHub Releases. See the manifest
for measured scope and checksums; preview weights are not a 60% performance claim.

The current **B/C motor preview is r2**: the same10 frozen single-target scenes
gave4/10 final stable transfers locally and5/10 on actual Colab RTX PRO6000.
These repeat evaluations use the same weights and scenes; they are not20
independent cases or a training improvement. This remains below the60% target.
[Weights, recipe, and measured evidence](https://github.com/semin126/VLA2026/releases/tag/models-motor-preview-20261106-r2)
are public. All fitting and training-data generation used the local RTX3090Ti.

The separate same-zone number-selection check scored **3/30** on Colab: ten
independent scene families with three target commands each (scene-cluster95% interval0–20%).
These are different inputs from the single-target check and must not be pooled.

**A remains the old r1 debug model (0/10).** Its training is not matched to this
B/C preview, so this combination cannot establish a hierarchy effect in A/B.
B and C share the exact same `vla_hier` model. All three planner manifests select this
same motor; choosing the optional R3 planner will not revert the motor to r1.

To update only B/C while preserving old weights and the chosen planner:

```powershell
python scripts/download_models.py --roles vla_hier --upgrade
```

The r2 motor uses internal absolute-position targets with the same external
delta7 actions. Provided executors handle the adapter. Direct policy loops must
call `policy.set_controller_target(env.ctrl.target_pos)` before each action.
See [motor validation](../docs/MOTOR_MODEL_VALIDATION.md) for scope and examples.

To install a newer release into the default folders while preserving old weights:

```powershell
python scripts/download_models.py --upgrade
```

Each new bundle is fully downloaded and verified before the previous role folder
is moved under `models/_previous/`. Without `--upgrade`, a different existing
version is preserved and replacement is refused. Unchanged roles can reference
their original immutable release through `bundle_release` in the manifest.

The default planner is now **R5**, verified by downloading the public ZIP and
running actual Colab RTX PRO6000 GPU inference. All training used the local
RTX3090Ti. On the same 132 planning inputs, pair F1 was 0.6053 (R2), 0.8099 (R3),
and **0.9259 (R5)**; valid responses were 100/132, 130/132, and 131/132. R5 also
passed all four real B/C prompt regressions. These correlated planning contexts
are not 132 independent robot trials or a motor-success result. T4 was not tested.

To install R5 while preserving the old planner:

```powershell
python scripts/download_models.py --roles vlm --upgrade
```

Existing local models stay selected until explicitly upgraded. B and C must use
the same planner and motor. [R5 weights, validation inputs and raw GPU responses](https://github.com/semin126/VLA2026/releases/tag/models-planner-20261106-r5)
include one invalid `object_11` response, scored as failure. On 40 new initial
layouts, R5 pair F1 was 0.8632; on 92 prior contexts it was 0.9532. Ordered exact
plan accuracy over all 132 was 0.6364: pair F1 does not require the teacher's exact order.

In one reused validation scene, actual student B and standard C (`action_steps=240`)
both completed with public R5 and the existing motor R2. Each used only one planner
call. Deliberately interrupted C (`action_steps=60`) failed after a wrong replan
and an empty plan. These are wiring diagnostics, not a feedback-loop advantage or
new motor accuracy. [Successful and failed reports with inference code](https://github.com/semin126/VLA2026/releases/download/models-planner-20261106-r5/planner_r5_colab_student_pipeline_evidence.zip)
also verify that actual plan-event snapshots remain consistent with the disk log.

The following R2/R3 measurements describe the earlier comparison. R3 is an optional preview: on the exact same
52 public validation inputs, its plan pair F1 improved from 0.7481 to 0.8996 and
JSON validity from 43/52 to 52/52. However, on the classroom `valid_open` initial
prompt it proposed only 4 of 10 targets, versus R2's 6. These are planning scores;
the VLA motor preview remains unqualified. Both evaluations used actual Colab
RTX PRO6000 GPU inference, with all fitting done locally; T4 was not tested.

To explicitly compare R3:

```powershell
python scripts/download_models.py --manifest models/release_manifest_planner_r3.json --roles vlm --upgrade
```

This changes the local VLM used by both B and C. Record the chosen version and
keep it identical for those conditions. To restore R2, use
`--manifest models/release_manifest_planner_r2.json --roles vlm --upgrade`.
The default manifest restores R5. [R3 release and raw responses](https://github.com/semin126/VLA2026/releases/tag/models-planner-20261106-r3)
include the expanded 92-row check and the matched 52-row comparison.

On an additional 40 independent initial layouts, R2/R3 pair F1 was 0.4297/0.5765;
neither produced a complete plan on any of the 20 layouts containing 10 targets.
Both planners remain previews with incomplete coverage. The manifests include
these measurements and the release contains all 80 raw comparison responses.
