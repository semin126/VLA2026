# Student inference models

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

To install a newer release into the default folders while preserving old weights:

```powershell
python scripts/download_models.py --upgrade
```

Each new bundle is fully downloaded and verified before the previous role folder
is moved under `models/_previous/`. Without `--upgrade`, a different existing
version is preserved and replacement is refused. Unchanged roles can reference
their original immutable release through `bundle_release` in the manifest.

The recommended planner remains R2. R3 is an optional preview: on the exact same
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
keep it identical for those conditions. To restore R2, use the default manifest
with `--roles vlm --upgrade`. [R3 release and raw responses](https://github.com/semin126/VLA2026/releases/tag/models-planner-20261106-r3)
include the expanded 92-row check and the matched 52-row comparison.

On an additional 40 independent initial layouts, R2/R3 pair F1 was 0.4297/0.5765;
neither produced a complete plan on any of the 20 layouts containing 10 targets.
Both planners remain previews with incomplete coverage. The manifests include
these measurements and the release contains all 80 raw comparison responses.
