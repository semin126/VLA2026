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
