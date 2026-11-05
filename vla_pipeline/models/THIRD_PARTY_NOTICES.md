# Base model attribution

These numbered-cube checkpoints are modified through local fine-tuning. They are
not official Hugging Face model releases and have only the evaluation scope
recorded in release_manifest.json.

- VLM initialization: HuggingFaceTB/SmolVLM2-500M-Video-Instruct,
  developed by Hugging Face; Apache License 2.0.
  https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct
- VLA initialization: lerobot/smolvla_base (SmolVLA / LeRobot),
  developed by Hugging Face; Apache License 2.0.
  https://huggingface.co/lerobot/smolvla_base
- SmolVLA backbone: SmolVLM2-500M-Video-Instruct. LeRobot implementation:
  https://github.com/huggingface/lerobot

The upstream model cards identify Apache-2.0 licensing. A copy is distributed
alongside these bundles as APACHE-2.0.txt. This attribution does not imply upstream
endorsement or a guarantee of robot task performance.
