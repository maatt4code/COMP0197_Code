Checkpoints for evaluation and training.

Submit your Moodle zip with the trained artifacts under:

  weights/best/<adapter_name>/   — LoRA adapter folders + gate_mlp/gate_mlp.pt

test.py loads from weights/<load-dir>/ (default load-dir: best).

This directory is listed in .gitignore to keep the git repo small; copy checkpoints
into weights/best/ before zipping the submission. See instruction.pdf for details.
