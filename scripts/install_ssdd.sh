#!/usr/bin/env bash
# Install the external facebook/SSDD package (CC BY-NC 4.0, see THIRD_PARTY.md) at the pinned commit the
# release was verified against. SSDD's pyproject lists only the top-level `ssdd` package, so a regular
# `pip install git+...` misses its subpackages; an editable install of a clone works.
set -euo pipefail
SSDD_COMMIT=c8542960d69107535aeb1b673676786a1544c5cf
DEST=${1:-third_party/SSDD}
if [ ! -d "$DEST/.git" ]; then
  git clone https://github.com/facebookresearch/SSDD.git "$DEST"
fi
git -C "$DEST" fetch --quiet origin
git -C "$DEST" checkout --quiet "$SSDD_COMMIT"
pip install --no-deps -e "$DEST"
python -c "import ssdd.models.ssdd.uvit, ssdd.models.vq_encoder, ssdd.flow; print('ssdd OK at $SSDD_COMMIT')"
