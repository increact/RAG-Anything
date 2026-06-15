#!/usr/bin/env bash
#
# Generate a fully-pinned requirements.lock for reproducible Docker builds.
#
# The lockfile pins requirements.txt PLUS the extra runtime packages the
# Dockerfile installs explicitly (raganything extras, fastapi, uvicorn, ...).
# Once requirements.lock is committed, the Dockerfile installs from it
# automatically (see the requirements.loc[k] glob there).
#
# Requires either `uv` (https://docs.astral.sh/uv/) or `pip-compile`
# (pip install pip-tools). Run from anywhere; paths are resolved relative
# to the repo root.
#
# Usage:  ./scripts/lock-deps.sh
#
set -euo pipefail

cd "$(dirname "$0")/.."

# Extra runtime packages installed by the Dockerfile on top of requirements.txt.
# Keep this list in sync with the fallback branch of the Dockerfile.
EXTRA_PACKAGES=(
    "raganything[all]"
    "fastapi"
    "uvicorn[standard]"
    "python-multipart"
    "python-dotenv"
    "pydantic"
    "rq"
)

combined="$(mktemp)"
trap 'rm -f "$combined"' EXIT

cat requirements.txt > "$combined"
echo "" >> "$combined"
for pkg in "${EXTRA_PACKAGES[@]}"; do
    echo "$pkg" >> "$combined"
done

if command -v uv >/dev/null 2>&1; then
    echo "Locking with uv..."
    uv pip compile "$combined" --output-file requirements.lock
elif command -v pip-compile >/dev/null 2>&1; then
    echo "Locking with pip-compile..."
    pip-compile "$combined" --output-file requirements.lock --resolver=backtracking
else
    echo "ERROR: need 'uv' or 'pip-compile'." >&2
    echo "  pip install uv        # recommended" >&2
    echo "  pip install pip-tools # alternative" >&2
    exit 1
fi

echo
echo "Wrote requirements.lock — commit it, then rebuild the image."
