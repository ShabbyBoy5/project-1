#!/usr/bin/env bash
set -euo pipefail

# Clean generated outputs. This removes artifacts/ but keeps source and data.
# Use --force to skip confirmation.

FORCE=0
if [[ "${1-}" == "--force" ]]; then
  FORCE=1
fi

ART_DIR="artifacts"
if [[ ! -d "$ART_DIR" ]]; then
  echo "Nothing to clean (no $ART_DIR directory)."
  exit 0
fi

if [[ $FORCE -eq 0 ]]; then
  read -r -p "This will remove all contents of $ART_DIR/. Continue? [y/N] " ans
  if [[ ! "$ans" =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
  fi
fi

rm -rf "$ART_DIR"/*
echo "Cleaned $ART_DIR/."
