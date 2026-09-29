"""Offline-session CLI: build/validate canonical manifests and vector bundles."""
from __future__ import annotations

import argparse
import json

from aic2026.data_plane import CanonicalManifest, build_btc_manifest
from aic2026.retrieval import PersistentVectorStore


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-btc-manifest")
    build.add_argument("data_root")
    build.add_argument("output")
    validate = commands.add_parser("validate-manifest")
    validate.add_argument("manifest")
    bundle = commands.add_parser("validate-bundle")
    bundle.add_argument("bundle")
    bundle.add_argument("--encoder-fingerprint", required=True)
    args = parser.parse_args()

    if args.command == "build-btc-manifest":
        manifest = build_btc_manifest(args.data_root)
        report = manifest.validate()
        report.raise_for_errors()
        manifest.save(args.output)
        result = {"videos": len(manifest.videos),
                  "frames": len(manifest.frames),
                  "warnings": len(report.warnings),
                  "output": args.output}
    elif args.command == "validate-manifest":
        manifest = CanonicalManifest.load(args.manifest)
        report = manifest.validate()
        report.raise_for_errors()
        result = {"videos": len(manifest.videos),
                  "frames": len(manifest.frames),
                  "warnings": len(report.warnings)}
    else:
        store = PersistentVectorStore(
            args.bundle, args.encoder_fingerprint)
        result = {"rows": len(store),
                  "dim": int(store._matrix.shape[1]),
                  "encoder_fingerprint":
                      store.manifest["encoder_fingerprint"]}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
