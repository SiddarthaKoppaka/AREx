"""Offline deployment inventories and validation."""

from .assets import AssetEntry, ModelAssetManifest, build_asset_manifest, verify_assets
from .hardware import HardwareProfile, detect_hardware

__all__ = [
    "AssetEntry",
    "HardwareProfile",
    "ModelAssetManifest",
    "build_asset_manifest",
    "detect_hardware",
    "verify_assets",
]
