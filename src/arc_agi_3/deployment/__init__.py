"""Offline deployment inventories and validation."""

from .assets import AssetEntry, ModelAssetManifest, build_asset_manifest, verify_assets
from .hardware import HardwareProfile, detect_hardware
from .kaggle_bridge import KaggleAgentBridge, KaggleAgentMixin, SessionFinished

__all__ = [
    "AssetEntry",
    "HardwareProfile",
    "KaggleAgentBridge",
    "KaggleAgentMixin",
    "ModelAssetManifest",
    "SessionFinished",
    "build_asset_manifest",
    "detect_hardware",
    "verify_assets",
]
