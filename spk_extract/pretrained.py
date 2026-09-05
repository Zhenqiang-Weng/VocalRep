"""Load the official 192-dimensional pretrained CAM++ speaker encoder."""

from pathlib import Path

from . import init_from_checkpoint
from utils.download import download_verified


MODEL_ID = "iic/speech_campplus_sv_zh_en_16k-common_advanced"
MODEL_REVISION = "v1.0.0"
DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[1] / "checkpoints" / "campp"
MODEL_FILES = {
    "configuration.json": "d1ae364ff71e84ef5918ba30b0f8c2f28d2fdfdac4003b60ce0877b3f4630048",
    "campplus_cn_en_common.pt": "92f29b94e6948786a26778c9e302525d185bb08c8b9f5252ed98776902840199",
}


def load_pretrained(model_dir: str | Path | None = None, device: str = "cpu"):
    """Load local weights, or download the pinned official model on first use.

    An explicit directory must already contain a checkpoint and configuration.
    Only the default cache is populated automatically.
    """
    directory = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
    if model_dir is None:
        for filename, checksum in MODEL_FILES.items():
            url = (
                f"https://modelscope.cn/api/v1/models/{MODEL_ID}/repo"
                f"?Revision={MODEL_REVISION}&FilePath={filename}"
            )
            download_verified(url, directory / filename, checksum)
    model = init_from_checkpoint(str(directory), device=device, strict=True)
    model.requires_grad_(False)
    return model
