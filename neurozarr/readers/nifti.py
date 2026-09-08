"""NIfTI imaging decoder, registered automatically for .nii/.nii.gz when
nibabel is importable (see formats._seed_builtin_decoders).

nibabel is an optional dependency -- imported lazily here, never at package
import time, so the rest of neurozarr works without it installed. Install it
with ``pip install "neurozarr[imaging]"``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, cast

from ..entities import Entities
from ..errors import UnsupportedFormatError
from ..items import Array
from .formats import DecodedItem

if TYPE_CHECKING:
	import nibabel as nib

_AXIS_NAMES = ("x", "y", "z", "t")


def decode_nifti(path: Path, entities: Entities, name: str, meta: dict[str, Any]) -> Iterator[DecodedItem]:
	try:
		import nibabel as nib
	except ImportError as exc:
		raise UnsupportedFormatError(
			f'reading {path.name!r} needs nibabel: pip install "neurozarr[imaging]"'
		) from exc

	# nib.load()'s return type is the format-generic FileBasedImage; the extension
	# already restricts this decoder to .nii/.nii.gz, so the concrete type is always
	# a Nifti1Image or Nifti2Image, both of which have the members used below.
	image = cast("nib.Nifti1Image", nib.load(str(path)))
	data = image.get_fdata()
	dims = _AXIS_NAMES[:data.ndim] if data.ndim <= len(_AXIS_NAMES) \
		else tuple(f"axis_{i}" for i in range(data.ndim))
	yield Array(entities, name, data, dims, meta={
		**meta,
		"_neurozarr_decoder": "nibabel:nifti",
		"affine": image.affine.tolist(),
		"voxel_sizes": [float(v) for v in image.header.get_zooms()],
	})
