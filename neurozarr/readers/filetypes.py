"""What a source file is, and how it becomes items -- the one place every reader asks.

BidsReader, SmartReader and ManifestReader each find files their own way; once one is
found, :func:`decode_file` turns it into items, so a format supported here is supported
by every reader. Order: a decoder registered for the extension (built-ins below, or a
user's :func:`~neurozarr.readers.register_format`) -> a folder format -> a table -> a
text document -> a signal (mne, neo, BIDS continuous TSV, WAV) -> the ``unclaimed``
policy (keep a reference, or embed the bytes).

Every library past mne is optional and imported only when its format turns up: a
missing one is a clear "install X" error for that file, never a failed import.
"""

from __future__ import annotations

import json
import mimetypes
import re
from pathlib import Path
from typing import Any, Iterator

import mne  # type: ignore[import-untyped]
import numpy as np
import pandas as pd

from ..entities import Entities
from ..errors import UnsupportedFormatError
from ..items import Array, Attrs, ExternalFile, Item, Recording, Table
from ..util import clean_nan
from .formats import UnclaimedPolicy, decode_embed, decoder_for, register_format

# Entry points mne.io.read_raw dispatches on by extension alone. Generic extensions it
# also lists (.eeg .bin .hdr .dat .txt .mat) need a look at siblings or bytes first;
# see _sniff. Curry's .cef/.dpa/.rs3 are sidecars of the .cdt/.dap entry point.
MNE_EXTS = {".edf", ".bdf", ".gdf", ".vhdr", ".ahdr", ".set", ".fif", ".fif.gz", ".cnt", ".con", ".sqd",
			".lay", ".cdt", ".dap", ".nedf", ".nxe", ".snirf", ".data", ".asc",
			".ns1", ".ns2", ".ns3", ".ns4", ".ns5", ".ns6"}
MNE_FOLDER_EXTS = {".ds", ".mff", ".mefd"}  # CTF, EGI, MEF3: recordings stored as a folder
_NEO_IO = {".rhd": "IntanIO", ".rhs": "IntanIO", ".smr": "Spike2IO", ".smrx": "CedIO", ".abf": "AxonIO",
		   ".plx": "PlexonIO"}
_BIDS_CONTINUOUS = {"physio", "stim", "motion"}  # headerless TSVs whose sidecar gives the rate
_TEXT_DOCS = {"readme", "changes", "license", "licence", "authors", "citation", "notes"}
_TEXT_EXTS = {"", ".md", ".txt", ".rst"}
# kept whole as embedded bytes: decoding pixels or frames would change the file, and
# nothing reads a window of a video the way it reads a window of a recording
# ponytail: embedding reads the whole file into memory; stream it in chunks for multi-GB video
MEDIA_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".svg", ".pdf",
			  ".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpg", ".mpeg", ".wmv",
			  ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
_EXT_PART = re.compile(r"(\.[A-Za-z0-9]+)+$")


def true_ext(path: Path) -> str:
	"""The extension, compound ones included (.nii.gz, .fif.gz, .tsv.gz, .ome.tif)."""
	name = path.name.lower()
	for ext in (".nii.gz", ".fif.gz", ".tsv.gz", ".ome.tif", ".ome.tiff", ".ome.zarr", ".cdt.dpa", ".cdt.cef"):
		if name.endswith(ext):
			return ext
	return path.suffix.lower()


def _siblings(path: Path, *exts: str) -> list[Path]:
	stem = path.name[:-len(path.suffix)] if path.suffix else path.name
	return [p for p in path.parent.glob(f"{glob_escape(stem)}.*") if p.suffix.lower() in exts and p != path]


def glob_escape(text: str) -> str:
	return re.sub(r"([\[\]*?])", r"[\1]", text)


def _head(path: Path, n: int = 256) -> bytes:
	try:
		with path.open("rb") as f:
			return f.read(n)
	except OSError:
		return b""


def _sniff(path: Path) -> str | None:
	"""Which mne reader a file with a generic extension needs, or None if it is a
	companion of another file (BrainVision's .eeg, Persyst's .dat) or not a recording."""
	ext = path.suffix.lower()
	if ext == ".eeg":  # Nihon Kohden, unless it is BrainVision's data file
		return None if _siblings(path, ".vhdr", ".ahdr", ".vmrk", ".amrk") else "nihon"
	if ext == ".bin":
		if _siblings(path, ".meta"):
			return None  # SpikeGLX: read as a folder, through neo
		if path.name.lower().endswith("_meg.bin"):
			return "fil"
		return "artemis123" if _siblings(path, ".txt") else None
	if ext == ".hdr":
		return "nirx" if _head(path, 16).startswith(b"[GeneralInfo]") else None
	if ext == ".dat":
		if _siblings(path, ".lay", ".dap"):
			return None  # Persyst / Curry 7 data, opened through its header
		return "bci2k" if _head(path, 16).startswith(b"HeaderLen=") else None
	return None


def folder_kind(path: Path) -> str | None:
	"""What a folder is when the folder itself is one recording or image: "mne" (CTF,
	EGI, MEF3), "bti", "nirx", "neo:<IO>" (Open Ephys, SpikeGLX, Neuralynx, TDT),
	"zarr" (OME-Zarr or any Zarr group), else None."""
	if not path.is_dir():
		return None
	ext = true_ext(path)
	if ext in MNE_FOLDER_EXTS:
		return "mne"
	if ext in (".zarr", ".ome.zarr") or (path / ".zgroup").exists() or (path / "zarr.json").exists():
		return "zarr"
	names = {p.name for p in path.iterdir()}
	suffixes = {p.suffix.lower() for p in path.iterdir() if p.is_file()}
	if "config" in names and any(n.startswith("c,rf") for n in names):
		return "bti"
	if any(p.suffix.lower() == ".hdr" and _sniff(p) == "nirx" for p in path.iterdir()):
		return "nirx"
	if "settings.xml" in names and any(n.startswith("experiment") or n.startswith("Record Node") for n in names) \
			or "structure.oebin" in names:
		return "neo:OpenEphysBinaryIO"
	if ".meta" in suffixes and ".bin" in suffixes or any(
			p.is_dir() and any(q.suffix == ".meta" for q in p.iterdir()) for p in path.iterdir() if p.name.endswith("_imec0")):
		return "neo:SpikeGLXIO"
	if ".ncs" in suffixes:
		return "neo:NeuralynxIO"
	if ".tev" in suffixes and ".tsq" in suffixes:
		return "neo:TdtIO"
	return None


def signal_kind(path: Path) -> str | None:
	"""How a file (not a folder) opens as a signal: "mne", "mne:<reader>", "neo:<IO>",
	"wav", or None."""
	ext = true_ext(path)
	if ext in MNE_EXTS:
		return "mne"
	if ext in _NEO_IO:
		return f"neo:{_NEO_IO[ext]}"
	if ext == ".wav":
		return "wav"
	sniffed = _sniff(path)
	return f"mne:{sniffed}" if sniffed else None


def companions(path: Path) -> set[Path]:
	"""Files an entry point reads besides itself, so a folder scan neither skips them as
	unknown nor opens them again: BrainVision .eeg/.vmrk, EEGLAB .fdt, Persyst .dat,
	Curry sidecars, Nihon Kohden's side files, Artemis's .txt."""
	ext = true_ext(path)
	exts = {".vhdr": (".eeg", ".vmrk"), ".ahdr": (".eeg", ".amrk"), ".set": (".fdt",), ".lay": (".dat",),
			".cdt": (".dpa", ".cef"), ".dap": (".dat", ".rs3", ".cef"), ".eeg": (".21e", ".pnt", ".log", ".11d", ".cmt", ".cn2"),
			".bin": (".txt",)}.get(ext, ())
	found = set(_siblings(path, *exts))
	if ext == ".cdt":
		found |= {p for p in path.parent.glob(f"{glob_escape(path.name)}.*")}
	return found


def _call(reader: Any, path: Path, **kwargs: Any) -> "mne.io.BaseRaw":
	"""An mne reader, lazily where it can be: some refuse preload=False (BCI2000), some
	take no preload or verbose at all (EyeLink, FieldTrip)."""
	try:
		return reader(path, preload=False, verbose=False, **kwargs)
	except NotImplementedError as exc:
		if "preload" not in str(exc):
			raise
		return reader(path, preload=True, verbose=False, **kwargs)
	except TypeError as exc:
		if "unexpected keyword" not in str(exc):
			raise
		return reader(path, **kwargs)


def open_raw(path: Path, kind: str | None = None) -> "mne.io.BaseRaw":
	"""One recording through mne, by the kind :func:`signal_kind`/:func:`folder_kind` gave."""
	kind = kind or (folder_kind(path) if path.is_dir() else signal_kind(path)) or "mne"
	if kind == "bti":
		pdf = next(p for p in sorted(path.iterdir()) if p.name.startswith("c,rf"))
		shape = path / "hs_file"
		return mne.io.read_raw_bti(pdf, config_fname=path / "config",
								   head_shape_fname=shape if shape.exists() else None, verbose=False)
	if kind == "nirx":
		return _call(mne.io.read_raw_nirx, path if path.is_dir() else path.parent)
	if kind == "mne" and true_ext(path) == ".asc":
		kind = "mne:eyelink"  # read_raw would pass it a preload it does not take
	if kind.startswith("mne:"):
		return _call(getattr(mne.io, f"read_raw_{kind[4:]}"), path)
	if true_ext(path) == ".cnt":
		# NeuroScan and ANT share .cnt. ANT's is a RIFF container -- check before calling
		# its reader, which segfaults on anything else. A NeuroScan file whose sample width
		# mne cannot work out reads as 16- or 32-bit with different lengths: refuse to guess.
		if _head(path, 4) in (b"RIFF", b"RF64"):
			return _call(mne.io.read_raw_ant, path)
		try:
			return _call(mne.io.read_raw_cnt, path)
		except RuntimeError as exc:
			raise UnsupportedFormatError(f"{path.name}: NeuroScan CNT with an ambiguous sample width "
										 f"(pass data_format to mne.io.read_raw_cnt): {exc}") from exc
	return _call(mne.io.read_raw, path)


def _alnum(text: str, fallback: str) -> str:
	return re.sub(r"[^A-Za-z0-9]", "", text) or fallback


def _unique(names: list[str]) -> list[str]:
	seen: dict[str, int] = {}
	out = []
	for name in names:
		seen[name] = seen.get(name, 0) + 1
		out.append(name if seen[name] == 1 else f"{name}-{seen[name]}")
	return out


def _neo_items(path: Path, io_name: str, entities: Entities, meta: dict[str, Any],
			   prefix: tuple[str, ...]) -> Iterator[Item]:
	"""Every analog stream of every segment as a Recording, and spike times as a table."""
	try:
		import neo  # type: ignore[import-untyped]
	except ImportError as exc:
		raise UnsupportedFormatError(f'reading {path.name!r} needs neo: pip install "neurozarr[formats]"') from exc
	io_cls = getattr(neo.io, io_name)
	io = io_cls(dirname=str(path)) if path.is_dir() else io_cls(filename=str(path))
	# ponytail: read_block loads every sample; a lazy mne BaseRaw over neo.rawio would stream big probes
	block = io.read_block(lazy=False)
	segments = block.segments
	for s, segment in enumerate(segments):
		streams = segment.analogsignals
		labels = _unique([_alnum(sig.name or "", f"stream{i}") for i, sig in enumerate(streams)])
		for sig, label in zip(streams, labels):
			extra = dict(entities.extra)
			if len(streams) > 1:
				extra["recording"] = label
			if len(segments) > 1:
				extra["split"] = s + 1
			names = sig.array_annotations.get("channel_names")
			names = _unique([str(n) for n in names]) if names is not None and len(names) == sig.shape[1] \
				else [f"ch{i:03d}" for i in range(sig.shape[1])]
			info = mne.create_info(names, float(sig.sampling_rate.rescale("Hz").magnitude), "misc")
			raw = mne.io.RawArray(np.asarray(sig.magnitude, dtype=np.float64).T, info, verbose=False)
			yield Recording(Entities(entities.sub, entities.ses, entities.datatype, extra), raw,
							{**meta, "_neurozarr_decoder": f"neo:{io_name}", "units": str(sig.units.dimensionality),
							 "t_start_s": float(sig.t_start.rescale("s").magnitude)}, prefix)
		spikes = [(train.name or f"unit{u}", float(t)) for u, train in enumerate(segment.spiketrains)
				  for t in train.rescale("s").magnitude]
		if spikes:
			extra = {**entities.extra, **({"split": s + 1} if len(segments) > 1 else {})}
			yield Table(Entities(entities.sub, entities.ses, entities.datatype, extra), "spikes",
						pd.DataFrame(spikes, columns=["unit", "time_s"]), {"_neurozarr_decoder": f"neo:{io_name}"}, prefix)


def _wav(path: Path) -> "mne.io.RawArray":
	from scipy.io import wavfile  # type: ignore[import-untyped]  # a dependency of mne
	rate, data = wavfile.read(path)
	data = data[:, None] if data.ndim == 1 else data
	info = mne.create_info([f"audio{i}" for i in range(data.shape[1])], float(rate), "misc")
	return mne.io.RawArray(data.T.astype(np.float64), info, verbose=False)


def _bids_continuous(path: Path, name: str, meta: dict[str, Any]) -> "mne.io.RawArray":
	"""A BIDS physio/stim/motion TSV: no header row; names in the sidecar's Columns or,
	for motion, the sibling channels.tsv; rate in SamplingFrequency."""
	table = pd.read_csv(path, sep="\t", header=None)
	columns = meta.get("Columns")
	# the spec says no header row, but some writers add one: a non-numeric first row is names
	if pd.to_numeric(table.iloc[0], errors="coerce").isna().all():
		columns, table = list(table.iloc[0]), table.iloc[1:]
	data = table.to_numpy(dtype=np.float64)
	if not columns:
		stem = path.name[:path.name.lower().rindex(f"_{name}")]
		channels = path.parent / f"{stem}_channels.tsv"
		columns = list(pd.read_csv(channels, sep="\t")["name"]) if channels.exists() else []
	if len(columns) != data.shape[1]:
		columns = [f"col{i:03d}" for i in range(data.shape[1])]
	info = mne.create_info(_unique([str(c) for c in columns]), float(meta["SamplingFrequency"]), "misc")
	return mne.io.RawArray(data.T, info, verbose=False)


def _text_doc(path: Path) -> bool:
	return path.is_file() and true_ext(path) in _TEXT_EXTS \
		and path.name[:len(path.name) - len(path.suffix)].lower() in _TEXT_DOCS


def text_attrs(path: Path, at: tuple[str, ...]) -> Attrs:
	"""A README/CHANGES/LICENSE as metadata at ``at``, named by the file."""
	name = _alnum(path.name.rsplit(".", 1)[0], "text") if path.suffix else _alnum(path.name, "text")
	return Attrs((*at, name), {"text": path.read_text(errors="replace"), "original_filename": path.name})


def decode_file(path: Path, entities: Entities, name: str, meta: dict[str, Any],
				unclaimed: UnclaimedPolicy = UnclaimedPolicy.REFERENCE,
				prefix: tuple[str, ...] = ()) -> Iterator[Item]:
	"""Turn one source file or folder into items. ``name`` is the item's name (a BIDS
	suffix, or a file stem); ``meta`` the sidecar metadata already gathered for it."""
	from dataclasses import replace

	if path.is_dir():
		kind = folder_kind(path)
		if kind == "zarr":
			yield from (replace(i, prefix=prefix) for i in decode_zarr(path, entities, name, meta))
		elif kind and kind.startswith("neo:"):
			yield from _neo_items(path, kind[4:], entities, meta, prefix)
		elif kind:
			yield Recording(entities, open_raw(path, kind), meta, prefix)
		else:
			# nothing reads this folder, and "embed" has no single byte stream to keep
			yield ExternalFile(entities, name, path, reader_hint="no reader for this folder", meta=meta, prefix=prefix)
		return
	ext = true_ext(path)
	if (decoder := decoder_for(path)) is not None:
		yield from (replace(i, prefix=prefix) for i in decoder(path, entities, name, meta))
	elif ext in (".tsv", ".csv"):
		if name in _BIDS_CONTINUOUS and "SamplingFrequency" in meta:
			yield Recording(entities, _bids_continuous(path, name, meta), meta, prefix)
		else:
			yield Table(entities, name, pd.read_csv(path, sep="\t" if ext == ".tsv" else ","), meta, prefix)
	elif _text_doc(path):
		yield text_attrs(path, (*prefix, *entities.path()))
	elif (kind := signal_kind(path)) is not None:
		if kind.startswith("neo:"):
			yield from _neo_items(path, kind[4:], entities, meta, prefix)
		else:
			yield Recording(entities, _wav(path) if kind == "wav" else open_raw(path, kind), meta, prefix)
	elif unclaimed is UnclaimedPolicy.EMBED:
		yield from (replace(i, prefix=prefix) for i in decode_embed(path, entities, name, meta))
	else:
		yield ExternalFile(entities, name, path, reader_hint=f"install or register a reader for {ext!r}",
						   meta=meta, prefix=prefix)


def decodable(path: Path) -> bool:
	"""Whether :func:`decode_file` reads data from this file or folder (rather than
	keeping a reference)."""
	if path.is_dir():
		return folder_kind(path) is not None
	return (decoder_for(path) is not None or true_ext(path) in (".tsv", ".csv")
			or _text_doc(path) or signal_kind(path) is not None)


# --- built-in decoders, registered like a user's would be ---------------------------------

def decode_tsv_gz(path: Path, entities: Entities, name: str, meta: dict[str, Any]) -> Iterator[Any]:
	if name in _BIDS_CONTINUOUS and "SamplingFrequency" in meta:
		yield Recording(entities, _bids_continuous(path, name, meta), meta)
	else:
		yield Table(entities, name, pd.read_csv(path, sep="\t"), meta)


def decode_media(path: Path, entities: Entities, name: str, meta: dict[str, Any]) -> Iterator[Any]:
	media_type = mimetypes.guess_type(path.name)[0]
	yield from decode_embed(path, entities, name, {**meta, **({"media_type": media_type} if media_type else {})})


def decode_tiff(path: Path, entities: Entities, name: str, meta: dict[str, Any]) -> Iterator[Any]:
	try:
		import tifffile
	except ImportError as exc:
		raise UnsupportedFormatError(f'reading {path.name!r} needs tifffile: pip install "neurozarr[formats]"') from exc
	with tifffile.TiffFile(path) as tif:
		series = tif.series
		for i, s in enumerate(series):
			dims = tuple(a.lower() for a in s.axes)
			extra: dict[str, Any] = {"_neurozarr_decoder": "tifffile", "original_filename": path.name}
			if tif.is_ome and i == 0:
				extra["ome_xml"] = tif.ome_metadata
			yield Array(entities, name if len(series) == 1 else f"{name}_series-{i}", s.asarray(), dims, meta={**meta, **extra})


_DICOM_TAGS = ("Modality", "Manufacturer", "ManufacturerModelName", "BodyPartExamined", "SeriesNumber",
			   "InstanceNumber", "Rows", "Columns", "PixelSpacing", "SliceThickness", "SpacingBetweenSlices",
			   "SliceLocation", "ImagePositionPatient", "ImageOrientationPatient", "RescaleSlope", "RescaleIntercept",
			   "BitsStored", "PhotometricInterpretation", "RepetitionTime", "EchoTime", "FlipAngle",
			   "MagneticFieldStrength", "NumberOfFrames")


def decode_dicom(path: Path, entities: Entities, name: str, meta: dict[str, Any]) -> Iterator[Any]:
	"""Pixels plus technical tags only: patient and study identifiers are never copied."""
	try:
		import pydicom
	except ImportError as exc:
		raise UnsupportedFormatError(f'reading {path.name!r} needs pydicom: pip install "neurozarr[formats]"') from exc
	ds = pydicom.dcmread(path)
	data = ds.pixel_array
	tags: dict[str, Any] = {}
	for tag in _DICOM_TAGS:
		value = ds.get(tag)
		if value is None:
			continue
		tags[tag] = [float(v) for v in value] if isinstance(value, pydicom.multival.MultiValue) else \
			(float(value) if isinstance(value, (pydicom.valuerep.DSfloat, float)) else
			 int(value) if isinstance(value, (pydicom.valuerep.IS, int)) else str(value))
	frames = int(ds.get("NumberOfFrames", 1) or 1)
	dims = (("frame",) if frames > 1 else ()) + ("y", "x") + (("channel",) if data.ndim == (3 if frames == 1 else 4) else ())
	yield Array(entities, name, data, dims[:data.ndim],
				meta={**meta, "_neurozarr_decoder": "pydicom", "dicom": tags, "original_filename": path.name})


def decode_zarr(path: Path, entities: Entities, name: str, meta: dict[str, Any]) -> Iterator[Any]:
	"""Every array of a Zarr group; an OME-Zarr's resolution levels keep its axis names
	and transforms."""
	import zarr

	group = zarr.open_group(str(path), mode="r")
	attrs: dict[str, Any] = dict(group.attrs.asdict())
	multiscales: Any = attrs.get("multiscales") or (attrs.get("ome") or {}).get("multiscales")
	if multiscales:
		ms = multiscales[0]
		axes = [a["name"] if isinstance(a, dict) else a for a in ms.get("axes", [])]
		for level, dataset in enumerate(ms["datasets"]):
			arr: Any = group[dataset["path"]]
			dims = tuple(str(a) for a in axes) if len(axes) == arr.ndim else ("t", "c", "z", "y", "x")[-arr.ndim:]
			yield Array(entities, f"{name}_level-{level}", arr[...], dims, meta={
				**meta, "_neurozarr_decoder": "ome-zarr", "ome_multiscales": clean_nan(json.loads(json.dumps(ms))),
				"level_path": dataset["path"],
				**({"omero": attrs["omero"]} if "omero" in attrs else {})})
		return
	for key, arr in group.arrays():
		named: Any = arr.attrs.get("_ARRAY_DIMENSIONS")
		dims = tuple(str(d) for d in named) if named else tuple(f"axis_{i}" for i in range(arr.ndim))
		yield Array(entities, f"{name}_{_alnum(key, 'array')}", arr[...], dims,
					meta={**meta, "_neurozarr_decoder": "zarr", "zarr_path": key})


def decode_gradients(path: Path, entities: Entities, name: str, meta: dict[str, Any]) -> Iterator[Any]:
	"""A DWI .bval (one b-value per volume) or .bvec (three rows of directions)."""
	values = np.loadtxt(path, ndmin=2 if path.suffix.lower() == ".bvec" else 1)
	dims = ("axis", "volume") if values.ndim == 2 else ("volume",)
	yield Array(entities, name, values, dims, meta={**meta, "_neurozarr_decoder": "text", "original_filename": path.name})


def _cells(df: pd.DataFrame) -> pd.DataFrame:
	"""NWB table columns holding arrays, lists or object references, as JSON text."""
	out = df.copy()
	for col in out.columns:
		if out[col].dtype == object and not all(isinstance(v, str) for v in out[col]):
			out[col] = [json.dumps(v.tolist() if hasattr(v, "tolist") else v, default=str) for v in out[col]]
	return out


def decode_nwb(path: Path, entities: Entities, name: str, meta: dict[str, Any]) -> Iterator[Any]:
	"""Every regularly sampled TimeSeries as a Recording (physical units: data x
	conversion + offset), every table (electrodes, units, trials, sweeps...) as a Table."""
	try:
		import pynwb  # type: ignore[import-untyped]
		from hdmf.common import DynamicTable  # type: ignore[import-untyped]
	except ImportError as exc:
		raise UnsupportedFormatError(f'reading {path.name!r} needs pynwb: pip install "neurozarr[formats]"') from exc
	used: set[str] = set()
	with pynwb.NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
		nwb = io.read()
		for obj in nwb.objects.values():
			label = _unique_label(_alnum(obj.name, "series"), used)
			if isinstance(obj, pynwb.TimeSeries) and obj.data is not None:
				rate = obj.rate
				if rate is None and obj.timestamps is not None and len(obj.timestamps) > 1:
					steps = np.diff(np.asarray(obj.timestamps[:]))
					rate = 1 / float(np.median(steps)) if np.allclose(steps, steps[0], rtol=1e-6) else None
				data = np.asarray(obj.data[:], dtype=np.float64)
				if rate is None or data.ndim > 2 or data.size == 0:
					continue  # ponytail: irregular or >2-D series (images) are not stored yet
				data = data[:, None] if data.ndim == 1 else data
				data = data * float(obj.conversion) + float(getattr(obj, "offset", 0.0) or 0.0)
				info = mne.create_info([f"ch{i:03d}" for i in range(data.shape[1])], float(rate), "misc")
				extra = {**entities.extra, "recording": label}
				yield Recording(Entities(entities.sub, entities.ses, entities.datatype, extra),
								mne.io.RawArray(data.T, info, verbose=False),
								clean_nan({**meta, "_neurozarr_decoder": "pynwb", "nwb_type": type(obj).__name__,
										   "nwb_path": obj.name, "units": str(obj.unit),
										   "starting_time_s": float(obj.starting_time or 0.0)}))
			elif isinstance(obj, DynamicTable) and len(obj) > 0:
				yield Table(entities, f"{name}_{label}", _cells(obj.to_dataframe().reset_index()),
							{**meta, "_neurozarr_decoder": "pynwb", "nwb_type": type(obj).__name__})


def _unique_label(label: str, used: set[str]) -> str:
	out, n = label, 1
	while out in used:
		n += 1
		out = f"{label}{n}"
	used.add(out)
	return out


def _register_builtins() -> None:
	register_format(".nwb", decode_nwb)
	register_format(".tsv.gz", decode_tsv_gz)
	register_format(".bval", decode_gradients)
	register_format(".bvec", decode_gradients)
	for ext in (".tif", ".tiff", ".ome.tif", ".ome.tiff"):
		register_format(ext, decode_tiff)
	register_format(".dcm", decode_dicom)
	for ext in MEDIA_EXTS:
		register_format(ext, decode_media)


_register_builtins()
