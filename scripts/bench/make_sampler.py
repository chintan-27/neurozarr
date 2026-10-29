"""Build the development sampler: one small folder (~125 MB) holding every file
type the paper touches -- EEG/iEEG/MEG/fNIRS/spike formats, NWB, MRI/PET,
DICOM, microscopy, video, audio, images, tables -- so arms and readers can be
exercised locally in minutes. It is a development fixture only: nothing in the
paper is measured on it, and no configuration is chosen with it.

	python scripts/bench/make_sampler.py [--out .bench_scratch/sampler] [--cache .bench_scratch/sampler_cache]

Layout:
	bids/<source>/      data that already is BIDS, as a small valid dataset each
	extras/<group>/<format>/   everything else, under its original file names
	                           (multi-file formats reference their own names)
	manifest.tsv        one row per item: group, format, path, source, licence,
	                    transform, bytes, files -- becomes the coverage table
	SHA256SUMS          every file

Whole files throughout, except where no small file exists: a BOLD run cut to
its first volumes, a WAV cut to its first second-ish, one resolution level of
an OME-Zarr. Those rows say so in `transform`. Downloads are cached by URL, so
a rebuild is offline. Zero-byte placeholder files (bids-examples has many)
are never copied: they would pass as "supported" while holding no data.
"""

import argparse
import gzip
import hashlib
import io
import itertools
import json
import re
import shutil
import struct
import time
import urllib.error
import urllib.parse
import urllib.request
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MNE = "https://raw.githubusercontent.com/mne-tools/mne-testing-data/master/"
BEX = "https://raw.githubusercontent.com/bids-standard/bids-examples/master/"
GIN = "https://gin.g-node.org/NeuralEnsemble/ephy_testing_data/"
OPENNEURO = "https://s3.amazonaws.com/openneuro.org/"
DANDI = "https://api.dandiarchive.org/api/assets/{}/download/"
IDR = "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/idr0062A/6001240.zarr/"

L_MNE = "see source (mne-testing-data; per-folder licence files where present)"
L_BEX = "see source (bids-examples)"
L_GIN = "CC-BY-SA-4.0 (ephy_testing_data LICENSE)"


def _get(url: str, cache: Path, byte_range: str | None = None) -> bytes:
	key = hashlib.sha256(f"{url}|{byte_range}".encode()).hexdigest()[:24]
	path = cache / key
	if not path.exists():
		req = urllib.request.Request(urllib.parse.quote(url, safe=":/?&=%"), headers={"User-Agent": "neurozarr-sampler"})
		if byte_range:
			req.add_header("Range", f"bytes={byte_range}")
		for attempt in range(5):
			try:
				with urllib.request.urlopen(req, timeout=120) as r:
					data = r.read()
				break
			except urllib.error.HTTPError as e:
				if e.code == 404:  # zarr leaves all-fill chunks unwritten: a 404 there is an empty chunk
					data = b""
					break
				if e.code < 500 or attempt == 4:
					raise
				time.sleep(2 ** attempt)  # GIN answers 500 under parallel load
		path.write_bytes(data)
	return path.read_bytes()


def _tree(repo: str, cache: Path) -> list[dict]:
	"""Every blob in a GitHub repo (one API call, cached)."""
	url = f"https://api.github.com/repos/{repo}/git/trees/master?recursive=1"
	return [x for x in json.loads(_get(url, cache))["tree"] if x["type"] == "blob"]


class Sampler:
	def __init__(self, out: Path, cache: Path):
		self.out, self.cache = out, cache
		self.rows: list[dict] = []
		self._trees: dict[str, list[dict]] = {}

	def _write(self, files: dict[str, bytes | Path]) -> tuple[int, int]:
		total = n = 0
		for rel, data in files.items():
			dest = self.out / rel
			dest.parent.mkdir(parents=True, exist_ok=True)
			if isinstance(data, Path):
				shutil.copyfile(data, dest)
			else:
				dest.write_bytes(data)
			total += dest.stat().st_size
			n += 1
		return total, n

	def _fetch(self, pairs: list[tuple[str, str]]) -> dict[str, bytes]:
		"""(dest rel path, url) -> bytes, downloaded in parallel."""
		with ThreadPoolExecutor(16) as pool:
			blobs = list(pool.map(lambda p: _get(p[1], self.cache), pairs))
		return {rel: b for (rel, _), b in zip(pairs, blobs) if b}

	def add(self, group: str, fmt: str, path: str, source: str, licence: str,
			files: dict[str, bytes | Path], transform: str = "") -> None:
		size, n = self._write(files)
		self.rows.append({"group": group, "format": fmt, "path": path, "source": source,
						  "licence": licence, "transform": transform, "bytes": size, "files": n})

	def github(self, repo: str, base: str, dest: str, items: list[str]) -> dict[str, bytes]:
		"""Files from a GitHub repo. An item ending in "/" is a whole folder; each item
		lands under `dest` by its own name (folder names kept, parents dropped)."""
		tree = self._trees.setdefault(repo, _tree(repo, self.cache))
		pairs = []
		for item in items:
			parent = item.rstrip("/").rpartition("/")[0]
			hits = [x for x in tree if (x["path"].startswith(item) if item.endswith("/") else x["path"] == item)]
			if not hits:
				raise SystemExit(f"not in {repo}: {item}")
			pairs += [(f"{dest}/{x['path'][len(parent):].lstrip('/')}", base + x["path"])
					  for x in hits if x["size"] > 0]
		return self._fetch(pairs)

	def mne(self, group: str, fmt: str, dest: str, items: list[str]) -> None:
		self.add(group, fmt, dest, MNE + items[0].rstrip("/"), L_MNE,
				 self.github("mne-tools/mne-testing-data", MNE, dest, items))

	def gin(self, group: str, fmt: str, dest: str, items: list[str]) -> None:
		"""ephy_testing_data files; an item ending in "/" is listed recursively."""
		pairs = []
		for item in items:
			parent = item.rstrip("/").rpartition("/")[0]
			for path in (self._gin_walk(item.rstrip("/")) if item.endswith("/") else [item]):
				pairs.append((f"{dest}/{path[len(parent):].lstrip('/')}", GIN + "raw/master/" + path))
		self.add(group, fmt, dest, GIN + "src/master/" + items[0].rstrip("/"), L_GIN, self._fetch(pairs))

	def _gin_walk(self, folder: str) -> list[str]:
		# ponytail: scrapes GIN's HTML listing (its API 404s for this repo); breaks if the page markup changes
		html = _get(GIN + "src/master/" + folder, self.cache).decode()
		out = []
		for row in re.findall(r"<tr>(.*?)</tr>", html, re.S):
			m = re.search(r'href="/NeuralEnsemble/ephy_testing_data/src/master/([^"]+)"', row)
			if m and m.group(1).startswith(folder + "/"):
				is_dir = "octicon-file-directory" in row or "fa-folder" in row
				out += self._gin_walk(m.group(1)) if is_dir else [m.group(1)]
		return out

	def bids_local(self, name: str, root: Path, keep: list[str], note: str) -> None:
		"""A local BIDS dataset, cut to its top-level files plus the `keep` subtrees."""
		files: dict[str, bytes | Path] = {}
		for f in root.iterdir():
			if f.is_file():
				files[f"bids/{name}/{f.name}"] = f
		for sub in keep:
			for f in (root / sub).rglob("*"):
				if f.is_file() and f.stat().st_size > 0:
					files[f"bids/{name}/{f.relative_to(root)}"] = f
		self.add("BIDS", note, f"bids/{name}", str(root.relative_to(ROOT)), "see source", files)

	def bids_example(self, name: str, keep: list[str], fmt: str, transform: str = "") -> None:
		"""A bids-examples dataset: its top-level files plus the `keep` subtrees, minus empty placeholders."""
		tree = self._trees.setdefault("bids-standard/bids-examples", _tree("bids-standard/bids-examples", self.cache))
		pairs = [(f"bids/{name}/{x['path'][len(name) + 1:]}", BEX + x["path"]) for x in tree
				 if x["path"].startswith(name + "/") and x["size"] > 0
				 and ("/" not in x["path"][len(name) + 1:] or any(x["path"].startswith(f"{name}/{k}") for k in keep))]
		# bids-examples also commits HTML pages under data-file names (pet001's T1w.nii)
		files = {k: v for k, v in self._fetch(pairs).items() if not v[:100].lstrip().lower().startswith(b"<!doctype html")}
		self.add("BIDS", fmt, f"bids/{name}", BEX + name, L_BEX, files, transform)

	def write_manifest(self) -> None:
		cols = ["group", "format", "path", "source", "licence", "transform", "bytes", "files"]
		lines = ["\t".join(cols)] + ["\t".join(str(r[c]) for c in cols) for r in self.rows]
		(self.out / "manifest.tsv").write_text("\n".join(lines) + "\n")
		sums = [f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.relative_to(self.out)}"
				for f in sorted(self.out.rglob("*")) if f.is_file() and f.name not in ("manifest.tsv", "SHA256SUMS")]
		(self.out / "SHA256SUMS").write_text("\n".join(sums) + "\n")


def nifti_head(gz: bytes, volumes: int) -> bytes:
	"""The first `volumes` volumes of a 4D NIfTI-1 (.nii.gz), header patched to match."""
	raw = gzip.decompress(gz)
	dims = list(struct.unpack("<8h", raw[40:56]))
	bitpix = struct.unpack("<h", raw[72:74])[0]
	offset = int(struct.unpack("<f", raw[108:112])[0])
	per_volume = dims[1] * dims[2] * dims[3] * bitpix // 8
	dims[4] = min(volumes, dims[4])
	head = raw[:40] + struct.pack("<8h", *dims) + raw[56:offset]
	return gzip.compress(head + raw[offset:offset + dims[4] * per_volume])


def wav_head(partial: bytes) -> bytes:
	"""A valid WAV from the leading bytes of a longer one (header sizes rewritten)."""
	src = wave.open(io.BytesIO(partial))
	frames = src.readframes(src.getnframes())  # returns only what the cut kept
	frames = frames[:len(frames) - len(frames) % (src.getsampwidth() * src.getnchannels())]  # whole frames only
	buf = io.BytesIO()
	with wave.open(buf, "wb") as dst:
		dst.setparams(src.getparams())
		dst.writeframes(frames)
	return buf.getvalue()


def build(out: Path, cache: Path) -> Sampler:
	if out.exists():
		shutil.rmtree(out)
	out.mkdir(parents=True)
	cache.mkdir(parents=True, exist_ok=True)
	s = Sampler(out, cache)

	# --- BIDS datasets -------------------------------------------------------
	s.bids_local("bravo", ROOT / "BIDS", ["sub-004"], "iEEG EDF + beh TSV (Percept DBS, BRAVO extras)")
	s.add("BIDS", "MEF3 (iEEG)", "bids/mef3", MNE + "MEF", L_MNE, {
		**s.github("mne-tools/mne-testing-data", MNE, "bids/mef3/sub-ieegModulator/ses-ieeg01/ieeg",
				   ["MEF/sub-ieegModulator_ses-ieeg01_task-photicstim_run-01_ieeg.mefd/"]),
		"bids/mef3/dataset_description.json": b'{"Name": "MEF3 sample", "BIDSVersion": "1.9.0"}\n'})
	s.add("BIDS", "OPM-MEG (FIL .bin)", "bids/opm_fil", MNE + "FIL", L_MNE, {
		**s.github("mne-tools/mne-testing-data", MNE, "bids/opm_fil/sub-noise/ses-001/meg", [
			f"FIL/sub-noise_ses-001_task-noise220622_run-001_{x}"
			for x in ("meg.bin", "meg.json", "channels.tsv", "positions.tsv")]),
		"bids/opm_fil/dataset_description.json": b'{"Name": "OPM sample", "BIDSVersion": "1.9.0"}\n'})
	s.add("BIDS", "motion (.tsv.gz)", "bids/motion", MNE + "MOTION", L_MNE, {
		**s.github("mne-tools/mne-testing-data", MNE, "bids/motion", ["MOTION/sub-S1/"]),
		"bids/motion/dataset_description.json": b'{"Name": "motion sample", "BIDSVersion": "1.9.0"}\n'})
	s.bids_example("ds210", ["sub-01/"], "physio .tsv.gz (fMRI)")
	s.bids_example("pet001", ["sub-01/"], "PET NIfTI")
	s.bids_example("micr_SPIM", ["sub-01/"], "microscopy OME-TIFF", "upstream images are 1x1-pixel placeholders: BIDS layout only")
	s.bids_example("ieeg_epilepsyNWB", ["sub-01/"], "iEEG NWB")
	s.bids_example("eeg_matchingpennies", ["stimuli/"], "stimuli PNG")

	# --- EEG / iEEG formats ----------------------------------------------------
	s.mne("EEG", "BDF", "extras/eeg/bdf", ["BDF/test_bdf_stim_channel.bdf"])
	s.mne("EEG", "BrainVision", "extras/eeg/brainvision", [f"Brainvision/test_NO.{x}" for x in ("vhdr", "vmrk", "eeg")])
	s.mne("EEG", "EEGLAB .set+.fdt", "extras/eeg/eeglab", ["EEGLAB/test_raw.set", "EEGLAB/test_raw.fdt"])
	s.mne("EEG", "GDF", "extras/eeg/gdf", ["GDF/test_gdf_2.20.gdf"])
	s.mne("EEG", "NeuroScan CNT", "extras/eeg/cnt", ["CNT/scan41_short.cnt"])
	s.mne("EEG", "EGI MFF", "extras/eeg/mff", ["EGI/test_egi_pns.mff/"])
	s.mne("EEG", "Nihon Kohden", "extras/eeg/nihonkohden", [
		f"NihonKohden/MB0400FU.{x}" for x in ("EEG", "11D", "21E", "CMT", "CN2", "PNT", "LOG")])
	s.mne("EEG", "Persyst", "extras/eeg/persyst", [
		f"Persyst/sub-pt1_ses-02_task-monitor_acq-ecog_run-01_clip2.{x}" for x in ("lay", "dat")])
	s.mne("EEG", "Curry 8", "extras/eeg/curry", [
		f"curry/test_bdf_stim_channel Curry 8.cdt{x}" for x in ("", ".dpa", ".cef")])
	s.mne("EEG", "ANT Neuro", "extras/eeg/antneuro", ["antio/andy_101/"])
	s.mne("EEG", "BCI2000", "extras/eeg/bci2000", ["BCI2k/bci2k_test.dat"])

	# --- MEG -------------------------------------------------------------------
	s.mne("MEG", "FIF", "extras/meg/fif", ["SSS/141027_cropped_90Hz_raw.fif"])
	s.mne("MEG", "CTF .ds", "extras/meg/ctf", ["CTF/catch-alp-good-f.ds/"])
	s.mne("MEG", "KIT .con", "extras/meg/kit", ["KIT/010409_Motor_task_coregist-export_tiny_1s.con"])
	s.mne("MEG", "BTi/4D", "extras/meg/bti", ["BTi/4Dsim/"])
	s.mne("MEG", "Artemis123", "extras/meg/artemis123", [
		f"ARTEMIS123/Artemis_Data_2017-04-14-10h-38m-59s_Phantom_1k_HPI_1s.{x}" for x in ("bin", "txt")])

	# --- fNIRS, eye tracking ---------------------------------------------------
	s.mne("fNIRS", "SNIRF", "extras/fnirs/snirf", ["SNIRF/MNE-NIRS/20220217/20220217_nirx_15_3_recording.snirf"])
	s.mne("fNIRS", "NIRx", "extras/fnirs/nirx", ["NIRx/nirsport_v1/"])
	s.mne("eye tracking", "EyeLink .asc", "extras/eyetrack/eyelink", ["eyetrack/test_eyelink.asc"])

	# --- extracellular / intracellular ephys -----------------------------------
	s.gin("ephys", "Intan .rhd", "extras/ephys/intan", ["intan/intan_rhd_test_1.rhd"])
	s.gin("ephys", "SpikeGLX", "extras/ephys/spikeglx", [
		"spikeglx/Noise4Sam_g0/Noise4Sam_g0_imec0/Noise4Sam_g0_t0.imec0.ap.bin",
		"spikeglx/Noise4Sam_g0/Noise4Sam_g0_imec0/Noise4Sam_g0_t0.imec0.ap.meta"])
	s.gin("ephys", "Open Ephys binary", "extras/ephys/openephys", ["openephysbinary/v0.4.4.1_with_spikes/"])
	s.gin("ephys", "Spike2 .smr", "extras/ephys/spike2", ["spike2/File_spike2_1.smr"])
	s.gin("ephys", "Axon .abf", "extras/ephys/axon", ["axon/File_axon_1.abf"])
	s.gin("ephys", "Plexon .plx", "extras/ephys/plexon", ["plexon/File_plexon_1.plx"])
	s.mne("ephys", "Neuralynx .ncs/.nev", "extras/ephys/neuralynx", ["neuralynx/LAHC1.ncs", "neuralynx/Events.nev"])
	s.mne("ephys", "Blackrock .ns3", "extras/ephys/blackrock", ["nsx/test_NEURALCD_raw.ns3"])

	# --- containers --------------------------------------------------------------
	s.add("container", "NWB (ecephys)", "extras/nwb/ecephys", "DANDI:000006", "CC-BY-4.0", s._fetch([(
		"extras/nwb/ecephys/sub-anm372907_ses-20170613.nwb", DANDI.format("f1b6ed8f-1e14-44fa-b4ee-240d0efe3759"))]))
	s.add("container", "NWB (icephys)", "extras/nwb/icephys", "DANDI:000008", "CC-BY-4.0", s._fetch([(
		"extras/nwb/icephys/sub-mouse-POHNW_icephys.nwb", DANDI.format("23482c59-e2ea-4206-8ea9-1a700d4dadc5"))]))
	s.mne("container", "MATLAB v7 / v7.3 (FieldTrip)", "extras/mat", [
		"fieldtrip/one_channel_elec_bug_data_v7.mat", "fieldtrip/one_channel_elec_bug_data_v73.mat"])

	# --- MRI, DICOM ----------------------------------------------------------------
	t1 = OPENNEURO + "ds000001/sub-01/anat/sub-01_T1w.nii.gz"
	s.add("MRI", "NIfTI T1w", "extras/mri/anat", t1, "CC0 (OpenNeuro)", s._fetch([("extras/mri/anat/sub-01_T1w.nii.gz", t1)]))
	bold = OPENNEURO + "ds000001/sub-01/func/sub-01_task-balloonanalogrisktask_run-01_bold.nii.gz"
	s.add("MRI", "NIfTI BOLD (4D)", "extras/mri/func", bold, "CC0 (OpenNeuro)",
		  {"extras/mri/func/sub-01_task-balloonanalogrisktask_run-01_bold.nii.gz": nifti_head(_get(bold, cache), 20)},
		  transform="first 20 volumes")
	dipy = "https://raw.githubusercontent.com/dipy/dipy/master/dipy/data/files/"
	s.add("MRI", "NIfTI DWI + bval/bvec", "extras/mri/dwi", dipy + "small_101D.nii.gz", "see source (dipy)",
		  s._fetch([(f"extras/mri/dwi/small_101D.{x}", dipy + f"small_101D.{x}") for x in ("nii.gz", "bval", "bvec")]))
	dcm = "https://raw.githubusercontent.com/pydicom/pydicom/main/src/pydicom/data/test_files/"
	s.add("MRI", "DICOM", "extras/dicom", dcm, "see source (pydicom)",
		  s._fetch([(f"extras/dicom/{x}", dcm + x) for x in ("CT_small.dcm", "MR_small.dcm")]))

	# --- microscopy ----------------------------------------------------------------
	ome = "https://downloads.openmicroscopy.org/images/OME-TIFF/2016-06/bioformats-artificial/multi-channel-z-series.ome.tif"
	s.add("microscopy", "OME-TIFF", "extras/microscopy/ome-tiff", ome, "see source (OME)",
		  s._fetch([("extras/microscopy/ome-tiff/multi-channel-z-series.ome.tif", ome)]))
	s.add("microscopy", "OME-Zarr (v0.4)", "extras/microscopy/6001240.zarr", IDR, "see source (IDR idr0062A)",
		  _omezarr_level(cache, 2), transform="resolution level 2 only (metadata rewritten to match)")

	# --- video, audio, images ------------------------------------------------------
	dlc = "https://raw.githubusercontent.com/DeepLabCut/DeepLabCut/main/examples/openfield-Pranav-2018-10-30/videos/m3v1mp4.mp4"
	s.add("video", "MP4 (behaviour)", "extras/video", dlc, "LGPL-3.0 (DeepLabCut)",
		  s._fetch([("extras/video/m3v1mp4.mp4", dlc)]))
	wav = OPENNEURO + "ds002345/stimuli/pieman_audio.wav"
	s.add("audio", "WAV (stimulus)", "extras/audio", wav, "CC0 (OpenNeuro)",
		  {"extras/audio/pieman_audio_head.wav": wav_head(_get(wav, cache, "0-999999"))}, transform="first 1 MB")
	bmp = OPENNEURO + "ds000117/stimuli/meg/Circle.bmp"
	s.add("image", "BMP (stimulus)", "extras/images/stimuli", bmp, "CC0 (OpenNeuro)",
		  s._fetch([("extras/images/stimuli/Circle.bmp", bmp)]))
	s.add("image", "JPEG (electrode photo)", "extras/images/photo", BEX + "ieeg_filtered_speech", L_BEX,
		  s.github("bids-standard/bids-examples", BEX, "extras/images/photo", ["ieeg_filtered_speech/sub-cm4/ieeg/sub-cm4_photo.jpg"]))

	s.write_manifest()
	return s


def _omezarr_level(cache: Path, level: int) -> dict[str, bytes | Path]:
	"""One resolution level of the IDR OME-Zarr, re-declared as its only dataset."""
	dest = "extras/microscopy/6001240.zarr"
	attrs = json.loads(_get(IDR + ".zattrs", cache))
	ms = attrs["multiscales"][0]
	ms["datasets"] = [d for d in ms["datasets"] if d["path"] == str(level)]
	zarray = json.loads(_get(IDR + f"{level}/.zarray", cache))
	sep = zarray.get("dimension_separator", ".")
	grid = [range(-(-n // c)) for n, c in zip(zarray["shape"], zarray["chunks"])]
	keys = [sep.join(map(str, idx)) for idx in itertools.product(*grid)]
	with ThreadPoolExecutor(16) as pool:
		chunks = list(pool.map(lambda k: _get(IDR + f"{level}/{k}", cache), keys))
	files: dict[str, bytes | Path] = {
		f"{dest}/.zgroup": _get(IDR + ".zgroup", cache),
		f"{dest}/.zattrs": json.dumps(attrs, indent=1).encode(),
		f"{dest}/{level}/.zarray": json.dumps(zarray, indent=1).encode()}
	files.update({f"{dest}/{level}/{k}": c for k, c in zip(keys, chunks)})
	return files


if __name__ == "__main__":
	ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	ap.add_argument("--out", type=Path, default=ROOT / ".bench_scratch/sampler")
	ap.add_argument("--cache", type=Path, default=ROOT / ".bench_scratch/sampler_cache")
	args = ap.parse_args()
	s = build(args.out, args.cache)
	for r in s.rows:
		print(f"{r['bytes'] / 1e6:8.2f} MB  {r['files']:4d} files  {r['group']:<12} {r['format']}")
	print(f"{sum(r['bytes'] for r in s.rows) / 1e6:8.2f} MB total, {len(s.rows)} items -> {args.out}")
