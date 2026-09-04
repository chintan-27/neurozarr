"""Read data back out of a store: navigate it, pull an mne.Raw, search across
subjects, and look at the version history.

    python examples/read_back.py ./study.zarr
"""

import sys

from neurozarr import Repo


def main(store="./study.zarr"):
	repo = Repo(store)
	print(f"{store}: {len(repo.subjects())} subject(s)\n")

	for sub_id in repo.subjects():
		subject = repo.subject(sub_id)
		print(f"{sub_id}  {subject.attrs}")
		print(f"  {len(subject.visits())} visits, {len(subject.recordings())} recordings")

	sub_id = repo.subjects()[0]
	subject = repo.subject(sub_id)

	# the longest recording this subject has
	longest = max(subject.recordings(), key=lambda r: r.shape[-1])
	print(f"\nlongest recording: {longest.path} {longest.shape}")

	values, meta = longest.data()          # physical units, calibration applied
	print(f"  data {values.shape} {values.dtype}, sfreq={meta.get('SamplingFrequency')}")

	raw = longest.raw()                     # a real mne object
	print(f"  as mne.Raw: {raw}")
	print(f"  channels: {longest.ch_names()}")

	# search the whole store by entity
	matches = list(repo.find(task="BrainSenseStream"))
	print(f"\nrecordings with task=BrainSenseStream: {len(matches)}")

	print("\nhistory:")
	for snapshot_id, message, when in repo.history(sub_id):
		print(f"  {str(snapshot_id)[:12]}  {when:%Y-%m-%d %H:%M}  {message}")
	print("tags:", repo.tags(sub_id) or "(none)")


if __name__ == "__main__":
	main(*sys.argv[1:])
