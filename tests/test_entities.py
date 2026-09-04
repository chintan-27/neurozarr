from neurozarr.entities import Entities, parse_entities, split_stem


def test_parses_real_bids_filename_fragments():
	# real filename fragments from BIDS/sub-001/ses-20220908/ieeg
	assert parse_entities("task-BaselineMontage_run-1") == {"task": "BaselineMontage", "run": "1"}
	assert parse_entities("task-PatientSurvey_acq-Test") == {"task": "PatientSurvey", "acq": "Test"}
	assert parse_entities("space-Other") == {"space": "Other"}


def test_split_stem_separates_suffix():
	assert split_stem("sub-001_ses-1_task-rest_run-1_channels") == (
		{"sub": "001", "ses": "1", "task": "rest", "run": "1"}, "channels")


def test_group_name_reproduces_bids_ordering():
	assert Entities("sub-001", "ses-20220908", "ieeg",
					parse_entities("task-BaselineMontage_run-1")).group_name() == "task-BaselineMontage_run-1"
	assert Entities("sub-001", "ses-20220908", "ieeg", {"space": "Other"}).group_name() == "space-Other"


def test_unrecognized_entities_are_kept_not_rejected():
	assert Entities("sub-001", "ses-1", "ieeg", {"hemi": "L"}).group_name() == "hemi-L"


def test_no_entities_gives_empty_group_name():
	# session-level tables like electrodes have no entities of their own
	assert Entities("sub-001", "ses-1", "ieeg").group_name() == ""


def test_session_is_optional():
	# session-less datasets are valid BIDS
	assert Entities("sub-001", None, "ieeg", {"task": "rest"}).path() == ("sub-001", "ieeg", "task-rest")
