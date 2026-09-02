from bidszarr.entities import Entities, parse_entities

# real filename fragments from BIDS/sub-001/ses-20220908/ieeg
assert parse_entities("task-BaselineMontage_run-1") == {"task": "BaselineMontage", "run": "1"}
assert parse_entities("task-PatientSurvey_acq-Test") == {"task": "PatientSurvey", "acq": "Test"}
assert parse_entities("space-Other") == {"space": "Other"}

assert Entities("sub-001", "ses-20220908", "ieeg", parse_entities("task-BaselineMontage_run-1")).group_name() \
	== "task-BaselineMontage_run-1"
assert Entities("sub-001", "ses-20220908", "beh", parse_entities("task-PatientSurvey_acq-Test")).group_name() \
	== "task-PatientSurvey_acq-Test"
assert Entities("sub-001", "ses-20220908", "ieeg", {"space": "Other"}).group_name() == "space-Other"

# unrecognized entity keys are kept, not rejected
assert Entities("sub-001", "ses-1", "ieeg", {"hemi": "L"}).group_name() == "hemi-L"

# no entities at all -> empty group name (session-level tables like electrodes)
assert Entities("sub-001", "ses-1", "ieeg").group_name() == ""

print("test_entities: all assertions passed")
