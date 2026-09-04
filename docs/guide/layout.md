# How the store is laid out

**One Icechunk repository per subject**, so a single subject can be shared, copied, or versioned on its own without shipping the whole study:

```text
study.zarr/
  _dataset/            # dataset-wide metadata (dataset_description, participants info)
  sub-001/             # an independent Icechunk repo
    ses-20220908/
      ieeg/
        task-BrainSenseStream_acq-TD_run-1/
          data         # (channels x samples) int16 + scale/offset
          channels     # the channels table
          events       # if present
      beh/
        task-TherapyHistory/
          table
  sub-002/
  ...
```

Inside a subject's repo the tree starts at its sessions — no redundant `sub-001/` level, because the repository already *is* that subject.
