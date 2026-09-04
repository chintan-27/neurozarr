# Command line

```bash
bidszarr validate ./my_bids_dataset          # check before converting
bidszarr convert ./my_bids_dataset ./out     # BIDS folder or manifest.csv
bidszarr convert ./BIDS ./out -j 6           # one worker per subject
bidszarr convert ./BIDS ./out --skip-existing   # only what's new
bidszarr info ./out                          # subjects, visits, counts
bidszarr history ./out                       # versions and tags
bidszarr verify ./BIDS ./out                 # confirm the store matches its source
bidszarr export ./out ./bids_again           # write the store back out as BIDS
```

`convert` takes `--dtype {int16,float16}`, `-m` for the commit message, `-q` to quiet it, and `--force` to convert despite validation warnings. `-v` turns on debug logging.

`export` writes BrainVision if `pybv` is installed, EDF if `edfio` is, and otherwise FIF (readable by mne, but not BIDS-conformant for ieeg/eeg).
