# Pipeline Generation

## SNR step
First we generate the detection analysis:

```bash
python submit_so3.py --mode snr
```

Then postprocess on a cluster using:
```bash
python submit_postprocess.py snr --num-jobs 58
```

or individual folders with:
```bash
python ./postprocess_snr.py data/snr_0/
```

## Fisher step

`postprocess_snr.py` generates a file `so3_sources_Dec8.json` in each folder that can be used for the Fisher to defined at which redshifts the source is.

```bash
python submit_so3.py --mode pe
```

Then postprocess:
```bash
python submit_postprocess.py inference
```

## Useful commands
srun --partition normal -c 8 --pty bash -i -l
source fom_venv/bin/activate
jupyter lab --ip="*" --no-browser