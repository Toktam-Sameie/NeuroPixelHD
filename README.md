# First paper code

Hyperdimensional-computing (HDC) analysis of Allen Brain Observatory (Neuropixels/ecephys) recordings: encoding spikes into hypervectors and decoding stimulus position/identity from brain-region activity.

## Layout

- `common.py` — functions shared verbatim across the scripts below (HDC encoding, classification, data-saving helpers), extracted to one place instead of being duplicated in every script.
- `alternative.py`, `time_hv.py`, `fano_factor.py`, `peak_to_trough.py`, `power_spectrum.py` — single-unit analyses (temporal hypervector encoding, Fano factor, spike peak-to-trough/cell typing, power spectrum) feeding the same decoding pipeline.
- `ml_models.py`, `ml_models_pop_level.py`, `ml_models_for_fareed.py` — classical ML baselines (RF/SVM/kNN/MLP/etc.) for Gabor-stimulus decoding, at unit and population level.
- `ml_models_natural_scenes.py`, `ml_models_natural_scenes_for_fareed.py`, `ml_models_natural_scenes_spatial_averaging_pop_level.py` — same ML baselines for natural-scene stimuli, including a spatially-averaged population-level variant.

## Setup

```
pip install -r requirements.txt
```

Requires AllenSDK and a local `ecephys_cache_dir/` with downloaded Allen Brain Observatory sessions (see [AllenSDK docs](https://allensdk.readthedocs.io/)).

## Notes

Each script still has its own copy of a handful of functions that read session-specific state (`session`, `encoder`, `encode_pos`/`decode_pos`, etc.) set elsewhere in the same script. These were kept local rather than merged into `common.py` because merging them would require rewiring that shared state across files — not something that can be verified without AllenSDK and a real dataset on hand.
