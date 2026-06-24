# Importing measured data

OptiVibe can replay a **measured** acceleration record through the same pipeline
as a synthetic one. Replay is selected with an excitation `kind` that names a
file format; each format has a loader registered in the data-loader registry
(seam SW-08), so adding a format never touches the core.

Two formats are built in (no extra dependency): **CSV** and **WAV**. Four
instrument-capture formats need the optional `io-formats` extra:

```bash
pip install 'optivibe[io-formats]'        # nptdms + pyuff + h5py
# or, from a checkout:
uv sync --extra io-formats
```

If the extra is missing, the loader fails with a clear hint naming the package
and the install command — the rest of OptiVibe keeps working.

## Supported formats

| `kind` | Format | Backend | Sampling rate from | Unit label |
|--------|--------|---------|--------------------|------------|
| `csv`  | CSV table | built-in | time column or `fs_hz` | `units` field |
| `wav`  | WAV PCM | built-in (scipy) | file header | full-scale mapping |
| `tdms` | NI TDMS | `nptdms` | `wf_increment` or `fs_hz` | channel `unit_string` |
| `uff`  | UFF/UNV dataset-58 | `pyuff` | `abscissa_inc` or `fs_hz` | ordinate unit label |
| `mat`  | MATLAB v4/v5/v7 | scipy (core) | `fs_hz` or `fs_key` | none (set `units`) |
| `hdf5` | HDF5 `.h5`/`.hdf5` | `h5py` | `fs_hz` or `fs_attr` | dataset attr (`units_attr`) |

A MATLAB **v7.3** file is HDF5 underneath; read it with the `hdf5` loader (or
re-save as v7). The loader detects v7.3 and says so.

## Units → SI at the boundary

Every record is converted to acceleration in **m/s²** at the loader, never
deeper in the pipeline. The `units` field on an instrument spec controls this:

- `m/s^2` — stored values are already SI (pass-through).
- `g` — multiplied by the standard gravity constant.
- `V` — a raw transducer voltage; supply `sensitivity` (and
  `sensitivity_unit`, default `mV/g`) to convert volts to acceleration.
- `auto` — read the unit label embedded in the file (TDMS/UFF/HDF5 only).

The conversion is deliberately **strict**. If you state `units` explicitly *and*
the file carries a recognized but different label (e.g. you say `m/s^2` but the
channel is labelled `g`), the loader refuses to guess and raises an error rather
than silently mis-scaling the record. MAT files have no unit convention, so
`units` must be explicit there (`auto` is rejected).

## Selecting a channel

A 2-D record (several channels in one array) selects one channel with `column`
(MAT/HDF5) or `channel` (TDMS, by 0-based index or name). For a 2-D array the
longer axis is taken as time, so a transposed `channels × samples` layout is
handled automatically. The chosen channel is placed on the spec's `axis`
(`x`/`y`/`z`); the other two axes are zero.

## Examples

Runnable scenarios live in `examples/`:

- `examples/replay_csv.yaml` — CSV, rate inferred from a time column, data in g.
- `examples/replay_wav.yaml` — int16 WAV normalised and mapped to a full-scale g.
- `examples/replay_tdms.yaml` — the same record as the CSV example, stored as an
  NI TDMS channel; `units: auto` reads the `g` label from the file.

```bash
optivibe run examples/replay_tdms.yaml
```

A minimal HDF5 scenario:

```yaml
name: replay-hdf5-demo
variant: B
excitation:
  kind: hdf5
  axis: x
  path: data/run012.h5
  dataset: /accel/x        # path of the dataset inside the file
  fs_attr: fs_hz           # dataset attribute holding the sampling rate
  units_attr: units        # dataset attribute holding the unit string
  units: auto
stages:
  excitation: hdf5
seed: 12345
```

The full set of fields for each format is documented on the
[API reference](reference.md) page (`optivibe.core.config.models`).
