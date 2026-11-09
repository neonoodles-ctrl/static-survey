# Static Survey

Least squares adjustment and cause-finding for survey control networks: total station directions, zenith angles and slope distances, plus static GNSS baselines. It reads a CompNet 3.x project, or the raw instrument files directly (Leica GSI-16 and Leica Infinity `.asc` baselines). It re-adjusts the job with a rigorous 3D model, then tests specific causes for the readings that don't fit.

It never changes your data. Every finding is a proposal with its numbers, for the surveyor to decide on.

## Windows program

1. Download `StaticSurvey.exe` from the [Releases](../../releases) page. It needs no install and no internet.
2. Run it and choose the job folder.
   - With a CompNet project in the folder (`.TRV` and its companion files), it adjusts that project. Any `.GSI` files in the folder supply the instrument timestamps.
   - Otherwise it uses the raw files: `.GSI`, `.asc`, and a `.FIX` or `.STD` file for the control marks.
3. Press **Run**. Four files are written to the job folder, named after the job, and the report opens in your browser:
   - `JOB-report.html` (and `.md`): what was adjusted, the findings, the largest residuals and the adjusted coordinates.
   - `JOB-coordinates.csv`: name, E, N, H, their standard deviations (m) and fixed / weighted / adjusted. The program reads the same layout as control, so one job's marks can be the next job's control.
   - `JOB-network.dxf`: the network for AutoCAD (2000 format, metres, MGA2020 E/N with the AHD height as Z). Layers `SS-CONTROL`, `SS-MARKS`, `SS-NAMES`, `SS-TS`, `SS-GNSS`, `SS-ELLIPSES` (95% error ellipses, enlarged; the enlargement is written on the drawing) and `SS-FLAGGED` (red: readings set aside by the w-test, or named by a finding, labelled with the finding).

   The drawing is not called `JOB.dxf` because CompNet writes its own drawing under that name.

Windows may warn that the program is from an unknown publisher, because it isn't code-signed. Choose **More info**, then **Run anyway**.

### Options

The options box takes the same options as the command line, for example:

- `--rename STN_1=STN1`: give a mark its project name when the field file booked it under another name.
- `--exclude "JOB.GSI line 83"`: leave out one reading, as you did in your own adjustment.
- `--file-settings JOB2.GSI:pointing-h=4,pointing-v=4,centring=0.002,height=0.005`: instrument settings for one GSI file.
- `--pointing-h 2 --pointing-v 4 --centring 0.0007 --edm-const 0.002 --edm-ppm 2`: instrument settings for the whole job.

## What it checks

- **Global test, both sides.** The ICSM SP1 v2.2 check of the variance factor against the upper and lower limits.
- **Setup twist or drift.** A setup's circle orientation stepped or drifted part-way through, matched against the instrument timestamps.
- **Refraction or a height error.** It fits refraction on a reciprocal leg, a wrong target height and a wrong instrument height, and names one only if it beats every rival with both in the model. Otherwise it says it can't tell, and what would.
- **Prism constant.** Distances from one setup, or to one mark, all off by the same amount.
- **Wrong or swapped point numbers.** A reading that fits another mark seen from the same setup.
- **Bad pointings.** A reading that disagrees with the other repeats of its own set, where only the pointing can differ.
- **Repeats and unchecked marks.** What the statistics look like with each set counted once, and marks that no independent observation checks.

## Command line

```
pip install .
static-survey compnet path/to/JOB --gsi path/to/JOB.GSI --out report.md --csv coordinates.csv --dxf network.dxf
static-survey raw --gsi JOB.GSI --asc DAY1.asc --control JOB.FIX --geoid-grid au_ga_AUSGeoid2020_20180201.tif --out report.md
```

`StaticSurvey.exe` accepts the same `compnet` and `raw` commands, and its `raw` command uses the geoid grid it ships with.

## Tests

```
pip install . pytest
python -m pytest -q
```

The tests use a made-up 6-mark network with planted mistakes. Each check must find its own mistake and stay quiet on clean data.

## Limits

- Deflections of the vertical are not modelled. Trig heights are treated as AHD differences, as CompNet does.
- Only Leica GSI-16, Leica Infinity `.asc` and CompNet 3.x projects are read.
- Coordinates are MGA2020 zone 56 only. Jobs in other zones are not handled yet.

## Geoid

The Windows program includes the AUSGeoid2020 grid (`au_ga_AUSGeoid2020_20180201.tif`) from the PROJ data CDN. It is © Commonwealth of Australia (Geoscience Australia), licensed under CC BY 4.0.
