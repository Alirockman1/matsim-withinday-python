# Creating the synthetic Sioux Falls population (10 % sample) – step by step

## 0. What you get
`generate_synthetic_siouxfalls.py` (+ `real_data.py`, `validation_report.py`) turns your facility file and network into
a new MATSim population (`population_synthetic.xml`) and facilities (`facilities_synthetic.xml.gz`), plus
`generation_meta.json`, `facility_links.csv`, `matsim_qsim_snippet.xml`, `validation_report.png`.
Ready-made 10 % output is in `output_10pct/` (built WITHOUT the LeBlanc table or LODES, see step 4/5).

## 1. Install
```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install "numpy>=1.26" scipy matplotlib
mkdir inputs   # copy facilities_xml.gz and network.xml here
```
Keep the three .py files in one folder (the script imports `real_data` and `validation_report`).

## 2. Decide the sample rate
`--sample-rate 0.10` -> agents = 10 % x 192,517 residents (Census 2020, Sioux Falls city) x 0.78 (share aged 16+) = **15,016 agents**.
Your previous file had 8,385 agents = 5.6 % of residents 16+, not 10 %.
If your network covers only part of the city, pass the real residents of that area: `--city-population N`
(sum the 2020 census block populations inside the network bounding box).

## 3. Run the baseline
```bash
./run_10pct.sh        # or the python command inside it
```
Read the console / `generation_meta.json`: share of multi-activity plans (~41 %), interzonal legs (~85 %),
mode shares, mean commute (3.3 km), link snapping quality.

## 4. Zones and inter-zonal trips (LeBlanc zones kept)
The 24 zone ids stay in the facility names (`<id>_<zone>`).
* Without extra data, work/secondary destinations come from a gravity model; ~85 % of legs cross a zone boundary.
* Your original file has 100 % interzonal legs because LeBlanc's OD table has an empty diagonal.
* To follow LeBlanc (as Chakirov & Fourie 2014 did): download `SiouxFalls_trips.tntp` from
  https://github.com/bstabler/TransportationNetworks (folder SiouxFalls), put it in `inputs/` and add
  `--leblanc-trips inputs/SiouxFalls_trips.tntp`. The work ZONE is drawn from the OD row of the home zone.
  `--intrazonal-share 0.15` (default) keeps 15 % of work trips inside the home zone; `0` reproduces the paper (100 % interzonal).
  The 15 % is a rough assumption chosen to match the gravity result, not a measured value.
* `generation_meta.json` -> `leblanc_zone_tv_distance_weighted` tells how close the realised zone shares are to the table (small = good).
  Secondary trips are not driven by the OD table.

## 5. Real government data (optional, needs internet)
Download once (see REFERENCES.md): LEHD LODES 8 for South Dakota (`sd_xwalk.csv.gz`, `sd_wac_S000_JT00_<year>.csv.gz`,
`sd_rac_...`, `sd_od_main_JT00_<year>.csv.gz`) from https://lehd.ces.census.gov/data . Check `version.txt` for available years.
Add the four `--lodes-*` flags (see run_10pct.sh). This makes job and home weights follow real block counts and
prints the LODES mean commute next to yours. Do not combine census `--taz-geojson` with `--leblanc-trips`.
NOTE: the LODES code was only tested on fake fixture files.

## 6. Calibrate before training
* Multi-activity share: `--chain-share` (workers with >= 1 non-work stop, default 0.45; try 0.35-0.60).
* Mode shares: `--pt-scale`, `--bike-scale` (PT default 12 % of legs vs < 1-2 % realistic).
* Commute length: `--tau-work-km` (default 4.4) against the LODES / ACS (~17 min) benchmark.
* Several populations for train/test: run with `--seed 1`, `--seed 2`, ... into different `--out-dir`s.

## 7. Use it in MATSim
1. Copy `population_synthetic.xml` and `facilities_synthetic.xml.gz` next to your network; set them in the config
   (`plans.inputPlansFile`, `facilities.inputFacilitiesFile`).
2. Paste `matsim_qsim_snippet.xml` values into the `qsim` module: `flowCapacityFactor` and `storageCapacityFactor` = 0.1
   (equal to the sample rate; some projects use a larger storage factor, e.g. sqrt of the rate).
3. Keep activity types home / work / secondary in the scoring config. Check typical durations vs. the generated plans
   (work ~8.5 h full-time, ~4.8 h part-time; lunch stop ~40 min).
4. Run MATSim once with 0 replanning iterations and check the network: peak speeds and queues should look plausible; adjust the sample rate / factors if everything is free-flowing.

## 8. Check the file
`validation_report.png` and `generation_meta.json` are written every run. Coordinates are plain numbers (numpy repr bug fixed).

## Known limits
No under-16s, every agent travels, one fixed plan per agent, ACS targets partly unverified, facility capacities are
relative weights (they sum to ~69,500 residents, 36 % of the city), no real government/LeBlanc table was processed here.
