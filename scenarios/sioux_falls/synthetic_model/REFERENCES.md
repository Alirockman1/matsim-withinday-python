# Synthetic Sioux Falls – data sources, how they are used, and what is still an assumption

## Run it

```bash
# 1) works now, with the files you already have
python generate_synthetic_siouxfalls.py \
  --facilities facilities_xml.gz --network network.xml \
  --population population_project_specific.xml \
  --sample-rate 0.10 --out-dir out --chain-share 0.45   # --population is optional (link-agreement check only)

# 2) with real government data (download first, see below)
python generate_synthetic_siouxfalls.py \
  --facilities facilities_xml.gz --network network.xml --out-dir out \
  --taz-geojson taz.geojson --rename-by-taz \
  --lodes-xwalk sd_xwalk.csv.gz --lodes-wac sd_wac_S000_JT00_<yr>.csv.gz \
  --lodes-rac sd_rac_S000_JT00_<yr>.csv.gz --lodes-od sd_od_main_JT00_<yr>.csv.gz
```
`out/generation_meta.json` reports link-snapping quality, share of multi-activity plans,
trips per plan, mean commute vs. LODES, and how much LODES job volume needed the fallback allocation.

## What each source feeds

| Layer | Source | Used for |
|---|---|---|
| Facility coordinates | Your `facilities_xml.gz` (Chakirov & Fourie 2014; per the MATSim scenario page built with land-use data from the City of Sioux Falls GIS division) | Geography of homes / work / secondary sites |
| Links | Your `network.xml` | Each facility snapped to nearest car link (freeways and links < 10 m excluded; max 500 m) |
| LeBlanc zones + OD | `SiouxFalls_trips.tntp` (LeBlanc et al. 1975; Transportation Networks for Research) | `--leblanc-trips`: work-destination ZONE drawn from the OD row of the home zone, as in Chakirov & Fourie 2014; site inside the zone by capacity x distance decay |
| Real TAZs | Census TIGER/Line 2010 TAZ (state file for SD, FIPS 46), or MPO layer | `--taz-geojson` re-assigns each facility to a polygon (point-in-polygon, UTM 14N) |
| Jobs per site | LEHD LODES 8 WAC (jobs by work block) + xwalk (block lat/lon) | `--lodes-wac`: block totals split over the block's work facilities |
| Residential weight | LODES 8 RAC (workers by home block) | `--lodes-rac`: relative home density (total residents preserved) |
| Commute length | LODES 8 OD main | `--lodes-od`: benchmark only (prints synthetic vs. LODES mean beeline); tune `--tau-work-km` |
| Chaining | FHWA NPTS 1995 / NHTS tables, Brookings analysis | `--chain-share` and the male/female ratio |

## References (all found during this session)

1. Chakirov, A., Fourie, P. J. (2014). *Enriched Sioux Falls Scenario with Dynamic and Disaggregate Demand.* Arbeitsberichte Verkehrs- und Raumplanung 978, ETH Zurich. DOI 10.3929/ethz-b-000080996. Scenario page: https://matsim.org/gallery/sioux-falls/
   - The authors state the scenario is a fictitious test case, not a replica of the real city. The 24 zones come from LeBlanc et al. (1975).
2. U.S. Census Bureau, LEHD. LODES Version 8 technical documentation: https://lehd.ces.census.gov/data/lodes/LODES8/LODESTechDoc8.4.pdf ; data: https://lehd.ces.census.gov/data
3. U.S. Census Bureau. TIGER/Line Shapefile, 2010 Census Traffic Analysis Zone (TAZ) state-based shapefile: https://catalog.data.gov/dataset/tiger-line-shapefile-2011-series-information-file-for-the-2010-census-traffic-analysis-zon
4. City of Sioux Falls open data portal (GIS layers, incl. zoning and traffic counts): https://dataworks.siouxfalls.gov
5. FHWA, National Personal Transportation Survey 1995, trip-chaining data points: https://www.fhwa.dot.gov/ohim/hs97/nptsdata.htm
   (stops HOME→WORK: men 19.8 %, women 33 %; WORK→HOME: men 46.4 %, women 61.2 %)
6. FHWA, *Mobility Trends – Drivers of Demand* brief (NHTS 2022): workers 2.38, non-workers 1.71 trips per person-day. https://www.fhwa.dot.gov/policy/otps/mobility_trends/pdf/Mobility_Trends_Drivers_Demand_Brief.pdf
7. Brookings, "With commuting down, cities must rethink their transportation networks": share of household trips made on the way to/from work fell from 50 % (2019) to 40 % (2022). https://www.brookings.edu/articles/with-commuting-down-cities-must-rethink-their-transportation-networks/
8. McGuckin, N., Zmud, J., Nakamoto, Y. (2005). Trip-chaining trends in the U.S. *Transportation Research Record*; background note: https://nhts.ornl.gov/assets/2001/download/TripChaining.pdf
9. NHTS data (2022 NextGen): https://nhts.ornl.gov
10. LeBlanc, L.J., Morlok, E.K., Pierskalla, W.P. (1975). An efficient approach to solving the road network equilibrium traffic assignment problem. *Transportation Research* 9, 309-318. Trip table + network: https://github.com/bstabler/TransportationNetworks (folder SiouxFalls) and https://tzin.bgu.ac.il/~bargera/tntp
    - The maintainers state the network is not realistic, node coordinates were generated artificially to reproduce the paper's diagram, and the OD flows are 100x the thousands-of-vehicles-per-day values in the paper's Table 1.
11. Census QuickFacts, Sioux Falls city: https://census.gov/quickfacts/fact/table/siouxfallscitysouthdakota/PST120223 (verified: 24.9 % under 18, 13.7 % aged 65+, 49.5 % female, 2.32 persons/household, 192,517 residents in the 2020 census)
12. Data USA, Sioux Falls City PUMA: https://datausa.io/profile/geo/sioux-falls-city-puma-sd (average commute 17 min, ~2 cars per household)
13. BTS, South Dakota commuting 2000: https://www.bts.gov/archive/publications/state_transportation_statistics/south_dakota/table_04_01 (transit 0.4 %, drove alone 76.6 %, worked at home 6.4 %)

## Review of the numbers hard-coded in the script
Verified this session: age shares (24.9 % / 13.7 %), 49.5 % female, 2.32 persons per household, ~17 min mean commute, ~2 vehicles per household.
Not verified (still `[ACS]`-tagged guesses): labour-force participation 72.6 %, unemployment 2.3 %, share working from home 6-7 %, 5 % zero-vehicle households, the age-by-employment shape, licence and bike ownership, all ASSUMPTION entries.
Known deviations: adults per household in the script (1.89) is ~4 % above 2.32 x 0.78 = 1.81; PT share of legs is 12 % by default, far above the <1 % transit commute share of South Dakota (BTS 2000; no current Sioux Falls figure retrieved) -> use `--pt-scale`; bike share 4 % is also high -> `--bike-scale`.
Facility capacities add up to ~69,500 residents, i.e. ~36 % of the 192,517 city population, so "sample fraction" figures computed from them are not city sampling rates.

## Caveats – please read

* **No real government files were processed.** The sandbox had no internet access, so the TAZ/LODES code paths were tested only on synthetic fixture files. The shipped files in `output_10pct/` use your facility geography and network plus the demographic targets from the earlier script version.
* The `[ACS]` demographic targets inside `CFG` (age bands, employment, vehicles, ...) were **not re-verified** in this session. Check them against current ACS 5-year tables for Sioux Falls city (or pull them from ACS PUMS) before publishing.
* `--chain-share 0.45` is a **calibration choice**, not a measured Sioux Falls value. NPTS 1995 implies roughly 57 % (men) / 74 % (women) of workers stop at least once on the commute if the two directions are independent; the Brookings result suggests chaining on commute trips declined after 2019, so the default sits below that range. Use 0.35–0.60 for sensitivity runs.
* LODES coverage for South Dakota has gaps in some years (e.g. 2017 is missing in LODES 7); check the tech doc / `version.txt` before choosing a year.
* TIGER TAZs are the 2010 census vintage. If you want the zones the Sioux Falls MPO currently uses, request them from the MPO (SECOG); I could not confirm a public download.
* Every synthetic person travels (no stay-at-home agents), so trips per person are higher than the NHTS averages in item 6, which include non-travellers.
* TAZ holes (inner rings) are ignored; facilities outside all polygons keep their old zone.
