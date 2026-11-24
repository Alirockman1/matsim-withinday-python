#!/usr/bin/env bash
# Rebuilds the 10 % population from your input files. Edit the paths below.
set -euo pipefail
IN=./inputs            # put facilities_xml.gz, network.xml (and optionally the LeBlanc table) here
python generate_synthetic_siouxfalls.py \
  --facilities $IN/facilities_xml.gz \
  --network    $IN/network.xml \
  --sample-rate 0.10 --chain-share 0.45 --seed 42 \
  --out-dir output_10pct
# With the LeBlanc OD table (keeps the 24 zones, work zone ~ OD row; 15 % of work trips stay in the home zone):
#   add:  --leblanc-trips $IN/SiouxFalls_trips.tntp --intrazonal-share 0.15
# With Census LODES (after downloading, see README_STEPS.md step 5):
#   add:  --lodes-xwalk $IN/sd_xwalk.csv.gz --lodes-wac $IN/sd_wac_S000_JT00_<yr>.csv.gz \
#         --lodes-rac $IN/sd_rac_S000_JT00_<yr>.csv.gz --lodes-od $IN/sd_od_main_JT00_<yr>.csv.gz
# Mode shares (default PT 12 % of legs is far above Sioux Falls reality):  --pt-scale 0.1 --bike-scale 0.3
