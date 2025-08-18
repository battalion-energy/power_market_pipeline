# ERCOT Parquet File Documentation

Generated: 2025-08-18T17:49:42.350458

## Overview

This document provides comprehensive documentation of all Parquet files in the ERCOT data pipeline.

---

## Table of Contents

- [Combined Price Data](#combined-price-data)
- [Flattened Price Data](#flattened-price-data)
- [Rollup - AS_prices](#rollup---as-prices)
- [Rollup - COP_Snapshots](#rollup---cop-snapshots)
- [Rollup - DAM_Gen_Resources](#rollup---dam-gen-resources)
- [Rollup - DAM_Load_Resources](#rollup---dam-load-resources)
- [Rollup - DA_prices](#rollup---da-prices)
- [Rollup - RT_prices](#rollup---rt-prices)
- [Rollup - SCED_Gen_Resources](#rollup---sced-gen-resources)
- [Rollup - SCED_Load_Resources](#rollup---sced-load-resources)
- [Rollup - combined_test](#rollup---combined-test)
- [Rollup - flattened_test](#rollup---flattened-test)

---

## Combined Price Data

### DA_AS_RT_15min_combined_2010.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/DA_AS_RT_15min_combined_2010.parquet`

**Size**: 363.7 KB

**Rows**: 2,855

**Last Modified**: 2025-08-18T16:05:23.631085


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DeliveryInterval | None | int64 | Yes |
| AS_NSPIN | None | float64 | Yes |
| AS_REGDN | None | float64 | Yes |
| AS_REGUP | None | float64 | Yes |
| AS_RRS | None | float64 | Yes |
| DA_DC_E | None | float64 | Yes |
| DA_DC_L | None | float64 | Yes |
| DA_DC_N | None | float64 | Yes |
| DA_DC_R | None | float64 | Yes |
| DA_DC_S | None | float64 | Yes |
| DA_HB_BUSAVG | None | float64 | Yes |
| DA_HB_HOUSTON | None | float64 | Yes |
| DA_HB_HUBAVG | None | float64 | Yes |
| DA_HB_NORTH | None | float64 | Yes |
| DA_HB_PAN | None | float64 | Yes |
| DA_HB_SOUTH | None | float64 | Yes |
| DA_HB_WEST | None | float64 | Yes |
| DA_LZ_AEN | None | float64 | Yes |
| DA_LZ_CPS | None | float64 | Yes |
| DA_LZ_HOUSTON | None | float64 | Yes |
| DA_LZ_LCRA | None | float64 | Yes |
| DA_LZ_NORTH | None | float64 | Yes |
| DA_LZ_RAYBN | None | float64 | Yes |
| DA_LZ_SOUTH | None | float64 | Yes |
| DA_LZ_WEST | None | float64 | Yes |
| RT_DC_E | None | float64 | Yes |
| RT_DC_L | None | float64 | Yes |
| RT_DC_N | None | float64 | Yes |
| RT_DC_R | None | float64 | Yes |
| RT_DC_S | None | float64 | Yes |
| RT_HB_BUSAVG | None | float64 | Yes |
| RT_HB_HOUSTON | None | float64 | Yes |
| RT_HB_HUBAVG | None | float64 | Yes |
| RT_HB_NORTH | None | float64 | Yes |
| RT_HB_PAN | None | float64 | Yes |
| RT_HB_SOUTH | None | float64 | Yes |
| RT_HB_WEST | None | float64 | Yes |
| RT_LZ_AEN | None | float64 | Yes |
| RT_LZ_CPS | None | float64 | Yes |
| RT_LZ_HOUSTON | None | float64 | Yes |
| RT_LZ_LCRA | None | float64 | Yes |
| RT_LZ_NORTH | None | float64 | Yes |
| RT_LZ_RAYBN | None | float64 | Yes |
| RT_LZ_SOUTH | None | float64 | Yes |
| RT_LZ_WEST | None | float64 | Yes |


### DA_AS_RT_15min_combined_2011.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/DA_AS_RT_15min_combined_2011.parquet`

**Size**: 2.9 MB

**Rows**: 33,579

**Last Modified**: 2025-08-18T16:05:23.718090


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DeliveryInterval | None | int64 | Yes |
| AS_NSPIN | None | float64 | Yes |
| AS_REGDN | None | float64 | Yes |
| AS_REGUP | None | float64 | Yes |
| AS_RRS | None | float64 | Yes |
| DA_DC_E | None | float64 | Yes |
| DA_DC_L | None | float64 | Yes |
| DA_DC_N | None | float64 | Yes |
| DA_DC_R | None | float64 | Yes |
| DA_DC_S | None | float64 | Yes |
| DA_HB_BUSAVG | None | float64 | Yes |
| DA_HB_HOUSTON | None | float64 | Yes |
| DA_HB_HUBAVG | None | float64 | Yes |
| DA_HB_NORTH | None | float64 | Yes |
| DA_HB_PAN | None | float64 | Yes |
| DA_HB_SOUTH | None | float64 | Yes |
| DA_HB_WEST | None | float64 | Yes |
| DA_LZ_AEN | None | float64 | Yes |
| DA_LZ_CPS | None | float64 | Yes |
| DA_LZ_HOUSTON | None | float64 | Yes |
| DA_LZ_LCRA | None | float64 | Yes |
| DA_LZ_NORTH | None | float64 | Yes |
| DA_LZ_RAYBN | None | float64 | Yes |
| DA_LZ_SOUTH | None | float64 | Yes |
| DA_LZ_WEST | None | float64 | Yes |
| RT_DC_E | None | float64 | Yes |
| RT_DC_L | None | float64 | Yes |
| RT_DC_N | None | float64 | Yes |
| RT_DC_R | None | float64 | Yes |
| RT_DC_S | None | float64 | Yes |
| RT_HB_BUSAVG | None | float64 | Yes |
| RT_HB_HOUSTON | None | float64 | Yes |
| RT_HB_HUBAVG | None | float64 | Yes |
| RT_HB_NORTH | None | float64 | Yes |
| RT_HB_PAN | None | float64 | Yes |
| RT_HB_SOUTH | None | float64 | Yes |
| RT_HB_WEST | None | float64 | Yes |
| RT_LZ_AEN | None | float64 | Yes |
| RT_LZ_CPS | None | float64 | Yes |
| RT_LZ_HOUSTON | None | float64 | Yes |
| RT_LZ_LCRA | None | float64 | Yes |
| RT_LZ_NORTH | None | float64 | Yes |
| RT_LZ_RAYBN | None | float64 | Yes |
| RT_LZ_SOUTH | None | float64 | Yes |
| RT_LZ_WEST | None | float64 | Yes |


### DA_AS_RT_15min_combined_2012.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/DA_AS_RT_15min_combined_2012.parquet`

**Size**: 2.6 MB

**Rows**: 33,671

**Last Modified**: 2025-08-18T16:05:23.855096


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DeliveryInterval | None | int64 | Yes |
| AS_NSPIN | None | float64 | Yes |
| AS_REGDN | None | float64 | Yes |
| AS_REGUP | None | float64 | Yes |
| AS_RRS | None | float64 | Yes |
| DA_DC_E | None | float64 | Yes |
| DA_DC_L | None | float64 | Yes |
| DA_DC_N | None | float64 | Yes |
| DA_DC_R | None | float64 | Yes |
| DA_DC_S | None | float64 | Yes |
| DA_HB_BUSAVG | None | float64 | Yes |
| DA_HB_HOUSTON | None | float64 | Yes |
| DA_HB_HUBAVG | None | float64 | Yes |
| DA_HB_NORTH | None | float64 | Yes |
| DA_HB_PAN | None | float64 | Yes |
| DA_HB_SOUTH | None | float64 | Yes |
| DA_HB_WEST | None | float64 | Yes |
| DA_LZ_AEN | None | float64 | Yes |
| DA_LZ_CPS | None | float64 | Yes |
| DA_LZ_HOUSTON | None | float64 | Yes |
| DA_LZ_LCRA | None | float64 | Yes |
| DA_LZ_NORTH | None | float64 | Yes |
| DA_LZ_RAYBN | None | float64 | Yes |
| DA_LZ_SOUTH | None | float64 | Yes |
| DA_LZ_WEST | None | float64 | Yes |
| RT_DC_E | None | float64 | Yes |
| RT_DC_L | None | float64 | Yes |
| RT_DC_N | None | float64 | Yes |
| RT_DC_R | None | float64 | Yes |
| RT_DC_S | None | float64 | Yes |
| RT_HB_BUSAVG | None | float64 | Yes |
| RT_HB_HOUSTON | None | float64 | Yes |
| RT_HB_HUBAVG | None | float64 | Yes |
| RT_HB_NORTH | None | float64 | Yes |
| RT_HB_PAN | None | float64 | Yes |
| RT_HB_SOUTH | None | float64 | Yes |
| RT_HB_WEST | None | float64 | Yes |
| RT_LZ_AEN | None | float64 | Yes |
| RT_LZ_CPS | None | float64 | Yes |
| RT_LZ_HOUSTON | None | float64 | Yes |
| RT_LZ_LCRA | None | float64 | Yes |
| RT_LZ_NORTH | None | float64 | Yes |
| RT_LZ_RAYBN | None | float64 | Yes |
| RT_LZ_SOUTH | None | float64 | Yes |
| RT_LZ_WEST | None | float64 | Yes |


### DA_AS_RT_15min_combined_2013.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/DA_AS_RT_15min_combined_2013.parquet`

**Size**: 2.6 MB

**Rows**: 33,579

**Last Modified**: 2025-08-18T16:05:23.993103


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DeliveryInterval | None | int64 | Yes |
| AS_NSPIN | None | float64 | Yes |
| AS_REGDN | None | float64 | Yes |
| AS_REGUP | None | float64 | Yes |
| AS_RRS | None | float64 | Yes |
| DA_DC_E | None | float64 | Yes |
| DA_DC_L | None | float64 | Yes |
| DA_DC_N | None | float64 | Yes |
| DA_DC_R | None | float64 | Yes |
| DA_DC_S | None | float64 | Yes |
| DA_HB_BUSAVG | None | float64 | Yes |
| DA_HB_HOUSTON | None | float64 | Yes |
| DA_HB_HUBAVG | None | float64 | Yes |
| DA_HB_NORTH | None | float64 | Yes |
| DA_HB_PAN | None | float64 | Yes |
| DA_HB_SOUTH | None | float64 | Yes |
| DA_HB_WEST | None | float64 | Yes |
| DA_LZ_AEN | None | float64 | Yes |
| DA_LZ_CPS | None | float64 | Yes |
| DA_LZ_HOUSTON | None | float64 | Yes |
| DA_LZ_LCRA | None | float64 | Yes |
| DA_LZ_NORTH | None | float64 | Yes |
| DA_LZ_RAYBN | None | float64 | Yes |
| DA_LZ_SOUTH | None | float64 | Yes |
| DA_LZ_WEST | None | float64 | Yes |
| RT_DC_E | None | float64 | Yes |
| RT_DC_L | None | float64 | Yes |
| RT_DC_N | None | float64 | Yes |
| RT_DC_R | None | float64 | Yes |
| RT_DC_S | None | float64 | Yes |
| RT_HB_BUSAVG | None | float64 | Yes |
| RT_HB_HOUSTON | None | float64 | Yes |
| RT_HB_HUBAVG | None | float64 | Yes |
| RT_HB_NORTH | None | float64 | Yes |
| RT_HB_PAN | None | float64 | Yes |
| RT_HB_SOUTH | None | float64 | Yes |
| RT_HB_WEST | None | float64 | Yes |
| RT_LZ_AEN | None | float64 | Yes |
| RT_LZ_CPS | None | float64 | Yes |
| RT_LZ_HOUSTON | None | float64 | Yes |
| RT_LZ_LCRA | None | float64 | Yes |
| RT_LZ_NORTH | None | float64 | Yes |
| RT_LZ_RAYBN | None | float64 | Yes |
| RT_LZ_SOUTH | None | float64 | Yes |
| RT_LZ_WEST | None | float64 | Yes |


### DA_AS_RT_15min_combined_2014.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/DA_AS_RT_15min_combined_2014.parquet`

**Size**: 2.8 MB

**Rows**: 33,579

**Last Modified**: 2025-08-18T16:05:24.119108


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DeliveryInterval | None | int64 | Yes |
| AS_NSPIN | None | float64 | Yes |
| AS_REGDN | None | float64 | Yes |
| AS_REGUP | None | float64 | Yes |
| AS_RRS | None | float64 | Yes |
| DA_DC_E | None | float64 | Yes |
| DA_DC_L | None | float64 | Yes |
| DA_DC_N | None | float64 | Yes |
| DA_DC_R | None | float64 | Yes |
| DA_DC_S | None | float64 | Yes |
| DA_HB_BUSAVG | None | float64 | Yes |
| DA_HB_HOUSTON | None | float64 | Yes |
| DA_HB_HUBAVG | None | float64 | Yes |
| DA_HB_NORTH | None | float64 | Yes |
| DA_HB_PAN | None | float64 | Yes |
| DA_HB_SOUTH | None | float64 | Yes |
| DA_HB_WEST | None | float64 | Yes |
| DA_LZ_AEN | None | float64 | Yes |
| DA_LZ_CPS | None | float64 | Yes |
| DA_LZ_HOUSTON | None | float64 | Yes |
| DA_LZ_LCRA | None | float64 | Yes |
| DA_LZ_NORTH | None | float64 | Yes |
| DA_LZ_RAYBN | None | float64 | Yes |
| DA_LZ_SOUTH | None | float64 | Yes |
| DA_LZ_WEST | None | float64 | Yes |
| RT_DC_E | None | float64 | Yes |
| RT_DC_L | None | float64 | Yes |
| RT_DC_N | None | float64 | Yes |
| RT_DC_R | None | float64 | Yes |
| RT_DC_S | None | float64 | Yes |
| RT_HB_BUSAVG | None | float64 | Yes |
| RT_HB_HOUSTON | None | float64 | Yes |
| RT_HB_HUBAVG | None | float64 | Yes |
| RT_HB_NORTH | None | float64 | Yes |
| RT_HB_PAN | None | float64 | Yes |
| RT_HB_SOUTH | None | float64 | Yes |
| RT_HB_WEST | None | float64 | Yes |
| RT_LZ_AEN | None | float64 | Yes |
| RT_LZ_CPS | None | float64 | Yes |
| RT_LZ_HOUSTON | None | float64 | Yes |
| RT_LZ_LCRA | None | float64 | Yes |
| RT_LZ_NORTH | None | float64 | Yes |
| RT_LZ_RAYBN | None | float64 | Yes |
| RT_LZ_SOUTH | None | float64 | Yes |
| RT_LZ_WEST | None | float64 | Yes |


### DA_AS_RT_15min_combined_2023.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/DA_AS_RT_15min_combined_2023.parquet`

**Size**: 3.3 MB

**Rows**: 33,577

**Last Modified**: 2025-08-18T16:05:25.388168


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DeliveryInterval | None | int64 | Yes |
| AS_ECRS | None | float64 | Yes |
| AS_NSPIN | None | float64 | Yes |
| AS_REGDN | None | float64 | Yes |
| AS_REGUP | None | float64 | Yes |
| AS_RRS | None | float64 | Yes |
| DA_DC_E | None | float64 | Yes |
| DA_DC_L | None | float64 | Yes |
| DA_DC_N | None | float64 | Yes |
| DA_DC_R | None | float64 | Yes |
| DA_DC_S | None | float64 | Yes |
| DA_HB_BUSAVG | None | float64 | Yes |
| DA_HB_HOUSTON | None | float64 | Yes |
| DA_HB_HUBAVG | None | float64 | Yes |
| DA_HB_NORTH | None | float64 | Yes |
| DA_HB_PAN | None | float64 | Yes |
| DA_HB_SOUTH | None | float64 | Yes |
| DA_HB_WEST | None | float64 | Yes |
| DA_LZ_AEN | None | float64 | Yes |
| DA_LZ_CPS | None | float64 | Yes |
| DA_LZ_HOUSTON | None | float64 | Yes |
| DA_LZ_LCRA | None | float64 | Yes |
| DA_LZ_NORTH | None | float64 | Yes |
| DA_LZ_RAYBN | None | float64 | Yes |
| DA_LZ_SOUTH | None | float64 | Yes |
| DA_LZ_WEST | None | float64 | Yes |
| RT_DC_E | None | float64 | Yes |
| RT_DC_L | None | float64 | Yes |
| RT_DC_N | None | float64 | Yes |
| RT_DC_R | None | float64 | Yes |
| RT_DC_S | None | float64 | Yes |
| RT_HB_BUSAVG | None | float64 | Yes |
| RT_HB_HOUSTON | None | float64 | Yes |
| RT_HB_HUBAVG | None | float64 | Yes |
| RT_HB_NORTH | None | float64 | Yes |
| RT_HB_PAN | None | float64 | Yes |
| RT_HB_SOUTH | None | float64 | Yes |
| RT_HB_WEST | None | float64 | Yes |
| RT_LZ_AEN | None | float64 | Yes |
| RT_LZ_CPS | None | float64 | Yes |
| RT_LZ_HOUSTON | None | float64 | Yes |
| RT_LZ_LCRA | None | float64 | Yes |
| RT_LZ_NORTH | None | float64 | Yes |
| RT_LZ_RAYBN | None | float64 | Yes |
| RT_LZ_SOUTH | None | float64 | Yes |
| RT_LZ_WEST | None | float64 | Yes |


### DA_AS_RT_15min_combined_2024.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/DA_AS_RT_15min_combined_2024.parquet`

**Size**: 2.5 MB

**Rows**: 33,671

**Last Modified**: 2025-08-18T16:05:25.522174


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DeliveryInterval | None | int64 | Yes |
| AS_ECRS | None | float64 | Yes |
| AS_NSPIN | None | float64 | Yes |
| AS_REGDN | None | float64 | Yes |
| AS_REGUP | None | float64 | Yes |
| AS_RRS | None | float64 | Yes |
| DA_DC_E | None | float64 | Yes |
| DA_DC_L | None | float64 | Yes |
| DA_DC_N | None | float64 | Yes |
| DA_DC_R | None | float64 | Yes |
| DA_DC_S | None | float64 | Yes |
| DA_HB_BUSAVG | None | float64 | Yes |
| DA_HB_HOUSTON | None | float64 | Yes |
| DA_HB_HUBAVG | None | float64 | Yes |
| DA_HB_NORTH | None | float64 | Yes |
| DA_HB_PAN | None | float64 | Yes |
| DA_HB_SOUTH | None | float64 | Yes |
| DA_HB_WEST | None | float64 | Yes |
| DA_LZ_AEN | None | float64 | Yes |
| DA_LZ_CPS | None | float64 | Yes |
| DA_LZ_HOUSTON | None | float64 | Yes |
| DA_LZ_LCRA | None | float64 | Yes |
| DA_LZ_NORTH | None | float64 | Yes |
| DA_LZ_RAYBN | None | float64 | Yes |
| DA_LZ_SOUTH | None | float64 | Yes |
| DA_LZ_WEST | None | float64 | Yes |
| RT_DC_E | None | float64 | Yes |
| RT_DC_L | None | float64 | Yes |
| RT_DC_N | None | float64 | Yes |
| RT_DC_R | None | float64 | Yes |
| RT_DC_S | None | float64 | Yes |
| RT_HB_BUSAVG | None | float64 | Yes |
| RT_HB_HOUSTON | None | float64 | Yes |
| RT_HB_HUBAVG | None | float64 | Yes |
| RT_HB_NORTH | None | float64 | Yes |
| RT_HB_PAN | None | float64 | Yes |
| RT_HB_SOUTH | None | float64 | Yes |
| RT_HB_WEST | None | float64 | Yes |
| RT_LZ_AEN | None | float64 | Yes |
| RT_LZ_CPS | None | float64 | Yes |
| RT_LZ_HOUSTON | None | float64 | Yes |
| RT_LZ_LCRA | None | float64 | Yes |
| RT_LZ_NORTH | None | float64 | Yes |
| RT_LZ_RAYBN | None | float64 | Yes |
| RT_LZ_SOUTH | None | float64 | Yes |
| RT_LZ_WEST | None | float64 | Yes |


### DA_AS_RT_5min_combined_2023.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/DA_AS_RT_5min_combined_2023.parquet`

**Size**: 2.6 MB

**Rows**: 33,577

**Last Modified**: 2025-08-17T23:54:34.854155


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DeliveryInterval | None | int64 | Yes |
| AS_ECRS | None | float64 | Yes |
| AS_NSPIN | None | float64 | Yes |
| AS_REGDN | None | float64 | Yes |
| AS_REGUP | None | float64 | Yes |
| AS_RRS | None | float64 | Yes |
| DA_DC_E | None | float64 | Yes |
| DA_DC_L | None | float64 | Yes |
| DA_DC_N | None | float64 | Yes |
| DA_DC_R | None | float64 | Yes |
| DA_DC_S | None | float64 | Yes |
| DA_HB_BUSAVG | None | float64 | Yes |
| DA_HB_HOUSTON | None | float64 | Yes |
| DA_HB_HUBAVG | None | float64 | Yes |
| DA_HB_NORTH | None | float64 | Yes |
| DA_HB_PAN | None | float64 | Yes |
| DA_HB_SOUTH | None | float64 | Yes |
| DA_HB_WEST | None | float64 | Yes |
| DA_LZ_AEN | None | float64 | Yes |
| DA_LZ_CPS | None | float64 | Yes |
| DA_LZ_HOUSTON | None | float64 | Yes |
| DA_LZ_LCRA | None | float64 | Yes |
| DA_LZ_NORTH | None | float64 | Yes |
| DA_LZ_RAYBN | None | float64 | Yes |
| DA_LZ_SOUTH | None | float64 | Yes |
| DA_LZ_WEST | None | float64 | Yes |
| RT_DC_E | None | float64 | Yes |
| RT_DC_L | None | float64 | Yes |
| RT_DC_N | None | float64 | Yes |
| RT_DC_R | None | float64 | Yes |
| RT_DC_S | None | float64 | Yes |
| RT_HB_BUSAVG | None | float64 | Yes |
| RT_HB_HOUSTON | None | float64 | Yes |
| RT_HB_HUBAVG | None | float64 | Yes |
| RT_HB_NORTH | None | float64 | Yes |
| RT_HB_PAN | None | float64 | Yes |
| RT_HB_SOUTH | None | float64 | Yes |
| RT_HB_WEST | None | float64 | Yes |
| RT_LZ_AEN | None | float64 | Yes |
| RT_LZ_CPS | None | float64 | Yes |
| RT_LZ_HOUSTON | None | float64 | Yes |
| RT_LZ_LCRA | None | float64 | Yes |
| RT_LZ_NORTH | None | float64 | Yes |
| RT_LZ_RAYBN | None | float64 | Yes |
| RT_LZ_SOUTH | None | float64 | Yes |
| RT_LZ_WEST | None | float64 | Yes |


### DA_AS_RT_5min_combined_2024.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/DA_AS_RT_5min_combined_2024.parquet`

**Size**: 2.5 MB

**Rows**: 33,671

**Last Modified**: 2025-08-17T23:54:34.978161


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DeliveryInterval | None | int64 | Yes |
| AS_ECRS | None | float64 | Yes |
| AS_NSPIN | None | float64 | Yes |
| AS_REGDN | None | float64 | Yes |
| AS_REGUP | None | float64 | Yes |
| AS_RRS | None | float64 | Yes |
| DA_DC_E | None | float64 | Yes |
| DA_DC_L | None | float64 | Yes |
| DA_DC_N | None | float64 | Yes |
| DA_DC_R | None | float64 | Yes |
| DA_DC_S | None | float64 | Yes |
| DA_HB_BUSAVG | None | float64 | Yes |
| DA_HB_HOUSTON | None | float64 | Yes |
| DA_HB_HUBAVG | None | float64 | Yes |
| DA_HB_NORTH | None | float64 | Yes |
| DA_HB_PAN | None | float64 | Yes |
| DA_HB_SOUTH | None | float64 | Yes |
| DA_HB_WEST | None | float64 | Yes |
| DA_LZ_AEN | None | float64 | Yes |
| DA_LZ_CPS | None | float64 | Yes |
| DA_LZ_HOUSTON | None | float64 | Yes |
| DA_LZ_LCRA | None | float64 | Yes |
| DA_LZ_NORTH | None | float64 | Yes |
| DA_LZ_RAYBN | None | float64 | Yes |
| DA_LZ_SOUTH | None | float64 | Yes |
| DA_LZ_WEST | None | float64 | Yes |
| RT_DC_E | None | float64 | Yes |
| RT_DC_L | None | float64 | Yes |
| RT_DC_N | None | float64 | Yes |
| RT_DC_R | None | float64 | Yes |
| RT_DC_S | None | float64 | Yes |
| RT_HB_BUSAVG | None | float64 | Yes |
| RT_HB_HOUSTON | None | float64 | Yes |
| RT_HB_HUBAVG | None | float64 | Yes |
| RT_HB_NORTH | None | float64 | Yes |
| RT_HB_PAN | None | float64 | Yes |
| RT_HB_SOUTH | None | float64 | Yes |
| RT_HB_WEST | None | float64 | Yes |
| RT_LZ_AEN | None | float64 | Yes |
| RT_LZ_CPS | None | float64 | Yes |
| RT_LZ_HOUSTON | None | float64 | Yes |
| RT_LZ_LCRA | None | float64 | Yes |
| RT_LZ_NORTH | None | float64 | Yes |
| RT_LZ_RAYBN | None | float64 | Yes |
| RT_LZ_SOUTH | None | float64 | Yes |
| RT_LZ_WEST | None | float64 | Yes |


### DA_AS_RT_combined_2023.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/DA_AS_RT_combined_2023.parquet`

**Size**: 1.7 MB

**Rows**: 8,830

**Last Modified**: 2025-08-18T16:05:23.433076


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DA_DC_E | None | float64 | Yes |
| DA_DC_L | None | float64 | Yes |
| DA_DC_N | None | float64 | Yes |
| DA_DC_R | None | float64 | Yes |
| DA_DC_S | None | float64 | Yes |
| DA_HB_BUSAVG | None | float64 | Yes |
| DA_HB_HOUSTON | None | float64 | Yes |
| DA_HB_HUBAVG | None | float64 | Yes |
| DA_HB_NORTH | None | float64 | Yes |
| DA_HB_PAN | None | float64 | Yes |
| DA_HB_SOUTH | None | float64 | Yes |
| DA_HB_WEST | None | float64 | Yes |
| DA_LZ_AEN | None | float64 | Yes |
| DA_LZ_CPS | None | float64 | Yes |
| DA_LZ_HOUSTON | None | float64 | Yes |
| DA_LZ_LCRA | None | float64 | Yes |
| DA_LZ_NORTH | None | float64 | Yes |
| DA_LZ_RAYBN | None | float64 | Yes |
| DA_LZ_SOUTH | None | float64 | Yes |
| DA_LZ_WEST | None | float64 | Yes |
| AS_ECRS | None | float64 | Yes |
| AS_NSPIN | None | float64 | Yes |
| AS_REGDN | None | float64 | Yes |
| AS_REGUP | None | float64 | Yes |
| AS_RRS | None | float64 | Yes |
| RT_DC_E | None | float64 | Yes |
| RT_DC_L | None | float64 | Yes |
| RT_DC_N | None | float64 | Yes |
| RT_DC_R | None | float64 | Yes |
| RT_DC_S | None | float64 | Yes |
| RT_HB_BUSAVG | None | float64 | Yes |
| RT_HB_HOUSTON | None | float64 | Yes |
| RT_HB_HUBAVG | None | float64 | Yes |
| RT_HB_NORTH | None | float64 | Yes |
| RT_HB_PAN | None | float64 | Yes |
| RT_HB_SOUTH | None | float64 | Yes |
| RT_HB_WEST | None | float64 | Yes |
| RT_LZ_AEN | None | float64 | Yes |
| RT_LZ_CPS | None | float64 | Yes |
| RT_LZ_HOUSTON | None | float64 | Yes |
| RT_LZ_LCRA | None | float64 | Yes |
| RT_LZ_NORTH | None | float64 | Yes |
| RT_LZ_RAYBN | None | float64 | Yes |
| RT_LZ_SOUTH | None | float64 | Yes |
| RT_LZ_WEST | None | float64 | Yes |


### DA_AS_RT_combined_2024.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/DA_AS_RT_combined_2024.parquet`

**Size**: 1.3 MB

**Rows**: 8,585

**Last Modified**: 2025-08-18T16:05:23.511080


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DA_DC_E | None | float64 | Yes |
| DA_DC_L | None | float64 | Yes |
| DA_DC_N | None | float64 | Yes |
| DA_DC_R | None | float64 | Yes |
| DA_DC_S | None | float64 | Yes |
| DA_HB_BUSAVG | None | float64 | Yes |
| DA_HB_HOUSTON | None | float64 | Yes |
| DA_HB_HUBAVG | None | float64 | Yes |
| DA_HB_NORTH | None | float64 | Yes |
| DA_HB_PAN | None | float64 | Yes |
| DA_HB_SOUTH | None | float64 | Yes |
| DA_HB_WEST | None | float64 | Yes |
| DA_LZ_AEN | None | float64 | Yes |
| DA_LZ_CPS | None | float64 | Yes |
| DA_LZ_HOUSTON | None | float64 | Yes |
| DA_LZ_LCRA | None | float64 | Yes |
| DA_LZ_NORTH | None | float64 | Yes |
| DA_LZ_RAYBN | None | float64 | Yes |
| DA_LZ_SOUTH | None | float64 | Yes |
| DA_LZ_WEST | None | float64 | Yes |
| AS_ECRS | None | float64 | Yes |
| AS_NSPIN | None | float64 | Yes |
| AS_REGDN | None | float64 | Yes |
| AS_REGUP | None | float64 | Yes |
| AS_RRS | None | float64 | Yes |
| RT_DC_E | None | float64 | Yes |
| RT_DC_L | None | float64 | Yes |
| RT_DC_N | None | float64 | Yes |
| RT_DC_R | None | float64 | Yes |
| RT_DC_S | None | float64 | Yes |
| RT_HB_BUSAVG | None | float64 | Yes |
| RT_HB_HOUSTON | None | float64 | Yes |
| RT_HB_HUBAVG | None | float64 | Yes |
| RT_HB_NORTH | None | float64 | Yes |
| RT_HB_PAN | None | float64 | Yes |
| RT_HB_SOUTH | None | float64 | Yes |
| RT_HB_WEST | None | float64 | Yes |
| RT_LZ_AEN | None | float64 | Yes |
| RT_LZ_CPS | None | float64 | Yes |
| RT_LZ_HOUSTON | None | float64 | Yes |
| RT_LZ_LCRA | None | float64 | Yes |
| RT_LZ_NORTH | None | float64 | Yes |
| RT_LZ_RAYBN | None | float64 | Yes |
| RT_LZ_SOUTH | None | float64 | Yes |
| RT_LZ_WEST | None | float64 | Yes |


### DA_AS_combined_2023.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/DA_AS_combined_2023.parquet`

**Size**: 831.3 KB

**Rows**: 8,759

**Last Modified**: 2025-08-18T17:38:27.034140


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DC_E | None | float64 | Yes |
| DC_L | None | float64 | Yes |
| DC_N | None | float64 | Yes |
| DC_R | None | float64 | Yes |
| DC_S | None | float64 | Yes |
| HB_BUSAVG | None | float64 | Yes |
| HB_HOUSTON | None | float64 | Yes |
| HB_HUBAVG | None | float64 | Yes |
| HB_NORTH | None | float64 | Yes |
| HB_PAN | None | float64 | Yes |
| HB_SOUTH | None | float64 | Yes |
| HB_WEST | None | float64 | Yes |
| LZ_AEN | None | float64 | Yes |
| LZ_CPS | None | float64 | Yes |
| LZ_HOUSTON | None | float64 | Yes |
| LZ_LCRA | None | float64 | Yes |
| LZ_NORTH | None | float64 | Yes |
| LZ_RAYBN | None | float64 | Yes |
| LZ_SOUTH | None | float64 | Yes |
| LZ_WEST | None | float64 | Yes |
| ECRS | None | float64 | Yes |
| NSPIN | None | float64 | Yes |
| REGDN | None | float64 | Yes |
| REGUP | None | float64 | Yes |
| RRS | None | float64 | Yes |


### DA_AS_combined_2024.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/DA_AS_combined_2024.parquet`

**Size**: 833.0 KB

**Rows**: 8,783

**Last Modified**: 2025-08-18T17:47:21.737889


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| DeliveryDate | Date | object | Yes |
| DeliveryDateStr | String | object | Yes |
| datetime_ts | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DC_E | None | float64 | Yes |
| DC_L | None | float64 | Yes |
| DC_N | None | float64 | Yes |
| DC_R | None | float64 | Yes |
| DC_S | None | float64 | Yes |
| HB_BUSAVG | None | float64 | Yes |
| HB_HOUSTON | None | float64 | Yes |
| HB_HUBAVG | None | float64 | Yes |
| HB_NORTH | None | float64 | Yes |
| HB_PAN | None | float64 | Yes |
| HB_SOUTH | None | float64 | Yes |
| HB_WEST | None | float64 | Yes |
| LZ_AEN | None | float64 | Yes |
| LZ_CPS | None | float64 | Yes |
| LZ_HOUSTON | None | float64 | Yes |
| LZ_LCRA | None | float64 | Yes |
| LZ_NORTH | None | float64 | Yes |
| LZ_RAYBN | None | float64 | Yes |
| LZ_SOUTH | None | float64 | Yes |
| LZ_WEST | None | float64 | Yes |
| ECRS | None | float64 | Yes |
| NSPIN | None | float64 | Yes |
| REGDN | None | float64 | Yes |
| REGUP | None | float64 | Yes |
| RRS | None | float64 | Yes |


### DA_AS_RT_15min_combined_2023_01.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_RT_15min_combined/DA_AS_RT_15min_combined_2023_01.parquet`

**Size**: 436.5 KB

**Rows**: 2,852

**Last Modified**: 2025-08-18T16:05:25.405169


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DeliveryInterval | None | int64 | Yes |
| AS_ECRS | None | float64 | Yes |
| AS_NSPIN | None | float64 | Yes |
| AS_REGDN | None | float64 | Yes |
| AS_REGUP | None | float64 | Yes |
| AS_RRS | None | float64 | Yes |
| DA_DC_E | None | float64 | Yes |
| DA_DC_L | None | float64 | Yes |
| DA_DC_N | None | float64 | Yes |
| DA_DC_R | None | float64 | Yes |
| DA_DC_S | None | float64 | Yes |
| DA_HB_BUSAVG | None | float64 | Yes |
| DA_HB_HOUSTON | None | float64 | Yes |
| DA_HB_HUBAVG | None | float64 | Yes |
| DA_HB_NORTH | None | float64 | Yes |
| DA_HB_PAN | None | float64 | Yes |
| DA_HB_SOUTH | None | float64 | Yes |
| DA_HB_WEST | None | float64 | Yes |
| DA_LZ_AEN | None | float64 | Yes |
| DA_LZ_CPS | None | float64 | Yes |
| DA_LZ_HOUSTON | None | float64 | Yes |
| DA_LZ_LCRA | None | float64 | Yes |
| DA_LZ_NORTH | None | float64 | Yes |
| DA_LZ_RAYBN | None | float64 | Yes |
| DA_LZ_SOUTH | None | float64 | Yes |
| DA_LZ_WEST | None | float64 | Yes |
| RT_DC_E | None | float64 | Yes |
| RT_DC_L | None | float64 | Yes |
| RT_DC_N | None | float64 | Yes |
| RT_DC_R | None | float64 | Yes |
| RT_DC_S | None | float64 | Yes |
| RT_HB_BUSAVG | None | float64 | Yes |
| RT_HB_HOUSTON | None | float64 | Yes |
| RT_HB_HUBAVG | None | float64 | Yes |
| RT_HB_NORTH | None | float64 | Yes |
| RT_HB_PAN | None | float64 | Yes |
| RT_HB_SOUTH | None | float64 | Yes |
| RT_HB_WEST | None | float64 | Yes |
| RT_LZ_AEN | None | float64 | Yes |
| RT_LZ_CPS | None | float64 | Yes |
| RT_LZ_HOUSTON | None | float64 | Yes |
| RT_LZ_LCRA | None | float64 | Yes |
| RT_LZ_NORTH | None | float64 | Yes |
| RT_LZ_RAYBN | None | float64 | Yes |
| RT_LZ_SOUTH | None | float64 | Yes |
| RT_LZ_WEST | None | float64 | Yes |


### DA_AS_RT_15min_combined_2023_02.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_RT_15min_combined/DA_AS_RT_15min_combined_2023_02.parquet`

**Size**: 407.5 KB

**Rows**: 2,576

**Last Modified**: 2025-08-18T16:05:25.410169


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DeliveryInterval | None | int64 | Yes |
| AS_ECRS | None | float64 | Yes |
| AS_NSPIN | None | float64 | Yes |
| AS_REGDN | None | float64 | Yes |
| AS_REGUP | None | float64 | Yes |
| AS_RRS | None | float64 | Yes |
| DA_DC_E | None | float64 | Yes |
| DA_DC_L | None | float64 | Yes |
| DA_DC_N | None | float64 | Yes |
| DA_DC_R | None | float64 | Yes |
| DA_DC_S | None | float64 | Yes |
| DA_HB_BUSAVG | None | float64 | Yes |
| DA_HB_HOUSTON | None | float64 | Yes |
| DA_HB_HUBAVG | None | float64 | Yes |
| DA_HB_NORTH | None | float64 | Yes |
| DA_HB_PAN | None | float64 | Yes |
| DA_HB_SOUTH | None | float64 | Yes |
| DA_HB_WEST | None | float64 | Yes |
| DA_LZ_AEN | None | float64 | Yes |
| DA_LZ_CPS | None | float64 | Yes |
| DA_LZ_HOUSTON | None | float64 | Yes |
| DA_LZ_LCRA | None | float64 | Yes |
| DA_LZ_NORTH | None | float64 | Yes |
| DA_LZ_RAYBN | None | float64 | Yes |
| DA_LZ_SOUTH | None | float64 | Yes |
| DA_LZ_WEST | None | float64 | Yes |
| RT_DC_E | None | float64 | Yes |
| RT_DC_L | None | float64 | Yes |
| RT_DC_N | None | float64 | Yes |
| RT_DC_R | None | float64 | Yes |
| RT_DC_S | None | float64 | Yes |
| RT_HB_BUSAVG | None | float64 | Yes |
| RT_HB_HOUSTON | None | float64 | Yes |
| RT_HB_HUBAVG | None | float64 | Yes |
| RT_HB_NORTH | None | float64 | Yes |
| RT_HB_PAN | None | float64 | Yes |
| RT_HB_SOUTH | None | float64 | Yes |
| RT_HB_WEST | None | float64 | Yes |
| RT_LZ_AEN | None | float64 | Yes |
| RT_LZ_CPS | None | float64 | Yes |
| RT_LZ_HOUSTON | None | float64 | Yes |
| RT_LZ_LCRA | None | float64 | Yes |
| RT_LZ_NORTH | None | float64 | Yes |
| RT_LZ_RAYBN | None | float64 | Yes |
| RT_LZ_SOUTH | None | float64 | Yes |
| RT_LZ_WEST | None | float64 | Yes |


### DA_AS_RT_15min_combined_2023_03.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_RT_15min_combined/DA_AS_RT_15min_combined_2023_03.parquet`

**Size**: 445.1 KB

**Rows**: 2,848

**Last Modified**: 2025-08-18T16:05:25.415169


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DeliveryInterval | None | int64 | Yes |
| AS_ECRS | None | float64 | Yes |
| AS_NSPIN | None | float64 | Yes |
| AS_REGDN | None | float64 | Yes |
| AS_REGUP | None | float64 | Yes |
| AS_RRS | None | float64 | Yes |
| DA_DC_E | None | float64 | Yes |
| DA_DC_L | None | float64 | Yes |
| DA_DC_N | None | float64 | Yes |
| DA_DC_R | None | float64 | Yes |
| DA_DC_S | None | float64 | Yes |
| DA_HB_BUSAVG | None | float64 | Yes |
| DA_HB_HOUSTON | None | float64 | Yes |
| DA_HB_HUBAVG | None | float64 | Yes |
| DA_HB_NORTH | None | float64 | Yes |
| DA_HB_PAN | None | float64 | Yes |
| DA_HB_SOUTH | None | float64 | Yes |
| DA_HB_WEST | None | float64 | Yes |
| DA_LZ_AEN | None | float64 | Yes |
| DA_LZ_CPS | None | float64 | Yes |
| DA_LZ_HOUSTON | None | float64 | Yes |
| DA_LZ_LCRA | None | float64 | Yes |
| DA_LZ_NORTH | None | float64 | Yes |
| DA_LZ_RAYBN | None | float64 | Yes |
| DA_LZ_SOUTH | None | float64 | Yes |
| DA_LZ_WEST | None | float64 | Yes |
| RT_DC_E | None | float64 | Yes |
| RT_DC_L | None | float64 | Yes |
| RT_DC_N | None | float64 | Yes |
| RT_DC_R | None | float64 | Yes |
| RT_DC_S | None | float64 | Yes |
| RT_HB_BUSAVG | None | float64 | Yes |
| RT_HB_HOUSTON | None | float64 | Yes |
| RT_HB_HUBAVG | None | float64 | Yes |
| RT_HB_NORTH | None | float64 | Yes |
| RT_HB_PAN | None | float64 | Yes |
| RT_HB_SOUTH | None | float64 | Yes |
| RT_HB_WEST | None | float64 | Yes |
| RT_LZ_AEN | None | float64 | Yes |
| RT_LZ_CPS | None | float64 | Yes |
| RT_LZ_HOUSTON | None | float64 | Yes |
| RT_LZ_LCRA | None | float64 | Yes |
| RT_LZ_NORTH | None | float64 | Yes |
| RT_LZ_RAYBN | None | float64 | Yes |
| RT_LZ_SOUTH | None | float64 | Yes |
| RT_LZ_WEST | None | float64 | Yes |


### DA_AS_RT_15min_combined_2023_04.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_RT_15min_combined/DA_AS_RT_15min_combined_2023_04.parquet`

**Size**: 428.1 KB

**Rows**: 2,758

**Last Modified**: 2025-08-18T16:05:25.420170


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DeliveryInterval | None | int64 | Yes |
| AS_ECRS | None | float64 | Yes |
| AS_NSPIN | None | float64 | Yes |
| AS_REGDN | None | float64 | Yes |
| AS_REGUP | None | float64 | Yes |
| AS_RRS | None | float64 | Yes |
| DA_DC_E | None | float64 | Yes |
| DA_DC_L | None | float64 | Yes |
| DA_DC_N | None | float64 | Yes |
| DA_DC_R | None | float64 | Yes |
| DA_DC_S | None | float64 | Yes |
| DA_HB_BUSAVG | None | float64 | Yes |
| DA_HB_HOUSTON | None | float64 | Yes |
| DA_HB_HUBAVG | None | float64 | Yes |
| DA_HB_NORTH | None | float64 | Yes |
| DA_HB_PAN | None | float64 | Yes |
| DA_HB_SOUTH | None | float64 | Yes |
| DA_HB_WEST | None | float64 | Yes |
| DA_LZ_AEN | None | float64 | Yes |
| DA_LZ_CPS | None | float64 | Yes |
| DA_LZ_HOUSTON | None | float64 | Yes |
| DA_LZ_LCRA | None | float64 | Yes |
| DA_LZ_NORTH | None | float64 | Yes |
| DA_LZ_RAYBN | None | float64 | Yes |
| DA_LZ_SOUTH | None | float64 | Yes |
| DA_LZ_WEST | None | float64 | Yes |
| RT_DC_E | None | float64 | Yes |
| RT_DC_L | None | float64 | Yes |
| RT_DC_N | None | float64 | Yes |
| RT_DC_R | None | float64 | Yes |
| RT_DC_S | None | float64 | Yes |
| RT_HB_BUSAVG | None | float64 | Yes |
| RT_HB_HOUSTON | None | float64 | Yes |
| RT_HB_HUBAVG | None | float64 | Yes |
| RT_HB_NORTH | None | float64 | Yes |
| RT_HB_PAN | None | float64 | Yes |
| RT_HB_SOUTH | None | float64 | Yes |
| RT_HB_WEST | None | float64 | Yes |
| RT_LZ_AEN | None | float64 | Yes |
| RT_LZ_CPS | None | float64 | Yes |
| RT_LZ_HOUSTON | None | float64 | Yes |
| RT_LZ_LCRA | None | float64 | Yes |
| RT_LZ_NORTH | None | float64 | Yes |
| RT_LZ_RAYBN | None | float64 | Yes |
| RT_LZ_SOUTH | None | float64 | Yes |
| RT_LZ_WEST | None | float64 | Yes |


### DA_AS_RT_15min_combined_2023_05.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_RT_15min_combined/DA_AS_RT_15min_combined_2023_05.parquet`

**Size**: 422.5 KB

**Rows**: 2,852

**Last Modified**: 2025-08-18T16:05:25.425170


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DeliveryInterval | None | int64 | Yes |
| AS_ECRS | None | float64 | Yes |
| AS_NSPIN | None | float64 | Yes |
| AS_REGDN | None | float64 | Yes |
| AS_REGUP | None | float64 | Yes |
| AS_RRS | None | float64 | Yes |
| DA_DC_E | None | float64 | Yes |
| DA_DC_L | None | float64 | Yes |
| DA_DC_N | None | float64 | Yes |
| DA_DC_R | None | float64 | Yes |
| DA_DC_S | None | float64 | Yes |
| DA_HB_BUSAVG | None | float64 | Yes |
| DA_HB_HOUSTON | None | float64 | Yes |
| DA_HB_HUBAVG | None | float64 | Yes |
| DA_HB_NORTH | None | float64 | Yes |
| DA_HB_PAN | None | float64 | Yes |
| DA_HB_SOUTH | None | float64 | Yes |
| DA_HB_WEST | None | float64 | Yes |
| DA_LZ_AEN | None | float64 | Yes |
| DA_LZ_CPS | None | float64 | Yes |
| DA_LZ_HOUSTON | None | float64 | Yes |
| DA_LZ_LCRA | None | float64 | Yes |
| DA_LZ_NORTH | None | float64 | Yes |
| DA_LZ_RAYBN | None | float64 | Yes |
| DA_LZ_SOUTH | None | float64 | Yes |
| DA_LZ_WEST | None | float64 | Yes |
| RT_DC_E | None | float64 | Yes |
| RT_DC_L | None | float64 | Yes |
| RT_DC_N | None | float64 | Yes |
| RT_DC_R | None | float64 | Yes |
| RT_DC_S | None | float64 | Yes |
| RT_HB_BUSAVG | None | float64 | Yes |
| RT_HB_HOUSTON | None | float64 | Yes |
| RT_HB_HUBAVG | None | float64 | Yes |
| RT_HB_NORTH | None | float64 | Yes |
| RT_HB_PAN | None | float64 | Yes |
| RT_HB_SOUTH | None | float64 | Yes |
| RT_HB_WEST | None | float64 | Yes |
| RT_LZ_AEN | None | float64 | Yes |
| RT_LZ_CPS | None | float64 | Yes |
| RT_LZ_HOUSTON | None | float64 | Yes |
| RT_LZ_LCRA | None | float64 | Yes |
| RT_LZ_NORTH | None | float64 | Yes |
| RT_LZ_RAYBN | None | float64 | Yes |
| RT_LZ_SOUTH | None | float64 | Yes |
| RT_LZ_WEST | None | float64 | Yes |


### DA_AS_RT_15min_combined_2023_06.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_RT_15min_combined/DA_AS_RT_15min_combined_2023_06.parquet`

**Size**: 422.7 KB

**Rows**: 2,760

**Last Modified**: 2025-08-18T16:05:25.430170


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DeliveryInterval | None | int64 | Yes |
| AS_ECRS | None | float64 | Yes |
| AS_NSPIN | None | float64 | Yes |
| AS_REGDN | None | float64 | Yes |
| AS_REGUP | None | float64 | Yes |
| AS_RRS | None | float64 | Yes |
| DA_DC_E | None | float64 | Yes |
| DA_DC_L | None | float64 | Yes |
| DA_DC_N | None | float64 | Yes |
| DA_DC_R | None | float64 | Yes |
| DA_DC_S | None | float64 | Yes |
| DA_HB_BUSAVG | None | float64 | Yes |
| DA_HB_HOUSTON | None | float64 | Yes |
| DA_HB_HUBAVG | None | float64 | Yes |
| DA_HB_NORTH | None | float64 | Yes |
| DA_HB_PAN | None | float64 | Yes |
| DA_HB_SOUTH | None | float64 | Yes |
| DA_HB_WEST | None | float64 | Yes |
| DA_LZ_AEN | None | float64 | Yes |
| DA_LZ_CPS | None | float64 | Yes |
| DA_LZ_HOUSTON | None | float64 | Yes |
| DA_LZ_LCRA | None | float64 | Yes |
| DA_LZ_NORTH | None | float64 | Yes |
| DA_LZ_RAYBN | None | float64 | Yes |
| DA_LZ_SOUTH | None | float64 | Yes |
| DA_LZ_WEST | None | float64 | Yes |
| RT_DC_E | None | float64 | Yes |
| RT_DC_L | None | float64 | Yes |
| RT_DC_N | None | float64 | Yes |
| RT_DC_R | None | float64 | Yes |
| RT_DC_S | None | float64 | Yes |
| RT_HB_BUSAVG | None | float64 | Yes |
| RT_HB_HOUSTON | None | float64 | Yes |
| RT_HB_HUBAVG | None | float64 | Yes |
| RT_HB_NORTH | None | float64 | Yes |
| RT_HB_PAN | None | float64 | Yes |
| RT_HB_SOUTH | None | float64 | Yes |
| RT_HB_WEST | None | float64 | Yes |
| RT_LZ_AEN | None | float64 | Yes |
| RT_LZ_CPS | None | float64 | Yes |
| RT_LZ_HOUSTON | None | float64 | Yes |
| RT_LZ_LCRA | None | float64 | Yes |
| RT_LZ_NORTH | None | float64 | Yes |
| RT_LZ_RAYBN | None | float64 | Yes |
| RT_LZ_SOUTH | None | float64 | Yes |
| RT_LZ_WEST | None | float64 | Yes |


### DA_AS_RT_15min_combined_2023_07.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_RT_15min_combined/DA_AS_RT_15min_combined_2023_07.parquet`

**Size**: 426.2 KB

**Rows**: 2,852

**Last Modified**: 2025-08-18T16:05:25.435170


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DeliveryInterval | None | int64 | Yes |
| AS_ECRS | None | float64 | Yes |
| AS_NSPIN | None | float64 | Yes |
| AS_REGDN | None | float64 | Yes |
| AS_REGUP | None | float64 | Yes |
| AS_RRS | None | float64 | Yes |
| DA_DC_E | None | float64 | Yes |
| DA_DC_L | None | float64 | Yes |
| DA_DC_N | None | float64 | Yes |
| DA_DC_R | None | float64 | Yes |
| DA_DC_S | None | float64 | Yes |
| DA_HB_BUSAVG | None | float64 | Yes |
| DA_HB_HOUSTON | None | float64 | Yes |
| DA_HB_HUBAVG | None | float64 | Yes |
| DA_HB_NORTH | None | float64 | Yes |
| DA_HB_PAN | None | float64 | Yes |
| DA_HB_SOUTH | None | float64 | Yes |
| DA_HB_WEST | None | float64 | Yes |
| DA_LZ_AEN | None | float64 | Yes |
| DA_LZ_CPS | None | float64 | Yes |
| DA_LZ_HOUSTON | None | float64 | Yes |
| DA_LZ_LCRA | None | float64 | Yes |
| DA_LZ_NORTH | None | float64 | Yes |
| DA_LZ_RAYBN | None | float64 | Yes |
| DA_LZ_SOUTH | None | float64 | Yes |
| DA_LZ_WEST | None | float64 | Yes |
| RT_DC_E | None | float64 | Yes |
| RT_DC_L | None | float64 | Yes |
| RT_DC_N | None | float64 | Yes |
| RT_DC_R | None | float64 | Yes |
| RT_DC_S | None | float64 | Yes |
| RT_HB_BUSAVG | None | float64 | Yes |
| RT_HB_HOUSTON | None | float64 | Yes |
| RT_HB_HUBAVG | None | float64 | Yes |
| RT_HB_NORTH | None | float64 | Yes |
| RT_HB_PAN | None | float64 | Yes |
| RT_HB_SOUTH | None | float64 | Yes |
| RT_HB_WEST | None | float64 | Yes |
| RT_LZ_AEN | None | float64 | Yes |
| RT_LZ_CPS | None | float64 | Yes |
| RT_LZ_HOUSTON | None | float64 | Yes |
| RT_LZ_LCRA | None | float64 | Yes |
| RT_LZ_NORTH | None | float64 | Yes |
| RT_LZ_RAYBN | None | float64 | Yes |
| RT_LZ_SOUTH | None | float64 | Yes |
| RT_LZ_WEST | None | float64 | Yes |


### DA_AS_RT_15min_combined_2023_08.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_RT_15min_combined/DA_AS_RT_15min_combined_2023_08.parquet`

**Size**: 465.3 KB

**Rows**: 2,852

**Last Modified**: 2025-08-18T16:05:25.440171


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DeliveryInterval | None | int64 | Yes |
| AS_ECRS | None | float64 | Yes |
| AS_NSPIN | None | float64 | Yes |
| AS_REGDN | None | float64 | Yes |
| AS_REGUP | None | float64 | Yes |
| AS_RRS | None | float64 | Yes |
| DA_DC_E | None | float64 | Yes |
| DA_DC_L | None | float64 | Yes |
| DA_DC_N | None | float64 | Yes |
| DA_DC_R | None | float64 | Yes |
| DA_DC_S | None | float64 | Yes |
| DA_HB_BUSAVG | None | float64 | Yes |
| DA_HB_HOUSTON | None | float64 | Yes |
| DA_HB_HUBAVG | None | float64 | Yes |
| DA_HB_NORTH | None | float64 | Yes |
| DA_HB_PAN | None | float64 | Yes |
| DA_HB_SOUTH | None | float64 | Yes |
| DA_HB_WEST | None | float64 | Yes |
| DA_LZ_AEN | None | float64 | Yes |
| DA_LZ_CPS | None | float64 | Yes |
| DA_LZ_HOUSTON | None | float64 | Yes |
| DA_LZ_LCRA | None | float64 | Yes |
| DA_LZ_NORTH | None | float64 | Yes |
| DA_LZ_RAYBN | None | float64 | Yes |
| DA_LZ_SOUTH | None | float64 | Yes |
| DA_LZ_WEST | None | float64 | Yes |
| RT_DC_E | None | float64 | Yes |
| RT_DC_L | None | float64 | Yes |
| RT_DC_N | None | float64 | Yes |
| RT_DC_R | None | float64 | Yes |
| RT_DC_S | None | float64 | Yes |
| RT_HB_BUSAVG | None | float64 | Yes |
| RT_HB_HOUSTON | None | float64 | Yes |
| RT_HB_HUBAVG | None | float64 | Yes |
| RT_HB_NORTH | None | float64 | Yes |
| RT_HB_PAN | None | float64 | Yes |
| RT_HB_SOUTH | None | float64 | Yes |
| RT_HB_WEST | None | float64 | Yes |
| RT_LZ_AEN | None | float64 | Yes |
| RT_LZ_CPS | None | float64 | Yes |
| RT_LZ_HOUSTON | None | float64 | Yes |
| RT_LZ_LCRA | None | float64 | Yes |
| RT_LZ_NORTH | None | float64 | Yes |
| RT_LZ_RAYBN | None | float64 | Yes |
| RT_LZ_SOUTH | None | float64 | Yes |
| RT_LZ_WEST | None | float64 | Yes |


### DA_AS_RT_15min_combined_2023_09.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_RT_15min_combined/DA_AS_RT_15min_combined_2023_09.parquet`

**Size**: 416.6 KB

**Rows**: 2,760

**Last Modified**: 2025-08-18T16:05:25.445171


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DeliveryInterval | None | int64 | Yes |
| AS_ECRS | None | float64 | Yes |
| AS_NSPIN | None | float64 | Yes |
| AS_REGDN | None | float64 | Yes |
| AS_REGUP | None | float64 | Yes |
| AS_RRS | None | float64 | Yes |
| DA_DC_E | None | float64 | Yes |
| DA_DC_L | None | float64 | Yes |
| DA_DC_N | None | float64 | Yes |
| DA_DC_R | None | float64 | Yes |
| DA_DC_S | None | float64 | Yes |
| DA_HB_BUSAVG | None | float64 | Yes |
| DA_HB_HOUSTON | None | float64 | Yes |
| DA_HB_HUBAVG | None | float64 | Yes |
| DA_HB_NORTH | None | float64 | Yes |
| DA_HB_PAN | None | float64 | Yes |
| DA_HB_SOUTH | None | float64 | Yes |
| DA_HB_WEST | None | float64 | Yes |
| DA_LZ_AEN | None | float64 | Yes |
| DA_LZ_CPS | None | float64 | Yes |
| DA_LZ_HOUSTON | None | float64 | Yes |
| DA_LZ_LCRA | None | float64 | Yes |
| DA_LZ_NORTH | None | float64 | Yes |
| DA_LZ_RAYBN | None | float64 | Yes |
| DA_LZ_SOUTH | None | float64 | Yes |
| DA_LZ_WEST | None | float64 | Yes |
| RT_DC_E | None | float64 | Yes |
| RT_DC_L | None | float64 | Yes |
| RT_DC_N | None | float64 | Yes |
| RT_DC_R | None | float64 | Yes |
| RT_DC_S | None | float64 | Yes |
| RT_HB_BUSAVG | None | float64 | Yes |
| RT_HB_HOUSTON | None | float64 | Yes |
| RT_HB_HUBAVG | None | float64 | Yes |
| RT_HB_NORTH | None | float64 | Yes |
| RT_HB_PAN | None | float64 | Yes |
| RT_HB_SOUTH | None | float64 | Yes |
| RT_HB_WEST | None | float64 | Yes |
| RT_LZ_AEN | None | float64 | Yes |
| RT_LZ_CPS | None | float64 | Yes |
| RT_LZ_HOUSTON | None | float64 | Yes |
| RT_LZ_LCRA | None | float64 | Yes |
| RT_LZ_NORTH | None | float64 | Yes |
| RT_LZ_RAYBN | None | float64 | Yes |
| RT_LZ_SOUTH | None | float64 | Yes |
| RT_LZ_WEST | None | float64 | Yes |


### DA_AS_RT_15min_combined_2023_10.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_RT_15min_combined/DA_AS_RT_15min_combined_2023_10.parquet`

**Size**: 412.7 KB

**Rows**: 2,852

**Last Modified**: 2025-08-18T16:05:25.450171


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DeliveryInterval | None | int64 | Yes |
| AS_ECRS | None | float64 | Yes |
| AS_NSPIN | None | float64 | Yes |
| AS_REGDN | None | float64 | Yes |
| AS_REGUP | None | float64 | Yes |
| AS_RRS | None | float64 | Yes |
| DA_DC_E | None | float64 | Yes |
| DA_DC_L | None | float64 | Yes |
| DA_DC_N | None | float64 | Yes |
| DA_DC_R | None | float64 | Yes |
| DA_DC_S | None | float64 | Yes |
| DA_HB_BUSAVG | None | float64 | Yes |
| DA_HB_HOUSTON | None | float64 | Yes |
| DA_HB_HUBAVG | None | float64 | Yes |
| DA_HB_NORTH | None | float64 | Yes |
| DA_HB_PAN | None | float64 | Yes |
| DA_HB_SOUTH | None | float64 | Yes |
| DA_HB_WEST | None | float64 | Yes |
| DA_LZ_AEN | None | float64 | Yes |
| DA_LZ_CPS | None | float64 | Yes |
| DA_LZ_HOUSTON | None | float64 | Yes |
| DA_LZ_LCRA | None | float64 | Yes |
| DA_LZ_NORTH | None | float64 | Yes |
| DA_LZ_RAYBN | None | float64 | Yes |
| DA_LZ_SOUTH | None | float64 | Yes |
| DA_LZ_WEST | None | float64 | Yes |
| RT_DC_E | None | float64 | Yes |
| RT_DC_L | None | float64 | Yes |
| RT_DC_N | None | float64 | Yes |
| RT_DC_R | None | float64 | Yes |
| RT_DC_S | None | float64 | Yes |
| RT_HB_BUSAVG | None | float64 | Yes |
| RT_HB_HOUSTON | None | float64 | Yes |
| RT_HB_HUBAVG | None | float64 | Yes |
| RT_HB_NORTH | None | float64 | Yes |
| RT_HB_PAN | None | float64 | Yes |
| RT_HB_SOUTH | None | float64 | Yes |
| RT_HB_WEST | None | float64 | Yes |
| RT_LZ_AEN | None | float64 | Yes |
| RT_LZ_CPS | None | float64 | Yes |
| RT_LZ_HOUSTON | None | float64 | Yes |
| RT_LZ_LCRA | None | float64 | Yes |
| RT_LZ_NORTH | None | float64 | Yes |
| RT_LZ_RAYBN | None | float64 | Yes |
| RT_LZ_SOUTH | None | float64 | Yes |
| RT_LZ_WEST | None | float64 | Yes |


### DA_AS_RT_15min_combined_2023_11.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_RT_15min_combined/DA_AS_RT_15min_combined_2023_11.parquet`

**Size**: 409.6 KB

**Rows**: 2,760

**Last Modified**: 2025-08-18T16:05:25.455171


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DeliveryInterval | None | int64 | Yes |
| AS_ECRS | None | float64 | Yes |
| AS_NSPIN | None | float64 | Yes |
| AS_REGDN | None | float64 | Yes |
| AS_REGUP | None | float64 | Yes |
| AS_RRS | None | float64 | Yes |
| DA_DC_E | None | float64 | Yes |
| DA_DC_L | None | float64 | Yes |
| DA_DC_N | None | float64 | Yes |
| DA_DC_R | None | float64 | Yes |
| DA_DC_S | None | float64 | Yes |
| DA_HB_BUSAVG | None | float64 | Yes |
| DA_HB_HOUSTON | None | float64 | Yes |
| DA_HB_HUBAVG | None | float64 | Yes |
| DA_HB_NORTH | None | float64 | Yes |
| DA_HB_PAN | None | float64 | Yes |
| DA_HB_SOUTH | None | float64 | Yes |
| DA_HB_WEST | None | float64 | Yes |
| DA_LZ_AEN | None | float64 | Yes |
| DA_LZ_CPS | None | float64 | Yes |
| DA_LZ_HOUSTON | None | float64 | Yes |
| DA_LZ_LCRA | None | float64 | Yes |
| DA_LZ_NORTH | None | float64 | Yes |
| DA_LZ_RAYBN | None | float64 | Yes |
| DA_LZ_SOUTH | None | float64 | Yes |
| DA_LZ_WEST | None | float64 | Yes |
| RT_DC_E | None | float64 | Yes |
| RT_DC_L | None | float64 | Yes |
| RT_DC_N | None | float64 | Yes |
| RT_DC_R | None | float64 | Yes |
| RT_DC_S | None | float64 | Yes |
| RT_HB_BUSAVG | None | float64 | Yes |
| RT_HB_HOUSTON | None | float64 | Yes |
| RT_HB_HUBAVG | None | float64 | Yes |
| RT_HB_NORTH | None | float64 | Yes |
| RT_HB_PAN | None | float64 | Yes |
| RT_HB_SOUTH | None | float64 | Yes |
| RT_HB_WEST | None | float64 | Yes |
| RT_LZ_AEN | None | float64 | Yes |
| RT_LZ_CPS | None | float64 | Yes |
| RT_LZ_HOUSTON | None | float64 | Yes |
| RT_LZ_LCRA | None | float64 | Yes |
| RT_LZ_NORTH | None | float64 | Yes |
| RT_LZ_RAYBN | None | float64 | Yes |
| RT_LZ_SOUTH | None | float64 | Yes |
| RT_LZ_WEST | None | float64 | Yes |


### DA_AS_RT_15min_combined_2023_12.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_RT_15min_combined/DA_AS_RT_15min_combined_2023_12.parquet`

**Size**: 402.3 KB

**Rows**: 2,852

**Last Modified**: 2025-08-18T16:05:25.460171


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DeliveryInterval | None | int64 | Yes |
| AS_ECRS | None | float64 | Yes |
| AS_NSPIN | None | float64 | Yes |
| AS_REGDN | None | float64 | Yes |
| AS_REGUP | None | float64 | Yes |
| AS_RRS | None | float64 | Yes |
| DA_DC_E | None | float64 | Yes |
| DA_DC_L | None | float64 | Yes |
| DA_DC_N | None | float64 | Yes |
| DA_DC_R | None | float64 | Yes |
| DA_DC_S | None | float64 | Yes |
| DA_HB_BUSAVG | None | float64 | Yes |
| DA_HB_HOUSTON | None | float64 | Yes |
| DA_HB_HUBAVG | None | float64 | Yes |
| DA_HB_NORTH | None | float64 | Yes |
| DA_HB_PAN | None | float64 | Yes |
| DA_HB_SOUTH | None | float64 | Yes |
| DA_HB_WEST | None | float64 | Yes |
| DA_LZ_AEN | None | float64 | Yes |
| DA_LZ_CPS | None | float64 | Yes |
| DA_LZ_HOUSTON | None | float64 | Yes |
| DA_LZ_LCRA | None | float64 | Yes |
| DA_LZ_NORTH | None | float64 | Yes |
| DA_LZ_RAYBN | None | float64 | Yes |
| DA_LZ_SOUTH | None | float64 | Yes |
| DA_LZ_WEST | None | float64 | Yes |
| RT_DC_E | None | float64 | Yes |
| RT_DC_L | None | float64 | Yes |
| RT_DC_N | None | float64 | Yes |
| RT_DC_R | None | float64 | Yes |
| RT_DC_S | None | float64 | Yes |
| RT_HB_BUSAVG | None | float64 | Yes |
| RT_HB_HOUSTON | None | float64 | Yes |
| RT_HB_HUBAVG | None | float64 | Yes |
| RT_HB_NORTH | None | float64 | Yes |
| RT_HB_PAN | None | float64 | Yes |
| RT_HB_SOUTH | None | float64 | Yes |
| RT_HB_WEST | None | float64 | Yes |
| RT_LZ_AEN | None | float64 | Yes |
| RT_LZ_CPS | None | float64 | Yes |
| RT_LZ_HOUSTON | None | float64 | Yes |
| RT_LZ_LCRA | None | float64 | Yes |
| RT_LZ_NORTH | None | float64 | Yes |
| RT_LZ_RAYBN | None | float64 | Yes |
| RT_LZ_SOUTH | None | float64 | Yes |
| RT_LZ_WEST | None | float64 | Yes |


### DA_AS_RT_15min_combined_2024_01.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_RT_15min_combined/DA_AS_RT_15min_combined_2024_01.parquet`

**Size**: 363.4 KB

**Rows**: 2,852

**Last Modified**: 2025-08-18T16:05:25.540175


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DeliveryInterval | None | int64 | Yes |
| AS_ECRS | None | float64 | Yes |
| AS_NSPIN | None | float64 | Yes |
| AS_REGDN | None | float64 | Yes |
| AS_REGUP | None | float64 | Yes |
| AS_RRS | None | float64 | Yes |
| DA_DC_E | None | float64 | Yes |
| DA_DC_L | None | float64 | Yes |
| DA_DC_N | None | float64 | Yes |
| DA_DC_R | None | float64 | Yes |
| DA_DC_S | None | float64 | Yes |
| DA_HB_BUSAVG | None | float64 | Yes |
| DA_HB_HOUSTON | None | float64 | Yes |
| DA_HB_HUBAVG | None | float64 | Yes |
| DA_HB_NORTH | None | float64 | Yes |
| DA_HB_PAN | None | float64 | Yes |
| DA_HB_SOUTH | None | float64 | Yes |
| DA_HB_WEST | None | float64 | Yes |
| DA_LZ_AEN | None | float64 | Yes |
| DA_LZ_CPS | None | float64 | Yes |
| DA_LZ_HOUSTON | None | float64 | Yes |
| DA_LZ_LCRA | None | float64 | Yes |
| DA_LZ_NORTH | None | float64 | Yes |
| DA_LZ_RAYBN | None | float64 | Yes |
| DA_LZ_SOUTH | None | float64 | Yes |
| DA_LZ_WEST | None | float64 | Yes |
| RT_DC_E | None | float64 | Yes |
| RT_DC_L | None | float64 | Yes |
| RT_DC_N | None | float64 | Yes |
| RT_DC_R | None | float64 | Yes |
| RT_DC_S | None | float64 | Yes |
| RT_HB_BUSAVG | None | float64 | Yes |
| RT_HB_HOUSTON | None | float64 | Yes |
| RT_HB_HUBAVG | None | float64 | Yes |
| RT_HB_NORTH | None | float64 | Yes |
| RT_HB_PAN | None | float64 | Yes |
| RT_HB_SOUTH | None | float64 | Yes |
| RT_HB_WEST | None | float64 | Yes |
| RT_LZ_AEN | None | float64 | Yes |
| RT_LZ_CPS | None | float64 | Yes |
| RT_LZ_HOUSTON | None | float64 | Yes |
| RT_LZ_LCRA | None | float64 | Yes |
| RT_LZ_NORTH | None | float64 | Yes |
| RT_LZ_RAYBN | None | float64 | Yes |
| RT_LZ_SOUTH | None | float64 | Yes |
| RT_LZ_WEST | None | float64 | Yes |


### DA_AS_RT_15min_combined_2024_02.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_RT_15min_combined/DA_AS_RT_15min_combined_2024_02.parquet`

**Size**: 323.7 KB

**Rows**: 2,668

**Last Modified**: 2025-08-18T16:05:25.545176


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DeliveryInterval | None | int64 | Yes |
| AS_ECRS | None | float64 | Yes |
| AS_NSPIN | None | float64 | Yes |
| AS_REGDN | None | float64 | Yes |
| AS_REGUP | None | float64 | Yes |
| AS_RRS | None | float64 | Yes |
| DA_DC_E | None | float64 | Yes |
| DA_DC_L | None | float64 | Yes |
| DA_DC_N | None | float64 | Yes |
| DA_DC_R | None | float64 | Yes |
| DA_DC_S | None | float64 | Yes |
| DA_HB_BUSAVG | None | float64 | Yes |
| DA_HB_HOUSTON | None | float64 | Yes |
| DA_HB_HUBAVG | None | float64 | Yes |
| DA_HB_NORTH | None | float64 | Yes |
| DA_HB_PAN | None | float64 | Yes |
| DA_HB_SOUTH | None | float64 | Yes |
| DA_HB_WEST | None | float64 | Yes |
| DA_LZ_AEN | None | float64 | Yes |
| DA_LZ_CPS | None | float64 | Yes |
| DA_LZ_HOUSTON | None | float64 | Yes |
| DA_LZ_LCRA | None | float64 | Yes |
| DA_LZ_NORTH | None | float64 | Yes |
| DA_LZ_RAYBN | None | float64 | Yes |
| DA_LZ_SOUTH | None | float64 | Yes |
| DA_LZ_WEST | None | float64 | Yes |
| RT_DC_E | None | float64 | Yes |
| RT_DC_L | None | float64 | Yes |
| RT_DC_N | None | float64 | Yes |
| RT_DC_R | None | float64 | Yes |
| RT_DC_S | None | float64 | Yes |
| RT_HB_BUSAVG | None | float64 | Yes |
| RT_HB_HOUSTON | None | float64 | Yes |
| RT_HB_HUBAVG | None | float64 | Yes |
| RT_HB_NORTH | None | float64 | Yes |
| RT_HB_PAN | None | float64 | Yes |
| RT_HB_SOUTH | None | float64 | Yes |
| RT_HB_WEST | None | float64 | Yes |
| RT_LZ_AEN | None | float64 | Yes |
| RT_LZ_CPS | None | float64 | Yes |
| RT_LZ_HOUSTON | None | float64 | Yes |
| RT_LZ_LCRA | None | float64 | Yes |
| RT_LZ_NORTH | None | float64 | Yes |
| RT_LZ_RAYBN | None | float64 | Yes |
| RT_LZ_SOUTH | None | float64 | Yes |
| RT_LZ_WEST | None | float64 | Yes |


### DA_AS_RT_15min_combined_2024_03.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_RT_15min_combined/DA_AS_RT_15min_combined_2024_03.parquet`

**Size**: 357.8 KB

**Rows**: 2,848

**Last Modified**: 2025-08-18T16:05:25.549176


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DeliveryInterval | None | int64 | Yes |
| AS_ECRS | None | float64 | Yes |
| AS_NSPIN | None | float64 | Yes |
| AS_REGDN | None | float64 | Yes |
| AS_REGUP | None | float64 | Yes |
| AS_RRS | None | float64 | Yes |
| DA_DC_E | None | float64 | Yes |
| DA_DC_L | None | float64 | Yes |
| DA_DC_N | None | float64 | Yes |
| DA_DC_R | None | float64 | Yes |
| DA_DC_S | None | float64 | Yes |
| DA_HB_BUSAVG | None | float64 | Yes |
| DA_HB_HOUSTON | None | float64 | Yes |
| DA_HB_HUBAVG | None | float64 | Yes |
| DA_HB_NORTH | None | float64 | Yes |
| DA_HB_PAN | None | float64 | Yes |
| DA_HB_SOUTH | None | float64 | Yes |
| DA_HB_WEST | None | float64 | Yes |
| DA_LZ_AEN | None | float64 | Yes |
| DA_LZ_CPS | None | float64 | Yes |
| DA_LZ_HOUSTON | None | float64 | Yes |
| DA_LZ_LCRA | None | float64 | Yes |
| DA_LZ_NORTH | None | float64 | Yes |
| DA_LZ_RAYBN | None | float64 | Yes |
| DA_LZ_SOUTH | None | float64 | Yes |
| DA_LZ_WEST | None | float64 | Yes |
| RT_DC_E | None | float64 | Yes |
| RT_DC_L | None | float64 | Yes |
| RT_DC_N | None | float64 | Yes |
| RT_DC_R | None | float64 | Yes |
| RT_DC_S | None | float64 | Yes |
| RT_HB_BUSAVG | None | float64 | Yes |
| RT_HB_HOUSTON | None | float64 | Yes |
| RT_HB_HUBAVG | None | float64 | Yes |
| RT_HB_NORTH | None | float64 | Yes |
| RT_HB_PAN | None | float64 | Yes |
| RT_HB_SOUTH | None | float64 | Yes |
| RT_HB_WEST | None | float64 | Yes |
| RT_LZ_AEN | None | float64 | Yes |
| RT_LZ_CPS | None | float64 | Yes |
| RT_LZ_HOUSTON | None | float64 | Yes |
| RT_LZ_LCRA | None | float64 | Yes |
| RT_LZ_NORTH | None | float64 | Yes |
| RT_LZ_RAYBN | None | float64 | Yes |
| RT_LZ_SOUTH | None | float64 | Yes |
| RT_LZ_WEST | None | float64 | Yes |


### DA_AS_RT_15min_combined_2024_04.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_RT_15min_combined/DA_AS_RT_15min_combined_2024_04.parquet`

**Size**: 342.9 KB

**Rows**: 2,760

**Last Modified**: 2025-08-18T16:05:25.554176


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DeliveryInterval | None | int64 | Yes |
| AS_ECRS | None | float64 | Yes |
| AS_NSPIN | None | float64 | Yes |
| AS_REGDN | None | float64 | Yes |
| AS_REGUP | None | float64 | Yes |
| AS_RRS | None | float64 | Yes |
| DA_DC_E | None | float64 | Yes |
| DA_DC_L | None | float64 | Yes |
| DA_DC_N | None | float64 | Yes |
| DA_DC_R | None | float64 | Yes |
| DA_DC_S | None | float64 | Yes |
| DA_HB_BUSAVG | None | float64 | Yes |
| DA_HB_HOUSTON | None | float64 | Yes |
| DA_HB_HUBAVG | None | float64 | Yes |
| DA_HB_NORTH | None | float64 | Yes |
| DA_HB_PAN | None | float64 | Yes |
| DA_HB_SOUTH | None | float64 | Yes |
| DA_HB_WEST | None | float64 | Yes |
| DA_LZ_AEN | None | float64 | Yes |
| DA_LZ_CPS | None | float64 | Yes |
| DA_LZ_HOUSTON | None | float64 | Yes |
| DA_LZ_LCRA | None | float64 | Yes |
| DA_LZ_NORTH | None | float64 | Yes |
| DA_LZ_RAYBN | None | float64 | Yes |
| DA_LZ_SOUTH | None | float64 | Yes |
| DA_LZ_WEST | None | float64 | Yes |
| RT_DC_E | None | float64 | Yes |
| RT_DC_L | None | float64 | Yes |
| RT_DC_N | None | float64 | Yes |
| RT_DC_R | None | float64 | Yes |
| RT_DC_S | None | float64 | Yes |
| RT_HB_BUSAVG | None | float64 | Yes |
| RT_HB_HOUSTON | None | float64 | Yes |
| RT_HB_HUBAVG | None | float64 | Yes |
| RT_HB_NORTH | None | float64 | Yes |
| RT_HB_PAN | None | float64 | Yes |
| RT_HB_SOUTH | None | float64 | Yes |
| RT_HB_WEST | None | float64 | Yes |
| RT_LZ_AEN | None | float64 | Yes |
| RT_LZ_CPS | None | float64 | Yes |
| RT_LZ_HOUSTON | None | float64 | Yes |
| RT_LZ_LCRA | None | float64 | Yes |
| RT_LZ_NORTH | None | float64 | Yes |
| RT_LZ_RAYBN | None | float64 | Yes |
| RT_LZ_SOUTH | None | float64 | Yes |
| RT_LZ_WEST | None | float64 | Yes |


### DA_AS_RT_15min_combined_2024_05.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_RT_15min_combined/DA_AS_RT_15min_combined_2024_05.parquet`

**Size**: 357.8 KB

**Rows**: 2,852

**Last Modified**: 2025-08-18T16:05:25.558176


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DeliveryInterval | None | int64 | Yes |
| AS_ECRS | None | float64 | Yes |
| AS_NSPIN | None | float64 | Yes |
| AS_REGDN | None | float64 | Yes |
| AS_REGUP | None | float64 | Yes |
| AS_RRS | None | float64 | Yes |
| DA_DC_E | None | float64 | Yes |
| DA_DC_L | None | float64 | Yes |
| DA_DC_N | None | float64 | Yes |
| DA_DC_R | None | float64 | Yes |
| DA_DC_S | None | float64 | Yes |
| DA_HB_BUSAVG | None | float64 | Yes |
| DA_HB_HOUSTON | None | float64 | Yes |
| DA_HB_HUBAVG | None | float64 | Yes |
| DA_HB_NORTH | None | float64 | Yes |
| DA_HB_PAN | None | float64 | Yes |
| DA_HB_SOUTH | None | float64 | Yes |
| DA_HB_WEST | None | float64 | Yes |
| DA_LZ_AEN | None | float64 | Yes |
| DA_LZ_CPS | None | float64 | Yes |
| DA_LZ_HOUSTON | None | float64 | Yes |
| DA_LZ_LCRA | None | float64 | Yes |
| DA_LZ_NORTH | None | float64 | Yes |
| DA_LZ_RAYBN | None | float64 | Yes |
| DA_LZ_SOUTH | None | float64 | Yes |
| DA_LZ_WEST | None | float64 | Yes |
| RT_DC_E | None | float64 | Yes |
| RT_DC_L | None | float64 | Yes |
| RT_DC_N | None | float64 | Yes |
| RT_DC_R | None | float64 | Yes |
| RT_DC_S | None | float64 | Yes |
| RT_HB_BUSAVG | None | float64 | Yes |
| RT_HB_HOUSTON | None | float64 | Yes |
| RT_HB_HUBAVG | None | float64 | Yes |
| RT_HB_NORTH | None | float64 | Yes |
| RT_HB_PAN | None | float64 | Yes |
| RT_HB_SOUTH | None | float64 | Yes |
| RT_HB_WEST | None | float64 | Yes |
| RT_LZ_AEN | None | float64 | Yes |
| RT_LZ_CPS | None | float64 | Yes |
| RT_LZ_HOUSTON | None | float64 | Yes |
| RT_LZ_LCRA | None | float64 | Yes |
| RT_LZ_NORTH | None | float64 | Yes |
| RT_LZ_RAYBN | None | float64 | Yes |
| RT_LZ_SOUTH | None | float64 | Yes |
| RT_LZ_WEST | None | float64 | Yes |


### DA_AS_RT_15min_combined_2024_06.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_RT_15min_combined/DA_AS_RT_15min_combined_2024_06.parquet`

**Size**: 346.7 KB

**Rows**: 2,760

**Last Modified**: 2025-08-18T16:05:25.563176


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DeliveryInterval | None | int64 | Yes |
| AS_ECRS | None | float64 | Yes |
| AS_NSPIN | None | float64 | Yes |
| AS_REGDN | None | float64 | Yes |
| AS_REGUP | None | float64 | Yes |
| AS_RRS | None | float64 | Yes |
| DA_DC_E | None | float64 | Yes |
| DA_DC_L | None | float64 | Yes |
| DA_DC_N | None | float64 | Yes |
| DA_DC_R | None | float64 | Yes |
| DA_DC_S | None | float64 | Yes |
| DA_HB_BUSAVG | None | float64 | Yes |
| DA_HB_HOUSTON | None | float64 | Yes |
| DA_HB_HUBAVG | None | float64 | Yes |
| DA_HB_NORTH | None | float64 | Yes |
| DA_HB_PAN | None | float64 | Yes |
| DA_HB_SOUTH | None | float64 | Yes |
| DA_HB_WEST | None | float64 | Yes |
| DA_LZ_AEN | None | float64 | Yes |
| DA_LZ_CPS | None | float64 | Yes |
| DA_LZ_HOUSTON | None | float64 | Yes |
| DA_LZ_LCRA | None | float64 | Yes |
| DA_LZ_NORTH | None | float64 | Yes |
| DA_LZ_RAYBN | None | float64 | Yes |
| DA_LZ_SOUTH | None | float64 | Yes |
| DA_LZ_WEST | None | float64 | Yes |
| RT_DC_E | None | float64 | Yes |
| RT_DC_L | None | float64 | Yes |
| RT_DC_N | None | float64 | Yes |
| RT_DC_R | None | float64 | Yes |
| RT_DC_S | None | float64 | Yes |
| RT_HB_BUSAVG | None | float64 | Yes |
| RT_HB_HOUSTON | None | float64 | Yes |
| RT_HB_HUBAVG | None | float64 | Yes |
| RT_HB_NORTH | None | float64 | Yes |
| RT_HB_PAN | None | float64 | Yes |
| RT_HB_SOUTH | None | float64 | Yes |
| RT_HB_WEST | None | float64 | Yes |
| RT_LZ_AEN | None | float64 | Yes |
| RT_LZ_CPS | None | float64 | Yes |
| RT_LZ_HOUSTON | None | float64 | Yes |
| RT_LZ_LCRA | None | float64 | Yes |
| RT_LZ_NORTH | None | float64 | Yes |
| RT_LZ_RAYBN | None | float64 | Yes |
| RT_LZ_SOUTH | None | float64 | Yes |
| RT_LZ_WEST | None | float64 | Yes |


### DA_AS_RT_15min_combined_2024_07.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_RT_15min_combined/DA_AS_RT_15min_combined_2024_07.parquet`

**Size**: 329.3 KB

**Rows**: 2,852

**Last Modified**: 2025-08-18T16:05:25.567177


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DeliveryInterval | None | int64 | Yes |
| AS_ECRS | None | float64 | Yes |
| AS_NSPIN | None | float64 | Yes |
| AS_REGDN | None | float64 | Yes |
| AS_REGUP | None | float64 | Yes |
| AS_RRS | None | float64 | Yes |
| DA_DC_E | None | float64 | Yes |
| DA_DC_L | None | float64 | Yes |
| DA_DC_N | None | float64 | Yes |
| DA_DC_R | None | float64 | Yes |
| DA_DC_S | None | float64 | Yes |
| DA_HB_BUSAVG | None | float64 | Yes |
| DA_HB_HOUSTON | None | float64 | Yes |
| DA_HB_HUBAVG | None | float64 | Yes |
| DA_HB_NORTH | None | float64 | Yes |
| DA_HB_PAN | None | float64 | Yes |
| DA_HB_SOUTH | None | float64 | Yes |
| DA_HB_WEST | None | float64 | Yes |
| DA_LZ_AEN | None | float64 | Yes |
| DA_LZ_CPS | None | float64 | Yes |
| DA_LZ_HOUSTON | None | float64 | Yes |
| DA_LZ_LCRA | None | float64 | Yes |
| DA_LZ_NORTH | None | float64 | Yes |
| DA_LZ_RAYBN | None | float64 | Yes |
| DA_LZ_SOUTH | None | float64 | Yes |
| DA_LZ_WEST | None | float64 | Yes |
| RT_DC_E | None | float64 | Yes |
| RT_DC_L | None | float64 | Yes |
| RT_DC_N | None | float64 | Yes |
| RT_DC_R | None | float64 | Yes |
| RT_DC_S | None | float64 | Yes |
| RT_HB_BUSAVG | None | float64 | Yes |
| RT_HB_HOUSTON | None | float64 | Yes |
| RT_HB_HUBAVG | None | float64 | Yes |
| RT_HB_NORTH | None | float64 | Yes |
| RT_HB_PAN | None | float64 | Yes |
| RT_HB_SOUTH | None | float64 | Yes |
| RT_HB_WEST | None | float64 | Yes |
| RT_LZ_AEN | None | float64 | Yes |
| RT_LZ_CPS | None | float64 | Yes |
| RT_LZ_HOUSTON | None | float64 | Yes |
| RT_LZ_LCRA | None | float64 | Yes |
| RT_LZ_NORTH | None | float64 | Yes |
| RT_LZ_RAYBN | None | float64 | Yes |
| RT_LZ_SOUTH | None | float64 | Yes |
| RT_LZ_WEST | None | float64 | Yes |


### DA_AS_RT_15min_combined_2024_08.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_RT_15min_combined/DA_AS_RT_15min_combined_2024_08.parquet`

**Size**: 338.6 KB

**Rows**: 2,852

**Last Modified**: 2025-08-18T16:05:25.572177


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DeliveryInterval | None | int64 | Yes |
| AS_ECRS | None | float64 | Yes |
| AS_NSPIN | None | float64 | Yes |
| AS_REGDN | None | float64 | Yes |
| AS_REGUP | None | float64 | Yes |
| AS_RRS | None | float64 | Yes |
| DA_DC_E | None | float64 | Yes |
| DA_DC_L | None | float64 | Yes |
| DA_DC_N | None | float64 | Yes |
| DA_DC_R | None | float64 | Yes |
| DA_DC_S | None | float64 | Yes |
| DA_HB_BUSAVG | None | float64 | Yes |
| DA_HB_HOUSTON | None | float64 | Yes |
| DA_HB_HUBAVG | None | float64 | Yes |
| DA_HB_NORTH | None | float64 | Yes |
| DA_HB_PAN | None | float64 | Yes |
| DA_HB_SOUTH | None | float64 | Yes |
| DA_HB_WEST | None | float64 | Yes |
| DA_LZ_AEN | None | float64 | Yes |
| DA_LZ_CPS | None | float64 | Yes |
| DA_LZ_HOUSTON | None | float64 | Yes |
| DA_LZ_LCRA | None | float64 | Yes |
| DA_LZ_NORTH | None | float64 | Yes |
| DA_LZ_RAYBN | None | float64 | Yes |
| DA_LZ_SOUTH | None | float64 | Yes |
| DA_LZ_WEST | None | float64 | Yes |
| RT_DC_E | None | float64 | Yes |
| RT_DC_L | None | float64 | Yes |
| RT_DC_N | None | float64 | Yes |
| RT_DC_R | None | float64 | Yes |
| RT_DC_S | None | float64 | Yes |
| RT_HB_BUSAVG | None | float64 | Yes |
| RT_HB_HOUSTON | None | float64 | Yes |
| RT_HB_HUBAVG | None | float64 | Yes |
| RT_HB_NORTH | None | float64 | Yes |
| RT_HB_PAN | None | float64 | Yes |
| RT_HB_SOUTH | None | float64 | Yes |
| RT_HB_WEST | None | float64 | Yes |
| RT_LZ_AEN | None | float64 | Yes |
| RT_LZ_CPS | None | float64 | Yes |
| RT_LZ_HOUSTON | None | float64 | Yes |
| RT_LZ_LCRA | None | float64 | Yes |
| RT_LZ_NORTH | None | float64 | Yes |
| RT_LZ_RAYBN | None | float64 | Yes |
| RT_LZ_SOUTH | None | float64 | Yes |
| RT_LZ_WEST | None | float64 | Yes |


### DA_AS_RT_15min_combined_2024_09.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_RT_15min_combined/DA_AS_RT_15min_combined_2024_09.parquet`

**Size**: 321.3 KB

**Rows**: 2,760

**Last Modified**: 2025-08-18T16:05:25.576177


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DeliveryInterval | None | int64 | Yes |
| AS_ECRS | None | float64 | Yes |
| AS_NSPIN | None | float64 | Yes |
| AS_REGDN | None | float64 | Yes |
| AS_REGUP | None | float64 | Yes |
| AS_RRS | None | float64 | Yes |
| DA_DC_E | None | float64 | Yes |
| DA_DC_L | None | float64 | Yes |
| DA_DC_N | None | float64 | Yes |
| DA_DC_R | None | float64 | Yes |
| DA_DC_S | None | float64 | Yes |
| DA_HB_BUSAVG | None | float64 | Yes |
| DA_HB_HOUSTON | None | float64 | Yes |
| DA_HB_HUBAVG | None | float64 | Yes |
| DA_HB_NORTH | None | float64 | Yes |
| DA_HB_PAN | None | float64 | Yes |
| DA_HB_SOUTH | None | float64 | Yes |
| DA_HB_WEST | None | float64 | Yes |
| DA_LZ_AEN | None | float64 | Yes |
| DA_LZ_CPS | None | float64 | Yes |
| DA_LZ_HOUSTON | None | float64 | Yes |
| DA_LZ_LCRA | None | float64 | Yes |
| DA_LZ_NORTH | None | float64 | Yes |
| DA_LZ_RAYBN | None | float64 | Yes |
| DA_LZ_SOUTH | None | float64 | Yes |
| DA_LZ_WEST | None | float64 | Yes |
| RT_DC_E | None | float64 | Yes |
| RT_DC_L | None | float64 | Yes |
| RT_DC_N | None | float64 | Yes |
| RT_DC_R | None | float64 | Yes |
| RT_DC_S | None | float64 | Yes |
| RT_HB_BUSAVG | None | float64 | Yes |
| RT_HB_HOUSTON | None | float64 | Yes |
| RT_HB_HUBAVG | None | float64 | Yes |
| RT_HB_NORTH | None | float64 | Yes |
| RT_HB_PAN | None | float64 | Yes |
| RT_HB_SOUTH | None | float64 | Yes |
| RT_HB_WEST | None | float64 | Yes |
| RT_LZ_AEN | None | float64 | Yes |
| RT_LZ_CPS | None | float64 | Yes |
| RT_LZ_HOUSTON | None | float64 | Yes |
| RT_LZ_LCRA | None | float64 | Yes |
| RT_LZ_NORTH | None | float64 | Yes |
| RT_LZ_RAYBN | None | float64 | Yes |
| RT_LZ_SOUTH | None | float64 | Yes |
| RT_LZ_WEST | None | float64 | Yes |


### DA_AS_RT_15min_combined_2024_10.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_RT_15min_combined/DA_AS_RT_15min_combined_2024_10.parquet`

**Size**: 345.4 KB

**Rows**: 2,852

**Last Modified**: 2025-08-18T16:05:25.581177


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DeliveryInterval | None | int64 | Yes |
| AS_ECRS | None | float64 | Yes |
| AS_NSPIN | None | float64 | Yes |
| AS_REGDN | None | float64 | Yes |
| AS_REGUP | None | float64 | Yes |
| AS_RRS | None | float64 | Yes |
| DA_DC_E | None | float64 | Yes |
| DA_DC_L | None | float64 | Yes |
| DA_DC_N | None | float64 | Yes |
| DA_DC_R | None | float64 | Yes |
| DA_DC_S | None | float64 | Yes |
| DA_HB_BUSAVG | None | float64 | Yes |
| DA_HB_HOUSTON | None | float64 | Yes |
| DA_HB_HUBAVG | None | float64 | Yes |
| DA_HB_NORTH | None | float64 | Yes |
| DA_HB_PAN | None | float64 | Yes |
| DA_HB_SOUTH | None | float64 | Yes |
| DA_HB_WEST | None | float64 | Yes |
| DA_LZ_AEN | None | float64 | Yes |
| DA_LZ_CPS | None | float64 | Yes |
| DA_LZ_HOUSTON | None | float64 | Yes |
| DA_LZ_LCRA | None | float64 | Yes |
| DA_LZ_NORTH | None | float64 | Yes |
| DA_LZ_RAYBN | None | float64 | Yes |
| DA_LZ_SOUTH | None | float64 | Yes |
| DA_LZ_WEST | None | float64 | Yes |
| RT_DC_E | None | float64 | Yes |
| RT_DC_L | None | float64 | Yes |
| RT_DC_N | None | float64 | Yes |
| RT_DC_R | None | float64 | Yes |
| RT_DC_S | None | float64 | Yes |
| RT_HB_BUSAVG | None | float64 | Yes |
| RT_HB_HOUSTON | None | float64 | Yes |
| RT_HB_HUBAVG | None | float64 | Yes |
| RT_HB_NORTH | None | float64 | Yes |
| RT_HB_PAN | None | float64 | Yes |
| RT_HB_SOUTH | None | float64 | Yes |
| RT_HB_WEST | None | float64 | Yes |
| RT_LZ_AEN | None | float64 | Yes |
| RT_LZ_CPS | None | float64 | Yes |
| RT_LZ_HOUSTON | None | float64 | Yes |
| RT_LZ_LCRA | None | float64 | Yes |
| RT_LZ_NORTH | None | float64 | Yes |
| RT_LZ_RAYBN | None | float64 | Yes |
| RT_LZ_SOUTH | None | float64 | Yes |
| RT_LZ_WEST | None | float64 | Yes |


### DA_AS_RT_15min_combined_2024_11.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_RT_15min_combined/DA_AS_RT_15min_combined_2024_11.parquet`

**Size**: 347.9 KB

**Rows**: 2,760

**Last Modified**: 2025-08-18T16:05:25.585177


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DeliveryInterval | None | int64 | Yes |
| AS_ECRS | None | float64 | Yes |
| AS_NSPIN | None | float64 | Yes |
| AS_REGDN | None | float64 | Yes |
| AS_REGUP | None | float64 | Yes |
| AS_RRS | None | float64 | Yes |
| DA_DC_E | None | float64 | Yes |
| DA_DC_L | None | float64 | Yes |
| DA_DC_N | None | float64 | Yes |
| DA_DC_R | None | float64 | Yes |
| DA_DC_S | None | float64 | Yes |
| DA_HB_BUSAVG | None | float64 | Yes |
| DA_HB_HOUSTON | None | float64 | Yes |
| DA_HB_HUBAVG | None | float64 | Yes |
| DA_HB_NORTH | None | float64 | Yes |
| DA_HB_PAN | None | float64 | Yes |
| DA_HB_SOUTH | None | float64 | Yes |
| DA_HB_WEST | None | float64 | Yes |
| DA_LZ_AEN | None | float64 | Yes |
| DA_LZ_CPS | None | float64 | Yes |
| DA_LZ_HOUSTON | None | float64 | Yes |
| DA_LZ_LCRA | None | float64 | Yes |
| DA_LZ_NORTH | None | float64 | Yes |
| DA_LZ_RAYBN | None | float64 | Yes |
| DA_LZ_SOUTH | None | float64 | Yes |
| DA_LZ_WEST | None | float64 | Yes |
| RT_DC_E | None | float64 | Yes |
| RT_DC_L | None | float64 | Yes |
| RT_DC_N | None | float64 | Yes |
| RT_DC_R | None | float64 | Yes |
| RT_DC_S | None | float64 | Yes |
| RT_HB_BUSAVG | None | float64 | Yes |
| RT_HB_HOUSTON | None | float64 | Yes |
| RT_HB_HUBAVG | None | float64 | Yes |
| RT_HB_NORTH | None | float64 | Yes |
| RT_HB_PAN | None | float64 | Yes |
| RT_HB_SOUTH | None | float64 | Yes |
| RT_HB_WEST | None | float64 | Yes |
| RT_LZ_AEN | None | float64 | Yes |
| RT_LZ_CPS | None | float64 | Yes |
| RT_LZ_HOUSTON | None | float64 | Yes |
| RT_LZ_LCRA | None | float64 | Yes |
| RT_LZ_NORTH | None | float64 | Yes |
| RT_LZ_RAYBN | None | float64 | Yes |
| RT_LZ_SOUTH | None | float64 | Yes |
| RT_LZ_WEST | None | float64 | Yes |


### DA_AS_RT_15min_combined_2024_12.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_RT_15min_combined/DA_AS_RT_15min_combined_2024_12.parquet`

**Size**: 349.9 KB

**Rows**: 2,852

**Last Modified**: 2025-08-18T16:05:25.590178


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DeliveryInterval | None | int64 | Yes |
| AS_ECRS | None | float64 | Yes |
| AS_NSPIN | None | float64 | Yes |
| AS_REGDN | None | float64 | Yes |
| AS_REGUP | None | float64 | Yes |
| AS_RRS | None | float64 | Yes |
| DA_DC_E | None | float64 | Yes |
| DA_DC_L | None | float64 | Yes |
| DA_DC_N | None | float64 | Yes |
| DA_DC_R | None | float64 | Yes |
| DA_DC_S | None | float64 | Yes |
| DA_HB_BUSAVG | None | float64 | Yes |
| DA_HB_HOUSTON | None | float64 | Yes |
| DA_HB_HUBAVG | None | float64 | Yes |
| DA_HB_NORTH | None | float64 | Yes |
| DA_HB_PAN | None | float64 | Yes |
| DA_HB_SOUTH | None | float64 | Yes |
| DA_HB_WEST | None | float64 | Yes |
| DA_LZ_AEN | None | float64 | Yes |
| DA_LZ_CPS | None | float64 | Yes |
| DA_LZ_HOUSTON | None | float64 | Yes |
| DA_LZ_LCRA | None | float64 | Yes |
| DA_LZ_NORTH | None | float64 | Yes |
| DA_LZ_RAYBN | None | float64 | Yes |
| DA_LZ_SOUTH | None | float64 | Yes |
| DA_LZ_WEST | None | float64 | Yes |
| RT_DC_E | None | float64 | Yes |
| RT_DC_L | None | float64 | Yes |
| RT_DC_N | None | float64 | Yes |
| RT_DC_R | None | float64 | Yes |
| RT_DC_S | None | float64 | Yes |
| RT_HB_BUSAVG | None | float64 | Yes |
| RT_HB_HOUSTON | None | float64 | Yes |
| RT_HB_HUBAVG | None | float64 | Yes |
| RT_HB_NORTH | None | float64 | Yes |
| RT_HB_PAN | None | float64 | Yes |
| RT_HB_SOUTH | None | float64 | Yes |
| RT_HB_WEST | None | float64 | Yes |
| RT_LZ_AEN | None | float64 | Yes |
| RT_LZ_CPS | None | float64 | Yes |
| RT_LZ_HOUSTON | None | float64 | Yes |
| RT_LZ_LCRA | None | float64 | Yes |
| RT_LZ_NORTH | None | float64 | Yes |
| RT_LZ_RAYBN | None | float64 | Yes |
| RT_LZ_SOUTH | None | float64 | Yes |
| RT_LZ_WEST | None | float64 | Yes |


### DA_AS_RT_5min_combined_2023_01.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_RT_5min_combined/DA_AS_RT_5min_combined_2023_01.parquet`

**Size**: 365.7 KB

**Rows**: 2,852

**Last Modified**: 2025-08-17T23:54:34.871156


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DeliveryInterval | None | int64 | Yes |
| AS_ECRS | None | float64 | Yes |
| AS_NSPIN | None | float64 | Yes |
| AS_REGDN | None | float64 | Yes |
| AS_REGUP | None | float64 | Yes |
| AS_RRS | None | float64 | Yes |
| DA_DC_E | None | float64 | Yes |
| DA_DC_L | None | float64 | Yes |
| DA_DC_N | None | float64 | Yes |
| DA_DC_R | None | float64 | Yes |
| DA_DC_S | None | float64 | Yes |
| DA_HB_BUSAVG | None | float64 | Yes |
| DA_HB_HOUSTON | None | float64 | Yes |
| DA_HB_HUBAVG | None | float64 | Yes |
| DA_HB_NORTH | None | float64 | Yes |
| DA_HB_PAN | None | float64 | Yes |
| DA_HB_SOUTH | None | float64 | Yes |
| DA_HB_WEST | None | float64 | Yes |
| DA_LZ_AEN | None | float64 | Yes |
| DA_LZ_CPS | None | float64 | Yes |
| DA_LZ_HOUSTON | None | float64 | Yes |
| DA_LZ_LCRA | None | float64 | Yes |
| DA_LZ_NORTH | None | float64 | Yes |
| DA_LZ_RAYBN | None | float64 | Yes |
| DA_LZ_SOUTH | None | float64 | Yes |
| DA_LZ_WEST | None | float64 | Yes |
| RT_DC_E | None | float64 | Yes |
| RT_DC_L | None | float64 | Yes |
| RT_DC_N | None | float64 | Yes |
| RT_DC_R | None | float64 | Yes |
| RT_DC_S | None | float64 | Yes |
| RT_HB_BUSAVG | None | float64 | Yes |
| RT_HB_HOUSTON | None | float64 | Yes |
| RT_HB_HUBAVG | None | float64 | Yes |
| RT_HB_NORTH | None | float64 | Yes |
| RT_HB_PAN | None | float64 | Yes |
| RT_HB_SOUTH | None | float64 | Yes |
| RT_HB_WEST | None | float64 | Yes |
| RT_LZ_AEN | None | float64 | Yes |
| RT_LZ_CPS | None | float64 | Yes |
| RT_LZ_HOUSTON | None | float64 | Yes |
| RT_LZ_LCRA | None | float64 | Yes |
| RT_LZ_NORTH | None | float64 | Yes |
| RT_LZ_RAYBN | None | float64 | Yes |
| RT_LZ_SOUTH | None | float64 | Yes |
| RT_LZ_WEST | None | float64 | Yes |


### DA_AS_RT_5min_combined_2023_02.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_RT_5min_combined/DA_AS_RT_5min_combined_2023_02.parquet`

**Size**: 347.1 KB

**Rows**: 2,576

**Last Modified**: 2025-08-17T23:54:34.875156


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DeliveryInterval | None | int64 | Yes |
| AS_ECRS | None | float64 | Yes |
| AS_NSPIN | None | float64 | Yes |
| AS_REGDN | None | float64 | Yes |
| AS_REGUP | None | float64 | Yes |
| AS_RRS | None | float64 | Yes |
| DA_DC_E | None | float64 | Yes |
| DA_DC_L | None | float64 | Yes |
| DA_DC_N | None | float64 | Yes |
| DA_DC_R | None | float64 | Yes |
| DA_DC_S | None | float64 | Yes |
| DA_HB_BUSAVG | None | float64 | Yes |
| DA_HB_HOUSTON | None | float64 | Yes |
| DA_HB_HUBAVG | None | float64 | Yes |
| DA_HB_NORTH | None | float64 | Yes |
| DA_HB_PAN | None | float64 | Yes |
| DA_HB_SOUTH | None | float64 | Yes |
| DA_HB_WEST | None | float64 | Yes |
| DA_LZ_AEN | None | float64 | Yes |
| DA_LZ_CPS | None | float64 | Yes |
| DA_LZ_HOUSTON | None | float64 | Yes |
| DA_LZ_LCRA | None | float64 | Yes |
| DA_LZ_NORTH | None | float64 | Yes |
| DA_LZ_RAYBN | None | float64 | Yes |
| DA_LZ_SOUTH | None | float64 | Yes |
| DA_LZ_WEST | None | float64 | Yes |
| RT_DC_E | None | float64 | Yes |
| RT_DC_L | None | float64 | Yes |
| RT_DC_N | None | float64 | Yes |
| RT_DC_R | None | float64 | Yes |
| RT_DC_S | None | float64 | Yes |
| RT_HB_BUSAVG | None | float64 | Yes |
| RT_HB_HOUSTON | None | float64 | Yes |
| RT_HB_HUBAVG | None | float64 | Yes |
| RT_HB_NORTH | None | float64 | Yes |
| RT_HB_PAN | None | float64 | Yes |
| RT_HB_SOUTH | None | float64 | Yes |
| RT_HB_WEST | None | float64 | Yes |
| RT_LZ_AEN | None | float64 | Yes |
| RT_LZ_CPS | None | float64 | Yes |
| RT_LZ_HOUSTON | None | float64 | Yes |
| RT_LZ_LCRA | None | float64 | Yes |
| RT_LZ_NORTH | None | float64 | Yes |
| RT_LZ_RAYBN | None | float64 | Yes |
| RT_LZ_SOUTH | None | float64 | Yes |
| RT_LZ_WEST | None | float64 | Yes |


### DA_AS_RT_5min_combined_2023_03.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_RT_5min_combined/DA_AS_RT_5min_combined_2023_03.parquet`

**Size**: 372.6 KB

**Rows**: 2,848

**Last Modified**: 2025-08-17T23:54:34.880157


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DeliveryInterval | None | int64 | Yes |
| AS_ECRS | None | float64 | Yes |
| AS_NSPIN | None | float64 | Yes |
| AS_REGDN | None | float64 | Yes |
| AS_REGUP | None | float64 | Yes |
| AS_RRS | None | float64 | Yes |
| DA_DC_E | None | float64 | Yes |
| DA_DC_L | None | float64 | Yes |
| DA_DC_N | None | float64 | Yes |
| DA_DC_R | None | float64 | Yes |
| DA_DC_S | None | float64 | Yes |
| DA_HB_BUSAVG | None | float64 | Yes |
| DA_HB_HOUSTON | None | float64 | Yes |
| DA_HB_HUBAVG | None | float64 | Yes |
| DA_HB_NORTH | None | float64 | Yes |
| DA_HB_PAN | None | float64 | Yes |
| DA_HB_SOUTH | None | float64 | Yes |
| DA_HB_WEST | None | float64 | Yes |
| DA_LZ_AEN | None | float64 | Yes |
| DA_LZ_CPS | None | float64 | Yes |
| DA_LZ_HOUSTON | None | float64 | Yes |
| DA_LZ_LCRA | None | float64 | Yes |
| DA_LZ_NORTH | None | float64 | Yes |
| DA_LZ_RAYBN | None | float64 | Yes |
| DA_LZ_SOUTH | None | float64 | Yes |
| DA_LZ_WEST | None | float64 | Yes |
| RT_DC_E | None | float64 | Yes |
| RT_DC_L | None | float64 | Yes |
| RT_DC_N | None | float64 | Yes |
| RT_DC_R | None | float64 | Yes |
| RT_DC_S | None | float64 | Yes |
| RT_HB_BUSAVG | None | float64 | Yes |
| RT_HB_HOUSTON | None | float64 | Yes |
| RT_HB_HUBAVG | None | float64 | Yes |
| RT_HB_NORTH | None | float64 | Yes |
| RT_HB_PAN | None | float64 | Yes |
| RT_HB_SOUTH | None | float64 | Yes |
| RT_HB_WEST | None | float64 | Yes |
| RT_LZ_AEN | None | float64 | Yes |
| RT_LZ_CPS | None | float64 | Yes |
| RT_LZ_HOUSTON | None | float64 | Yes |
| RT_LZ_LCRA | None | float64 | Yes |
| RT_LZ_NORTH | None | float64 | Yes |
| RT_LZ_RAYBN | None | float64 | Yes |
| RT_LZ_SOUTH | None | float64 | Yes |
| RT_LZ_WEST | None | float64 | Yes |


### DA_AS_RT_5min_combined_2023_04.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_RT_5min_combined/DA_AS_RT_5min_combined_2023_04.parquet`

**Size**: 354.0 KB

**Rows**: 2,758

**Last Modified**: 2025-08-17T23:54:34.884157


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DeliveryInterval | None | int64 | Yes |
| AS_ECRS | None | float64 | Yes |
| AS_NSPIN | None | float64 | Yes |
| AS_REGDN | None | float64 | Yes |
| AS_REGUP | None | float64 | Yes |
| AS_RRS | None | float64 | Yes |
| DA_DC_E | None | float64 | Yes |
| DA_DC_L | None | float64 | Yes |
| DA_DC_N | None | float64 | Yes |
| DA_DC_R | None | float64 | Yes |
| DA_DC_S | None | float64 | Yes |
| DA_HB_BUSAVG | None | float64 | Yes |
| DA_HB_HOUSTON | None | float64 | Yes |
| DA_HB_HUBAVG | None | float64 | Yes |
| DA_HB_NORTH | None | float64 | Yes |
| DA_HB_PAN | None | float64 | Yes |
| DA_HB_SOUTH | None | float64 | Yes |
| DA_HB_WEST | None | float64 | Yes |
| DA_LZ_AEN | None | float64 | Yes |
| DA_LZ_CPS | None | float64 | Yes |
| DA_LZ_HOUSTON | None | float64 | Yes |
| DA_LZ_LCRA | None | float64 | Yes |
| DA_LZ_NORTH | None | float64 | Yes |
| DA_LZ_RAYBN | None | float64 | Yes |
| DA_LZ_SOUTH | None | float64 | Yes |
| DA_LZ_WEST | None | float64 | Yes |
| RT_DC_E | None | float64 | Yes |
| RT_DC_L | None | float64 | Yes |
| RT_DC_N | None | float64 | Yes |
| RT_DC_R | None | float64 | Yes |
| RT_DC_S | None | float64 | Yes |
| RT_HB_BUSAVG | None | float64 | Yes |
| RT_HB_HOUSTON | None | float64 | Yes |
| RT_HB_HUBAVG | None | float64 | Yes |
| RT_HB_NORTH | None | float64 | Yes |
| RT_HB_PAN | None | float64 | Yes |
| RT_HB_SOUTH | None | float64 | Yes |
| RT_HB_WEST | None | float64 | Yes |
| RT_LZ_AEN | None | float64 | Yes |
| RT_LZ_CPS | None | float64 | Yes |
| RT_LZ_HOUSTON | None | float64 | Yes |
| RT_LZ_LCRA | None | float64 | Yes |
| RT_LZ_NORTH | None | float64 | Yes |
| RT_LZ_RAYBN | None | float64 | Yes |
| RT_LZ_SOUTH | None | float64 | Yes |
| RT_LZ_WEST | None | float64 | Yes |


### DA_AS_RT_5min_combined_2023_05.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_RT_5min_combined/DA_AS_RT_5min_combined_2023_05.parquet`

**Size**: 351.3 KB

**Rows**: 2,852

**Last Modified**: 2025-08-17T23:54:34.889157


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DeliveryInterval | None | int64 | Yes |
| AS_ECRS | None | float64 | Yes |
| AS_NSPIN | None | float64 | Yes |
| AS_REGDN | None | float64 | Yes |
| AS_REGUP | None | float64 | Yes |
| AS_RRS | None | float64 | Yes |
| DA_DC_E | None | float64 | Yes |
| DA_DC_L | None | float64 | Yes |
| DA_DC_N | None | float64 | Yes |
| DA_DC_R | None | float64 | Yes |
| DA_DC_S | None | float64 | Yes |
| DA_HB_BUSAVG | None | float64 | Yes |
| DA_HB_HOUSTON | None | float64 | Yes |
| DA_HB_HUBAVG | None | float64 | Yes |
| DA_HB_NORTH | None | float64 | Yes |
| DA_HB_PAN | None | float64 | Yes |
| DA_HB_SOUTH | None | float64 | Yes |
| DA_HB_WEST | None | float64 | Yes |
| DA_LZ_AEN | None | float64 | Yes |
| DA_LZ_CPS | None | float64 | Yes |
| DA_LZ_HOUSTON | None | float64 | Yes |
| DA_LZ_LCRA | None | float64 | Yes |
| DA_LZ_NORTH | None | float64 | Yes |
| DA_LZ_RAYBN | None | float64 | Yes |
| DA_LZ_SOUTH | None | float64 | Yes |
| DA_LZ_WEST | None | float64 | Yes |
| RT_DC_E | None | float64 | Yes |
| RT_DC_L | None | float64 | Yes |
| RT_DC_N | None | float64 | Yes |
| RT_DC_R | None | float64 | Yes |
| RT_DC_S | None | float64 | Yes |
| RT_HB_BUSAVG | None | float64 | Yes |
| RT_HB_HOUSTON | None | float64 | Yes |
| RT_HB_HUBAVG | None | float64 | Yes |
| RT_HB_NORTH | None | float64 | Yes |
| RT_HB_PAN | None | float64 | Yes |
| RT_HB_SOUTH | None | float64 | Yes |
| RT_HB_WEST | None | float64 | Yes |
| RT_LZ_AEN | None | float64 | Yes |
| RT_LZ_CPS | None | float64 | Yes |
| RT_LZ_HOUSTON | None | float64 | Yes |
| RT_LZ_LCRA | None | float64 | Yes |
| RT_LZ_NORTH | None | float64 | Yes |
| RT_LZ_RAYBN | None | float64 | Yes |
| RT_LZ_SOUTH | None | float64 | Yes |
| RT_LZ_WEST | None | float64 | Yes |


### DA_AS_RT_5min_combined_2023_06.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_RT_5min_combined/DA_AS_RT_5min_combined_2023_06.parquet`

**Size**: 348.8 KB

**Rows**: 2,760

**Last Modified**: 2025-08-17T23:54:34.893157


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DeliveryInterval | None | int64 | Yes |
| AS_ECRS | None | float64 | Yes |
| AS_NSPIN | None | float64 | Yes |
| AS_REGDN | None | float64 | Yes |
| AS_REGUP | None | float64 | Yes |
| AS_RRS | None | float64 | Yes |
| DA_DC_E | None | float64 | Yes |
| DA_DC_L | None | float64 | Yes |
| DA_DC_N | None | float64 | Yes |
| DA_DC_R | None | float64 | Yes |
| DA_DC_S | None | float64 | Yes |
| DA_HB_BUSAVG | None | float64 | Yes |
| DA_HB_HOUSTON | None | float64 | Yes |
| DA_HB_HUBAVG | None | float64 | Yes |
| DA_HB_NORTH | None | float64 | Yes |
| DA_HB_PAN | None | float64 | Yes |
| DA_HB_SOUTH | None | float64 | Yes |
| DA_HB_WEST | None | float64 | Yes |
| DA_LZ_AEN | None | float64 | Yes |
| DA_LZ_CPS | None | float64 | Yes |
| DA_LZ_HOUSTON | None | float64 | Yes |
| DA_LZ_LCRA | None | float64 | Yes |
| DA_LZ_NORTH | None | float64 | Yes |
| DA_LZ_RAYBN | None | float64 | Yes |
| DA_LZ_SOUTH | None | float64 | Yes |
| DA_LZ_WEST | None | float64 | Yes |
| RT_DC_E | None | float64 | Yes |
| RT_DC_L | None | float64 | Yes |
| RT_DC_N | None | float64 | Yes |
| RT_DC_R | None | float64 | Yes |
| RT_DC_S | None | float64 | Yes |
| RT_HB_BUSAVG | None | float64 | Yes |
| RT_HB_HOUSTON | None | float64 | Yes |
| RT_HB_HUBAVG | None | float64 | Yes |
| RT_HB_NORTH | None | float64 | Yes |
| RT_HB_PAN | None | float64 | Yes |
| RT_HB_SOUTH | None | float64 | Yes |
| RT_HB_WEST | None | float64 | Yes |
| RT_LZ_AEN | None | float64 | Yes |
| RT_LZ_CPS | None | float64 | Yes |
| RT_LZ_HOUSTON | None | float64 | Yes |
| RT_LZ_LCRA | None | float64 | Yes |
| RT_LZ_NORTH | None | float64 | Yes |
| RT_LZ_RAYBN | None | float64 | Yes |
| RT_LZ_SOUTH | None | float64 | Yes |
| RT_LZ_WEST | None | float64 | Yes |


### DA_AS_RT_5min_combined_2023_07.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_RT_5min_combined/DA_AS_RT_5min_combined_2023_07.parquet`

**Size**: 351.9 KB

**Rows**: 2,852

**Last Modified**: 2025-08-17T23:54:34.898157


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DeliveryInterval | None | int64 | Yes |
| AS_ECRS | None | float64 | Yes |
| AS_NSPIN | None | float64 | Yes |
| AS_REGDN | None | float64 | Yes |
| AS_REGUP | None | float64 | Yes |
| AS_RRS | None | float64 | Yes |
| DA_DC_E | None | float64 | Yes |
| DA_DC_L | None | float64 | Yes |
| DA_DC_N | None | float64 | Yes |
| DA_DC_R | None | float64 | Yes |
| DA_DC_S | None | float64 | Yes |
| DA_HB_BUSAVG | None | float64 | Yes |
| DA_HB_HOUSTON | None | float64 | Yes |
| DA_HB_HUBAVG | None | float64 | Yes |
| DA_HB_NORTH | None | float64 | Yes |
| DA_HB_PAN | None | float64 | Yes |
| DA_HB_SOUTH | None | float64 | Yes |
| DA_HB_WEST | None | float64 | Yes |
| DA_LZ_AEN | None | float64 | Yes |
| DA_LZ_CPS | None | float64 | Yes |
| DA_LZ_HOUSTON | None | float64 | Yes |
| DA_LZ_LCRA | None | float64 | Yes |
| DA_LZ_NORTH | None | float64 | Yes |
| DA_LZ_RAYBN | None | float64 | Yes |
| DA_LZ_SOUTH | None | float64 | Yes |
| DA_LZ_WEST | None | float64 | Yes |
| RT_DC_E | None | float64 | Yes |
| RT_DC_L | None | float64 | Yes |
| RT_DC_N | None | float64 | Yes |
| RT_DC_R | None | float64 | Yes |
| RT_DC_S | None | float64 | Yes |
| RT_HB_BUSAVG | None | float64 | Yes |
| RT_HB_HOUSTON | None | float64 | Yes |
| RT_HB_HUBAVG | None | float64 | Yes |
| RT_HB_NORTH | None | float64 | Yes |
| RT_HB_PAN | None | float64 | Yes |
| RT_HB_SOUTH | None | float64 | Yes |
| RT_HB_WEST | None | float64 | Yes |
| RT_LZ_AEN | None | float64 | Yes |
| RT_LZ_CPS | None | float64 | Yes |
| RT_LZ_HOUSTON | None | float64 | Yes |
| RT_LZ_LCRA | None | float64 | Yes |
| RT_LZ_NORTH | None | float64 | Yes |
| RT_LZ_RAYBN | None | float64 | Yes |
| RT_LZ_SOUTH | None | float64 | Yes |
| RT_LZ_WEST | None | float64 | Yes |


### DA_AS_RT_5min_combined_2023_08.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_RT_5min_combined/DA_AS_RT_5min_combined_2023_08.parquet`

**Size**: 389.5 KB

**Rows**: 2,852

**Last Modified**: 2025-08-17T23:54:34.902158


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DeliveryInterval | None | int64 | Yes |
| AS_ECRS | None | float64 | Yes |
| AS_NSPIN | None | float64 | Yes |
| AS_REGDN | None | float64 | Yes |
| AS_REGUP | None | float64 | Yes |
| AS_RRS | None | float64 | Yes |
| DA_DC_E | None | float64 | Yes |
| DA_DC_L | None | float64 | Yes |
| DA_DC_N | None | float64 | Yes |
| DA_DC_R | None | float64 | Yes |
| DA_DC_S | None | float64 | Yes |
| DA_HB_BUSAVG | None | float64 | Yes |
| DA_HB_HOUSTON | None | float64 | Yes |
| DA_HB_HUBAVG | None | float64 | Yes |
| DA_HB_NORTH | None | float64 | Yes |
| DA_HB_PAN | None | float64 | Yes |
| DA_HB_SOUTH | None | float64 | Yes |
| DA_HB_WEST | None | float64 | Yes |
| DA_LZ_AEN | None | float64 | Yes |
| DA_LZ_CPS | None | float64 | Yes |
| DA_LZ_HOUSTON | None | float64 | Yes |
| DA_LZ_LCRA | None | float64 | Yes |
| DA_LZ_NORTH | None | float64 | Yes |
| DA_LZ_RAYBN | None | float64 | Yes |
| DA_LZ_SOUTH | None | float64 | Yes |
| DA_LZ_WEST | None | float64 | Yes |
| RT_DC_E | None | float64 | Yes |
| RT_DC_L | None | float64 | Yes |
| RT_DC_N | None | float64 | Yes |
| RT_DC_R | None | float64 | Yes |
| RT_DC_S | None | float64 | Yes |
| RT_HB_BUSAVG | None | float64 | Yes |
| RT_HB_HOUSTON | None | float64 | Yes |
| RT_HB_HUBAVG | None | float64 | Yes |
| RT_HB_NORTH | None | float64 | Yes |
| RT_HB_PAN | None | float64 | Yes |
| RT_HB_SOUTH | None | float64 | Yes |
| RT_HB_WEST | None | float64 | Yes |
| RT_LZ_AEN | None | float64 | Yes |
| RT_LZ_CPS | None | float64 | Yes |
| RT_LZ_HOUSTON | None | float64 | Yes |
| RT_LZ_LCRA | None | float64 | Yes |
| RT_LZ_NORTH | None | float64 | Yes |
| RT_LZ_RAYBN | None | float64 | Yes |
| RT_LZ_SOUTH | None | float64 | Yes |
| RT_LZ_WEST | None | float64 | Yes |


### DA_AS_RT_5min_combined_2023_09.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_RT_5min_combined/DA_AS_RT_5min_combined_2023_09.parquet`

**Size**: 346.6 KB

**Rows**: 2,760

**Last Modified**: 2025-08-17T23:54:34.907158


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DeliveryInterval | None | int64 | Yes |
| AS_ECRS | None | float64 | Yes |
| AS_NSPIN | None | float64 | Yes |
| AS_REGDN | None | float64 | Yes |
| AS_REGUP | None | float64 | Yes |
| AS_RRS | None | float64 | Yes |
| DA_DC_E | None | float64 | Yes |
| DA_DC_L | None | float64 | Yes |
| DA_DC_N | None | float64 | Yes |
| DA_DC_R | None | float64 | Yes |
| DA_DC_S | None | float64 | Yes |
| DA_HB_BUSAVG | None | float64 | Yes |
| DA_HB_HOUSTON | None | float64 | Yes |
| DA_HB_HUBAVG | None | float64 | Yes |
| DA_HB_NORTH | None | float64 | Yes |
| DA_HB_PAN | None | float64 | Yes |
| DA_HB_SOUTH | None | float64 | Yes |
| DA_HB_WEST | None | float64 | Yes |
| DA_LZ_AEN | None | float64 | Yes |
| DA_LZ_CPS | None | float64 | Yes |
| DA_LZ_HOUSTON | None | float64 | Yes |
| DA_LZ_LCRA | None | float64 | Yes |
| DA_LZ_NORTH | None | float64 | Yes |
| DA_LZ_RAYBN | None | float64 | Yes |
| DA_LZ_SOUTH | None | float64 | Yes |
| DA_LZ_WEST | None | float64 | Yes |
| RT_DC_E | None | float64 | Yes |
| RT_DC_L | None | float64 | Yes |
| RT_DC_N | None | float64 | Yes |
| RT_DC_R | None | float64 | Yes |
| RT_DC_S | None | float64 | Yes |
| RT_HB_BUSAVG | None | float64 | Yes |
| RT_HB_HOUSTON | None | float64 | Yes |
| RT_HB_HUBAVG | None | float64 | Yes |
| RT_HB_NORTH | None | float64 | Yes |
| RT_HB_PAN | None | float64 | Yes |
| RT_HB_SOUTH | None | float64 | Yes |
| RT_HB_WEST | None | float64 | Yes |
| RT_LZ_AEN | None | float64 | Yes |
| RT_LZ_CPS | None | float64 | Yes |
| RT_LZ_HOUSTON | None | float64 | Yes |
| RT_LZ_LCRA | None | float64 | Yes |
| RT_LZ_NORTH | None | float64 | Yes |
| RT_LZ_RAYBN | None | float64 | Yes |
| RT_LZ_SOUTH | None | float64 | Yes |
| RT_LZ_WEST | None | float64 | Yes |


### DA_AS_RT_5min_combined_2023_10.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_RT_5min_combined/DA_AS_RT_5min_combined_2023_10.parquet`

**Size**: 343.0 KB

**Rows**: 2,852

**Last Modified**: 2025-08-17T23:54:34.911158


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DeliveryInterval | None | int64 | Yes |
| AS_ECRS | None | float64 | Yes |
| AS_NSPIN | None | float64 | Yes |
| AS_REGDN | None | float64 | Yes |
| AS_REGUP | None | float64 | Yes |
| AS_RRS | None | float64 | Yes |
| DA_DC_E | None | float64 | Yes |
| DA_DC_L | None | float64 | Yes |
| DA_DC_N | None | float64 | Yes |
| DA_DC_R | None | float64 | Yes |
| DA_DC_S | None | float64 | Yes |
| DA_HB_BUSAVG | None | float64 | Yes |
| DA_HB_HOUSTON | None | float64 | Yes |
| DA_HB_HUBAVG | None | float64 | Yes |
| DA_HB_NORTH | None | float64 | Yes |
| DA_HB_PAN | None | float64 | Yes |
| DA_HB_SOUTH | None | float64 | Yes |
| DA_HB_WEST | None | float64 | Yes |
| DA_LZ_AEN | None | float64 | Yes |
| DA_LZ_CPS | None | float64 | Yes |
| DA_LZ_HOUSTON | None | float64 | Yes |
| DA_LZ_LCRA | None | float64 | Yes |
| DA_LZ_NORTH | None | float64 | Yes |
| DA_LZ_RAYBN | None | float64 | Yes |
| DA_LZ_SOUTH | None | float64 | Yes |
| DA_LZ_WEST | None | float64 | Yes |
| RT_DC_E | None | float64 | Yes |
| RT_DC_L | None | float64 | Yes |
| RT_DC_N | None | float64 | Yes |
| RT_DC_R | None | float64 | Yes |
| RT_DC_S | None | float64 | Yes |
| RT_HB_BUSAVG | None | float64 | Yes |
| RT_HB_HOUSTON | None | float64 | Yes |
| RT_HB_HUBAVG | None | float64 | Yes |
| RT_HB_NORTH | None | float64 | Yes |
| RT_HB_PAN | None | float64 | Yes |
| RT_HB_SOUTH | None | float64 | Yes |
| RT_HB_WEST | None | float64 | Yes |
| RT_LZ_AEN | None | float64 | Yes |
| RT_LZ_CPS | None | float64 | Yes |
| RT_LZ_HOUSTON | None | float64 | Yes |
| RT_LZ_LCRA | None | float64 | Yes |
| RT_LZ_NORTH | None | float64 | Yes |
| RT_LZ_RAYBN | None | float64 | Yes |
| RT_LZ_SOUTH | None | float64 | Yes |
| RT_LZ_WEST | None | float64 | Yes |


### DA_AS_RT_5min_combined_2023_11.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_RT_5min_combined/DA_AS_RT_5min_combined_2023_11.parquet`

**Size**: 344.8 KB

**Rows**: 2,760

**Last Modified**: 2025-08-17T23:54:34.915158


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DeliveryInterval | None | int64 | Yes |
| AS_ECRS | None | float64 | Yes |
| AS_NSPIN | None | float64 | Yes |
| AS_REGDN | None | float64 | Yes |
| AS_REGUP | None | float64 | Yes |
| AS_RRS | None | float64 | Yes |
| DA_DC_E | None | float64 | Yes |
| DA_DC_L | None | float64 | Yes |
| DA_DC_N | None | float64 | Yes |
| DA_DC_R | None | float64 | Yes |
| DA_DC_S | None | float64 | Yes |
| DA_HB_BUSAVG | None | float64 | Yes |
| DA_HB_HOUSTON | None | float64 | Yes |
| DA_HB_HUBAVG | None | float64 | Yes |
| DA_HB_NORTH | None | float64 | Yes |
| DA_HB_PAN | None | float64 | Yes |
| DA_HB_SOUTH | None | float64 | Yes |
| DA_HB_WEST | None | float64 | Yes |
| DA_LZ_AEN | None | float64 | Yes |
| DA_LZ_CPS | None | float64 | Yes |
| DA_LZ_HOUSTON | None | float64 | Yes |
| DA_LZ_LCRA | None | float64 | Yes |
| DA_LZ_NORTH | None | float64 | Yes |
| DA_LZ_RAYBN | None | float64 | Yes |
| DA_LZ_SOUTH | None | float64 | Yes |
| DA_LZ_WEST | None | float64 | Yes |
| RT_DC_E | None | float64 | Yes |
| RT_DC_L | None | float64 | Yes |
| RT_DC_N | None | float64 | Yes |
| RT_DC_R | None | float64 | Yes |
| RT_DC_S | None | float64 | Yes |
| RT_HB_BUSAVG | None | float64 | Yes |
| RT_HB_HOUSTON | None | float64 | Yes |
| RT_HB_HUBAVG | None | float64 | Yes |
| RT_HB_NORTH | None | float64 | Yes |
| RT_HB_PAN | None | float64 | Yes |
| RT_HB_SOUTH | None | float64 | Yes |
| RT_HB_WEST | None | float64 | Yes |
| RT_LZ_AEN | None | float64 | Yes |
| RT_LZ_CPS | None | float64 | Yes |
| RT_LZ_HOUSTON | None | float64 | Yes |
| RT_LZ_LCRA | None | float64 | Yes |
| RT_LZ_NORTH | None | float64 | Yes |
| RT_LZ_RAYBN | None | float64 | Yes |
| RT_LZ_SOUTH | None | float64 | Yes |
| RT_LZ_WEST | None | float64 | Yes |


### DA_AS_RT_5min_combined_2023_12.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_RT_5min_combined/DA_AS_RT_5min_combined_2023_12.parquet`

**Size**: 339.5 KB

**Rows**: 2,852

**Last Modified**: 2025-08-17T23:54:34.920158


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DeliveryInterval | None | int64 | Yes |
| AS_ECRS | None | float64 | Yes |
| AS_NSPIN | None | float64 | Yes |
| AS_REGDN | None | float64 | Yes |
| AS_REGUP | None | float64 | Yes |
| AS_RRS | None | float64 | Yes |
| DA_DC_E | None | float64 | Yes |
| DA_DC_L | None | float64 | Yes |
| DA_DC_N | None | float64 | Yes |
| DA_DC_R | None | float64 | Yes |
| DA_DC_S | None | float64 | Yes |
| DA_HB_BUSAVG | None | float64 | Yes |
| DA_HB_HOUSTON | None | float64 | Yes |
| DA_HB_HUBAVG | None | float64 | Yes |
| DA_HB_NORTH | None | float64 | Yes |
| DA_HB_PAN | None | float64 | Yes |
| DA_HB_SOUTH | None | float64 | Yes |
| DA_HB_WEST | None | float64 | Yes |
| DA_LZ_AEN | None | float64 | Yes |
| DA_LZ_CPS | None | float64 | Yes |
| DA_LZ_HOUSTON | None | float64 | Yes |
| DA_LZ_LCRA | None | float64 | Yes |
| DA_LZ_NORTH | None | float64 | Yes |
| DA_LZ_RAYBN | None | float64 | Yes |
| DA_LZ_SOUTH | None | float64 | Yes |
| DA_LZ_WEST | None | float64 | Yes |
| RT_DC_E | None | float64 | Yes |
| RT_DC_L | None | float64 | Yes |
| RT_DC_N | None | float64 | Yes |
| RT_DC_R | None | float64 | Yes |
| RT_DC_S | None | float64 | Yes |
| RT_HB_BUSAVG | None | float64 | Yes |
| RT_HB_HOUSTON | None | float64 | Yes |
| RT_HB_HUBAVG | None | float64 | Yes |
| RT_HB_NORTH | None | float64 | Yes |
| RT_HB_PAN | None | float64 | Yes |
| RT_HB_SOUTH | None | float64 | Yes |
| RT_HB_WEST | None | float64 | Yes |
| RT_LZ_AEN | None | float64 | Yes |
| RT_LZ_CPS | None | float64 | Yes |
| RT_LZ_HOUSTON | None | float64 | Yes |
| RT_LZ_LCRA | None | float64 | Yes |
| RT_LZ_NORTH | None | float64 | Yes |
| RT_LZ_RAYBN | None | float64 | Yes |
| RT_LZ_SOUTH | None | float64 | Yes |
| RT_LZ_WEST | None | float64 | Yes |


### DA_AS_RT_5min_combined_2024_01.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_RT_5min_combined/DA_AS_RT_5min_combined_2024_01.parquet`

**Size**: 363.4 KB

**Rows**: 2,852

**Last Modified**: 2025-08-17T23:54:35.003162


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DeliveryInterval | None | int64 | Yes |
| AS_ECRS | None | float64 | Yes |
| AS_NSPIN | None | float64 | Yes |
| AS_REGDN | None | float64 | Yes |
| AS_REGUP | None | float64 | Yes |
| AS_RRS | None | float64 | Yes |
| DA_DC_E | None | float64 | Yes |
| DA_DC_L | None | float64 | Yes |
| DA_DC_N | None | float64 | Yes |
| DA_DC_R | None | float64 | Yes |
| DA_DC_S | None | float64 | Yes |
| DA_HB_BUSAVG | None | float64 | Yes |
| DA_HB_HOUSTON | None | float64 | Yes |
| DA_HB_HUBAVG | None | float64 | Yes |
| DA_HB_NORTH | None | float64 | Yes |
| DA_HB_PAN | None | float64 | Yes |
| DA_HB_SOUTH | None | float64 | Yes |
| DA_HB_WEST | None | float64 | Yes |
| DA_LZ_AEN | None | float64 | Yes |
| DA_LZ_CPS | None | float64 | Yes |
| DA_LZ_HOUSTON | None | float64 | Yes |
| DA_LZ_LCRA | None | float64 | Yes |
| DA_LZ_NORTH | None | float64 | Yes |
| DA_LZ_RAYBN | None | float64 | Yes |
| DA_LZ_SOUTH | None | float64 | Yes |
| DA_LZ_WEST | None | float64 | Yes |
| RT_DC_E | None | float64 | Yes |
| RT_DC_L | None | float64 | Yes |
| RT_DC_N | None | float64 | Yes |
| RT_DC_R | None | float64 | Yes |
| RT_DC_S | None | float64 | Yes |
| RT_HB_BUSAVG | None | float64 | Yes |
| RT_HB_HOUSTON | None | float64 | Yes |
| RT_HB_HUBAVG | None | float64 | Yes |
| RT_HB_NORTH | None | float64 | Yes |
| RT_HB_PAN | None | float64 | Yes |
| RT_HB_SOUTH | None | float64 | Yes |
| RT_HB_WEST | None | float64 | Yes |
| RT_LZ_AEN | None | float64 | Yes |
| RT_LZ_CPS | None | float64 | Yes |
| RT_LZ_HOUSTON | None | float64 | Yes |
| RT_LZ_LCRA | None | float64 | Yes |
| RT_LZ_NORTH | None | float64 | Yes |
| RT_LZ_RAYBN | None | float64 | Yes |
| RT_LZ_SOUTH | None | float64 | Yes |
| RT_LZ_WEST | None | float64 | Yes |


### DA_AS_RT_5min_combined_2024_02.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_RT_5min_combined/DA_AS_RT_5min_combined_2024_02.parquet`

**Size**: 323.7 KB

**Rows**: 2,668

**Last Modified**: 2025-08-17T23:54:35.007163


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DeliveryInterval | None | int64 | Yes |
| AS_ECRS | None | float64 | Yes |
| AS_NSPIN | None | float64 | Yes |
| AS_REGDN | None | float64 | Yes |
| AS_REGUP | None | float64 | Yes |
| AS_RRS | None | float64 | Yes |
| DA_DC_E | None | float64 | Yes |
| DA_DC_L | None | float64 | Yes |
| DA_DC_N | None | float64 | Yes |
| DA_DC_R | None | float64 | Yes |
| DA_DC_S | None | float64 | Yes |
| DA_HB_BUSAVG | None | float64 | Yes |
| DA_HB_HOUSTON | None | float64 | Yes |
| DA_HB_HUBAVG | None | float64 | Yes |
| DA_HB_NORTH | None | float64 | Yes |
| DA_HB_PAN | None | float64 | Yes |
| DA_HB_SOUTH | None | float64 | Yes |
| DA_HB_WEST | None | float64 | Yes |
| DA_LZ_AEN | None | float64 | Yes |
| DA_LZ_CPS | None | float64 | Yes |
| DA_LZ_HOUSTON | None | float64 | Yes |
| DA_LZ_LCRA | None | float64 | Yes |
| DA_LZ_NORTH | None | float64 | Yes |
| DA_LZ_RAYBN | None | float64 | Yes |
| DA_LZ_SOUTH | None | float64 | Yes |
| DA_LZ_WEST | None | float64 | Yes |
| RT_DC_E | None | float64 | Yes |
| RT_DC_L | None | float64 | Yes |
| RT_DC_N | None | float64 | Yes |
| RT_DC_R | None | float64 | Yes |
| RT_DC_S | None | float64 | Yes |
| RT_HB_BUSAVG | None | float64 | Yes |
| RT_HB_HOUSTON | None | float64 | Yes |
| RT_HB_HUBAVG | None | float64 | Yes |
| RT_HB_NORTH | None | float64 | Yes |
| RT_HB_PAN | None | float64 | Yes |
| RT_HB_SOUTH | None | float64 | Yes |
| RT_HB_WEST | None | float64 | Yes |
| RT_LZ_AEN | None | float64 | Yes |
| RT_LZ_CPS | None | float64 | Yes |
| RT_LZ_HOUSTON | None | float64 | Yes |
| RT_LZ_LCRA | None | float64 | Yes |
| RT_LZ_NORTH | None | float64 | Yes |
| RT_LZ_RAYBN | None | float64 | Yes |
| RT_LZ_SOUTH | None | float64 | Yes |
| RT_LZ_WEST | None | float64 | Yes |


### DA_AS_RT_5min_combined_2024_03.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_RT_5min_combined/DA_AS_RT_5min_combined_2024_03.parquet`

**Size**: 357.8 KB

**Rows**: 2,848

**Last Modified**: 2025-08-17T23:54:35.012163


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DeliveryInterval | None | int64 | Yes |
| AS_ECRS | None | float64 | Yes |
| AS_NSPIN | None | float64 | Yes |
| AS_REGDN | None | float64 | Yes |
| AS_REGUP | None | float64 | Yes |
| AS_RRS | None | float64 | Yes |
| DA_DC_E | None | float64 | Yes |
| DA_DC_L | None | float64 | Yes |
| DA_DC_N | None | float64 | Yes |
| DA_DC_R | None | float64 | Yes |
| DA_DC_S | None | float64 | Yes |
| DA_HB_BUSAVG | None | float64 | Yes |
| DA_HB_HOUSTON | None | float64 | Yes |
| DA_HB_HUBAVG | None | float64 | Yes |
| DA_HB_NORTH | None | float64 | Yes |
| DA_HB_PAN | None | float64 | Yes |
| DA_HB_SOUTH | None | float64 | Yes |
| DA_HB_WEST | None | float64 | Yes |
| DA_LZ_AEN | None | float64 | Yes |
| DA_LZ_CPS | None | float64 | Yes |
| DA_LZ_HOUSTON | None | float64 | Yes |
| DA_LZ_LCRA | None | float64 | Yes |
| DA_LZ_NORTH | None | float64 | Yes |
| DA_LZ_RAYBN | None | float64 | Yes |
| DA_LZ_SOUTH | None | float64 | Yes |
| DA_LZ_WEST | None | float64 | Yes |
| RT_DC_E | None | float64 | Yes |
| RT_DC_L | None | float64 | Yes |
| RT_DC_N | None | float64 | Yes |
| RT_DC_R | None | float64 | Yes |
| RT_DC_S | None | float64 | Yes |
| RT_HB_BUSAVG | None | float64 | Yes |
| RT_HB_HOUSTON | None | float64 | Yes |
| RT_HB_HUBAVG | None | float64 | Yes |
| RT_HB_NORTH | None | float64 | Yes |
| RT_HB_PAN | None | float64 | Yes |
| RT_HB_SOUTH | None | float64 | Yes |
| RT_HB_WEST | None | float64 | Yes |
| RT_LZ_AEN | None | float64 | Yes |
| RT_LZ_CPS | None | float64 | Yes |
| RT_LZ_HOUSTON | None | float64 | Yes |
| RT_LZ_LCRA | None | float64 | Yes |
| RT_LZ_NORTH | None | float64 | Yes |
| RT_LZ_RAYBN | None | float64 | Yes |
| RT_LZ_SOUTH | None | float64 | Yes |
| RT_LZ_WEST | None | float64 | Yes |


### DA_AS_RT_5min_combined_2024_04.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_RT_5min_combined/DA_AS_RT_5min_combined_2024_04.parquet`

**Size**: 342.9 KB

**Rows**: 2,760

**Last Modified**: 2025-08-17T23:54:35.016163


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DeliveryInterval | None | int64 | Yes |
| AS_ECRS | None | float64 | Yes |
| AS_NSPIN | None | float64 | Yes |
| AS_REGDN | None | float64 | Yes |
| AS_REGUP | None | float64 | Yes |
| AS_RRS | None | float64 | Yes |
| DA_DC_E | None | float64 | Yes |
| DA_DC_L | None | float64 | Yes |
| DA_DC_N | None | float64 | Yes |
| DA_DC_R | None | float64 | Yes |
| DA_DC_S | None | float64 | Yes |
| DA_HB_BUSAVG | None | float64 | Yes |
| DA_HB_HOUSTON | None | float64 | Yes |
| DA_HB_HUBAVG | None | float64 | Yes |
| DA_HB_NORTH | None | float64 | Yes |
| DA_HB_PAN | None | float64 | Yes |
| DA_HB_SOUTH | None | float64 | Yes |
| DA_HB_WEST | None | float64 | Yes |
| DA_LZ_AEN | None | float64 | Yes |
| DA_LZ_CPS | None | float64 | Yes |
| DA_LZ_HOUSTON | None | float64 | Yes |
| DA_LZ_LCRA | None | float64 | Yes |
| DA_LZ_NORTH | None | float64 | Yes |
| DA_LZ_RAYBN | None | float64 | Yes |
| DA_LZ_SOUTH | None | float64 | Yes |
| DA_LZ_WEST | None | float64 | Yes |
| RT_DC_E | None | float64 | Yes |
| RT_DC_L | None | float64 | Yes |
| RT_DC_N | None | float64 | Yes |
| RT_DC_R | None | float64 | Yes |
| RT_DC_S | None | float64 | Yes |
| RT_HB_BUSAVG | None | float64 | Yes |
| RT_HB_HOUSTON | None | float64 | Yes |
| RT_HB_HUBAVG | None | float64 | Yes |
| RT_HB_NORTH | None | float64 | Yes |
| RT_HB_PAN | None | float64 | Yes |
| RT_HB_SOUTH | None | float64 | Yes |
| RT_HB_WEST | None | float64 | Yes |
| RT_LZ_AEN | None | float64 | Yes |
| RT_LZ_CPS | None | float64 | Yes |
| RT_LZ_HOUSTON | None | float64 | Yes |
| RT_LZ_LCRA | None | float64 | Yes |
| RT_LZ_NORTH | None | float64 | Yes |
| RT_LZ_RAYBN | None | float64 | Yes |
| RT_LZ_SOUTH | None | float64 | Yes |
| RT_LZ_WEST | None | float64 | Yes |


### DA_AS_RT_5min_combined_2024_05.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_RT_5min_combined/DA_AS_RT_5min_combined_2024_05.parquet`

**Size**: 352.6 KB

**Rows**: 2,852

**Last Modified**: 2025-08-17T23:54:35.021163


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DeliveryInterval | None | int64 | Yes |
| AS_ECRS | None | float64 | Yes |
| AS_NSPIN | None | float64 | Yes |
| AS_REGDN | None | float64 | Yes |
| AS_REGUP | None | float64 | Yes |
| AS_RRS | None | float64 | Yes |
| DA_DC_E | None | float64 | Yes |
| DA_DC_L | None | float64 | Yes |
| DA_DC_N | None | float64 | Yes |
| DA_DC_R | None | float64 | Yes |
| DA_DC_S | None | float64 | Yes |
| DA_HB_BUSAVG | None | float64 | Yes |
| DA_HB_HOUSTON | None | float64 | Yes |
| DA_HB_HUBAVG | None | float64 | Yes |
| DA_HB_NORTH | None | float64 | Yes |
| DA_HB_PAN | None | float64 | Yes |
| DA_HB_SOUTH | None | float64 | Yes |
| DA_HB_WEST | None | float64 | Yes |
| DA_LZ_AEN | None | float64 | Yes |
| DA_LZ_CPS | None | float64 | Yes |
| DA_LZ_HOUSTON | None | float64 | Yes |
| DA_LZ_LCRA | None | float64 | Yes |
| DA_LZ_NORTH | None | float64 | Yes |
| DA_LZ_RAYBN | None | float64 | Yes |
| DA_LZ_SOUTH | None | float64 | Yes |
| DA_LZ_WEST | None | float64 | Yes |
| RT_DC_E | None | float64 | Yes |
| RT_DC_L | None | float64 | Yes |
| RT_DC_N | None | float64 | Yes |
| RT_DC_R | None | float64 | Yes |
| RT_DC_S | None | float64 | Yes |
| RT_HB_BUSAVG | None | float64 | Yes |
| RT_HB_HOUSTON | None | float64 | Yes |
| RT_HB_HUBAVG | None | float64 | Yes |
| RT_HB_NORTH | None | float64 | Yes |
| RT_HB_PAN | None | float64 | Yes |
| RT_HB_SOUTH | None | float64 | Yes |
| RT_HB_WEST | None | float64 | Yes |
| RT_LZ_AEN | None | float64 | Yes |
| RT_LZ_CPS | None | float64 | Yes |
| RT_LZ_HOUSTON | None | float64 | Yes |
| RT_LZ_LCRA | None | float64 | Yes |
| RT_LZ_NORTH | None | float64 | Yes |
| RT_LZ_RAYBN | None | float64 | Yes |
| RT_LZ_SOUTH | None | float64 | Yes |
| RT_LZ_WEST | None | float64 | Yes |


### DA_AS_RT_5min_combined_2024_06.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_RT_5min_combined/DA_AS_RT_5min_combined_2024_06.parquet`

**Size**: 346.7 KB

**Rows**: 2,760

**Last Modified**: 2025-08-17T23:54:35.025163


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DeliveryInterval | None | int64 | Yes |
| AS_ECRS | None | float64 | Yes |
| AS_NSPIN | None | float64 | Yes |
| AS_REGDN | None | float64 | Yes |
| AS_REGUP | None | float64 | Yes |
| AS_RRS | None | float64 | Yes |
| DA_DC_E | None | float64 | Yes |
| DA_DC_L | None | float64 | Yes |
| DA_DC_N | None | float64 | Yes |
| DA_DC_R | None | float64 | Yes |
| DA_DC_S | None | float64 | Yes |
| DA_HB_BUSAVG | None | float64 | Yes |
| DA_HB_HOUSTON | None | float64 | Yes |
| DA_HB_HUBAVG | None | float64 | Yes |
| DA_HB_NORTH | None | float64 | Yes |
| DA_HB_PAN | None | float64 | Yes |
| DA_HB_SOUTH | None | float64 | Yes |
| DA_HB_WEST | None | float64 | Yes |
| DA_LZ_AEN | None | float64 | Yes |
| DA_LZ_CPS | None | float64 | Yes |
| DA_LZ_HOUSTON | None | float64 | Yes |
| DA_LZ_LCRA | None | float64 | Yes |
| DA_LZ_NORTH | None | float64 | Yes |
| DA_LZ_RAYBN | None | float64 | Yes |
| DA_LZ_SOUTH | None | float64 | Yes |
| DA_LZ_WEST | None | float64 | Yes |
| RT_DC_E | None | float64 | Yes |
| RT_DC_L | None | float64 | Yes |
| RT_DC_N | None | float64 | Yes |
| RT_DC_R | None | float64 | Yes |
| RT_DC_S | None | float64 | Yes |
| RT_HB_BUSAVG | None | float64 | Yes |
| RT_HB_HOUSTON | None | float64 | Yes |
| RT_HB_HUBAVG | None | float64 | Yes |
| RT_HB_NORTH | None | float64 | Yes |
| RT_HB_PAN | None | float64 | Yes |
| RT_HB_SOUTH | None | float64 | Yes |
| RT_HB_WEST | None | float64 | Yes |
| RT_LZ_AEN | None | float64 | Yes |
| RT_LZ_CPS | None | float64 | Yes |
| RT_LZ_HOUSTON | None | float64 | Yes |
| RT_LZ_LCRA | None | float64 | Yes |
| RT_LZ_NORTH | None | float64 | Yes |
| RT_LZ_RAYBN | None | float64 | Yes |
| RT_LZ_SOUTH | None | float64 | Yes |
| RT_LZ_WEST | None | float64 | Yes |


### DA_AS_RT_5min_combined_2024_07.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_RT_5min_combined/DA_AS_RT_5min_combined_2024_07.parquet`

**Size**: 329.3 KB

**Rows**: 2,852

**Last Modified**: 2025-08-17T23:54:35.030164


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DeliveryInterval | None | int64 | Yes |
| AS_ECRS | None | float64 | Yes |
| AS_NSPIN | None | float64 | Yes |
| AS_REGDN | None | float64 | Yes |
| AS_REGUP | None | float64 | Yes |
| AS_RRS | None | float64 | Yes |
| DA_DC_E | None | float64 | Yes |
| DA_DC_L | None | float64 | Yes |
| DA_DC_N | None | float64 | Yes |
| DA_DC_R | None | float64 | Yes |
| DA_DC_S | None | float64 | Yes |
| DA_HB_BUSAVG | None | float64 | Yes |
| DA_HB_HOUSTON | None | float64 | Yes |
| DA_HB_HUBAVG | None | float64 | Yes |
| DA_HB_NORTH | None | float64 | Yes |
| DA_HB_PAN | None | float64 | Yes |
| DA_HB_SOUTH | None | float64 | Yes |
| DA_HB_WEST | None | float64 | Yes |
| DA_LZ_AEN | None | float64 | Yes |
| DA_LZ_CPS | None | float64 | Yes |
| DA_LZ_HOUSTON | None | float64 | Yes |
| DA_LZ_LCRA | None | float64 | Yes |
| DA_LZ_NORTH | None | float64 | Yes |
| DA_LZ_RAYBN | None | float64 | Yes |
| DA_LZ_SOUTH | None | float64 | Yes |
| DA_LZ_WEST | None | float64 | Yes |
| RT_DC_E | None | float64 | Yes |
| RT_DC_L | None | float64 | Yes |
| RT_DC_N | None | float64 | Yes |
| RT_DC_R | None | float64 | Yes |
| RT_DC_S | None | float64 | Yes |
| RT_HB_BUSAVG | None | float64 | Yes |
| RT_HB_HOUSTON | None | float64 | Yes |
| RT_HB_HUBAVG | None | float64 | Yes |
| RT_HB_NORTH | None | float64 | Yes |
| RT_HB_PAN | None | float64 | Yes |
| RT_HB_SOUTH | None | float64 | Yes |
| RT_HB_WEST | None | float64 | Yes |
| RT_LZ_AEN | None | float64 | Yes |
| RT_LZ_CPS | None | float64 | Yes |
| RT_LZ_HOUSTON | None | float64 | Yes |
| RT_LZ_LCRA | None | float64 | Yes |
| RT_LZ_NORTH | None | float64 | Yes |
| RT_LZ_RAYBN | None | float64 | Yes |
| RT_LZ_SOUTH | None | float64 | Yes |
| RT_LZ_WEST | None | float64 | Yes |


### DA_AS_RT_5min_combined_2024_08.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_RT_5min_combined/DA_AS_RT_5min_combined_2024_08.parquet`

**Size**: 338.6 KB

**Rows**: 2,852

**Last Modified**: 2025-08-17T23:54:35.034164


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DeliveryInterval | None | int64 | Yes |
| AS_ECRS | None | float64 | Yes |
| AS_NSPIN | None | float64 | Yes |
| AS_REGDN | None | float64 | Yes |
| AS_REGUP | None | float64 | Yes |
| AS_RRS | None | float64 | Yes |
| DA_DC_E | None | float64 | Yes |
| DA_DC_L | None | float64 | Yes |
| DA_DC_N | None | float64 | Yes |
| DA_DC_R | None | float64 | Yes |
| DA_DC_S | None | float64 | Yes |
| DA_HB_BUSAVG | None | float64 | Yes |
| DA_HB_HOUSTON | None | float64 | Yes |
| DA_HB_HUBAVG | None | float64 | Yes |
| DA_HB_NORTH | None | float64 | Yes |
| DA_HB_PAN | None | float64 | Yes |
| DA_HB_SOUTH | None | float64 | Yes |
| DA_HB_WEST | None | float64 | Yes |
| DA_LZ_AEN | None | float64 | Yes |
| DA_LZ_CPS | None | float64 | Yes |
| DA_LZ_HOUSTON | None | float64 | Yes |
| DA_LZ_LCRA | None | float64 | Yes |
| DA_LZ_NORTH | None | float64 | Yes |
| DA_LZ_RAYBN | None | float64 | Yes |
| DA_LZ_SOUTH | None | float64 | Yes |
| DA_LZ_WEST | None | float64 | Yes |
| RT_DC_E | None | float64 | Yes |
| RT_DC_L | None | float64 | Yes |
| RT_DC_N | None | float64 | Yes |
| RT_DC_R | None | float64 | Yes |
| RT_DC_S | None | float64 | Yes |
| RT_HB_BUSAVG | None | float64 | Yes |
| RT_HB_HOUSTON | None | float64 | Yes |
| RT_HB_HUBAVG | None | float64 | Yes |
| RT_HB_NORTH | None | float64 | Yes |
| RT_HB_PAN | None | float64 | Yes |
| RT_HB_SOUTH | None | float64 | Yes |
| RT_HB_WEST | None | float64 | Yes |
| RT_LZ_AEN | None | float64 | Yes |
| RT_LZ_CPS | None | float64 | Yes |
| RT_LZ_HOUSTON | None | float64 | Yes |
| RT_LZ_LCRA | None | float64 | Yes |
| RT_LZ_NORTH | None | float64 | Yes |
| RT_LZ_RAYBN | None | float64 | Yes |
| RT_LZ_SOUTH | None | float64 | Yes |
| RT_LZ_WEST | None | float64 | Yes |


### DA_AS_RT_5min_combined_2024_09.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_RT_5min_combined/DA_AS_RT_5min_combined_2024_09.parquet`

**Size**: 321.3 KB

**Rows**: 2,760

**Last Modified**: 2025-08-17T23:54:35.038164


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DeliveryInterval | None | int64 | Yes |
| AS_ECRS | None | float64 | Yes |
| AS_NSPIN | None | float64 | Yes |
| AS_REGDN | None | float64 | Yes |
| AS_REGUP | None | float64 | Yes |
| AS_RRS | None | float64 | Yes |
| DA_DC_E | None | float64 | Yes |
| DA_DC_L | None | float64 | Yes |
| DA_DC_N | None | float64 | Yes |
| DA_DC_R | None | float64 | Yes |
| DA_DC_S | None | float64 | Yes |
| DA_HB_BUSAVG | None | float64 | Yes |
| DA_HB_HOUSTON | None | float64 | Yes |
| DA_HB_HUBAVG | None | float64 | Yes |
| DA_HB_NORTH | None | float64 | Yes |
| DA_HB_PAN | None | float64 | Yes |
| DA_HB_SOUTH | None | float64 | Yes |
| DA_HB_WEST | None | float64 | Yes |
| DA_LZ_AEN | None | float64 | Yes |
| DA_LZ_CPS | None | float64 | Yes |
| DA_LZ_HOUSTON | None | float64 | Yes |
| DA_LZ_LCRA | None | float64 | Yes |
| DA_LZ_NORTH | None | float64 | Yes |
| DA_LZ_RAYBN | None | float64 | Yes |
| DA_LZ_SOUTH | None | float64 | Yes |
| DA_LZ_WEST | None | float64 | Yes |
| RT_DC_E | None | float64 | Yes |
| RT_DC_L | None | float64 | Yes |
| RT_DC_N | None | float64 | Yes |
| RT_DC_R | None | float64 | Yes |
| RT_DC_S | None | float64 | Yes |
| RT_HB_BUSAVG | None | float64 | Yes |
| RT_HB_HOUSTON | None | float64 | Yes |
| RT_HB_HUBAVG | None | float64 | Yes |
| RT_HB_NORTH | None | float64 | Yes |
| RT_HB_PAN | None | float64 | Yes |
| RT_HB_SOUTH | None | float64 | Yes |
| RT_HB_WEST | None | float64 | Yes |
| RT_LZ_AEN | None | float64 | Yes |
| RT_LZ_CPS | None | float64 | Yes |
| RT_LZ_HOUSTON | None | float64 | Yes |
| RT_LZ_LCRA | None | float64 | Yes |
| RT_LZ_NORTH | None | float64 | Yes |
| RT_LZ_RAYBN | None | float64 | Yes |
| RT_LZ_SOUTH | None | float64 | Yes |
| RT_LZ_WEST | None | float64 | Yes |


### DA_AS_RT_5min_combined_2024_10.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_RT_5min_combined/DA_AS_RT_5min_combined_2024_10.parquet`

**Size**: 345.4 KB

**Rows**: 2,852

**Last Modified**: 2025-08-17T23:54:35.043164


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DeliveryInterval | None | int64 | Yes |
| AS_ECRS | None | float64 | Yes |
| AS_NSPIN | None | float64 | Yes |
| AS_REGDN | None | float64 | Yes |
| AS_REGUP | None | float64 | Yes |
| AS_RRS | None | float64 | Yes |
| DA_DC_E | None | float64 | Yes |
| DA_DC_L | None | float64 | Yes |
| DA_DC_N | None | float64 | Yes |
| DA_DC_R | None | float64 | Yes |
| DA_DC_S | None | float64 | Yes |
| DA_HB_BUSAVG | None | float64 | Yes |
| DA_HB_HOUSTON | None | float64 | Yes |
| DA_HB_HUBAVG | None | float64 | Yes |
| DA_HB_NORTH | None | float64 | Yes |
| DA_HB_PAN | None | float64 | Yes |
| DA_HB_SOUTH | None | float64 | Yes |
| DA_HB_WEST | None | float64 | Yes |
| DA_LZ_AEN | None | float64 | Yes |
| DA_LZ_CPS | None | float64 | Yes |
| DA_LZ_HOUSTON | None | float64 | Yes |
| DA_LZ_LCRA | None | float64 | Yes |
| DA_LZ_NORTH | None | float64 | Yes |
| DA_LZ_RAYBN | None | float64 | Yes |
| DA_LZ_SOUTH | None | float64 | Yes |
| DA_LZ_WEST | None | float64 | Yes |
| RT_DC_E | None | float64 | Yes |
| RT_DC_L | None | float64 | Yes |
| RT_DC_N | None | float64 | Yes |
| RT_DC_R | None | float64 | Yes |
| RT_DC_S | None | float64 | Yes |
| RT_HB_BUSAVG | None | float64 | Yes |
| RT_HB_HOUSTON | None | float64 | Yes |
| RT_HB_HUBAVG | None | float64 | Yes |
| RT_HB_NORTH | None | float64 | Yes |
| RT_HB_PAN | None | float64 | Yes |
| RT_HB_SOUTH | None | float64 | Yes |
| RT_HB_WEST | None | float64 | Yes |
| RT_LZ_AEN | None | float64 | Yes |
| RT_LZ_CPS | None | float64 | Yes |
| RT_LZ_HOUSTON | None | float64 | Yes |
| RT_LZ_LCRA | None | float64 | Yes |
| RT_LZ_NORTH | None | float64 | Yes |
| RT_LZ_RAYBN | None | float64 | Yes |
| RT_LZ_SOUTH | None | float64 | Yes |
| RT_LZ_WEST | None | float64 | Yes |


### DA_AS_RT_5min_combined_2024_11.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_RT_5min_combined/DA_AS_RT_5min_combined_2024_11.parquet`

**Size**: 347.9 KB

**Rows**: 2,760

**Last Modified**: 2025-08-17T23:54:35.047164


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DeliveryInterval | None | int64 | Yes |
| AS_ECRS | None | float64 | Yes |
| AS_NSPIN | None | float64 | Yes |
| AS_REGDN | None | float64 | Yes |
| AS_REGUP | None | float64 | Yes |
| AS_RRS | None | float64 | Yes |
| DA_DC_E | None | float64 | Yes |
| DA_DC_L | None | float64 | Yes |
| DA_DC_N | None | float64 | Yes |
| DA_DC_R | None | float64 | Yes |
| DA_DC_S | None | float64 | Yes |
| DA_HB_BUSAVG | None | float64 | Yes |
| DA_HB_HOUSTON | None | float64 | Yes |
| DA_HB_HUBAVG | None | float64 | Yes |
| DA_HB_NORTH | None | float64 | Yes |
| DA_HB_PAN | None | float64 | Yes |
| DA_HB_SOUTH | None | float64 | Yes |
| DA_HB_WEST | None | float64 | Yes |
| DA_LZ_AEN | None | float64 | Yes |
| DA_LZ_CPS | None | float64 | Yes |
| DA_LZ_HOUSTON | None | float64 | Yes |
| DA_LZ_LCRA | None | float64 | Yes |
| DA_LZ_NORTH | None | float64 | Yes |
| DA_LZ_RAYBN | None | float64 | Yes |
| DA_LZ_SOUTH | None | float64 | Yes |
| DA_LZ_WEST | None | float64 | Yes |
| RT_DC_E | None | float64 | Yes |
| RT_DC_L | None | float64 | Yes |
| RT_DC_N | None | float64 | Yes |
| RT_DC_R | None | float64 | Yes |
| RT_DC_S | None | float64 | Yes |
| RT_HB_BUSAVG | None | float64 | Yes |
| RT_HB_HOUSTON | None | float64 | Yes |
| RT_HB_HUBAVG | None | float64 | Yes |
| RT_HB_NORTH | None | float64 | Yes |
| RT_HB_PAN | None | float64 | Yes |
| RT_HB_SOUTH | None | float64 | Yes |
| RT_HB_WEST | None | float64 | Yes |
| RT_LZ_AEN | None | float64 | Yes |
| RT_LZ_CPS | None | float64 | Yes |
| RT_LZ_HOUSTON | None | float64 | Yes |
| RT_LZ_LCRA | None | float64 | Yes |
| RT_LZ_NORTH | None | float64 | Yes |
| RT_LZ_RAYBN | None | float64 | Yes |
| RT_LZ_SOUTH | None | float64 | Yes |
| RT_LZ_WEST | None | float64 | Yes |


### DA_AS_RT_5min_combined_2024_12.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_RT_5min_combined/DA_AS_RT_5min_combined_2024_12.parquet`

**Size**: 349.9 KB

**Rows**: 2,852

**Last Modified**: 2025-08-17T23:54:35.051165


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DeliveryInterval | None | int64 | Yes |
| AS_ECRS | None | float64 | Yes |
| AS_NSPIN | None | float64 | Yes |
| AS_REGDN | None | float64 | Yes |
| AS_REGUP | None | float64 | Yes |
| AS_RRS | None | float64 | Yes |
| DA_DC_E | None | float64 | Yes |
| DA_DC_L | None | float64 | Yes |
| DA_DC_N | None | float64 | Yes |
| DA_DC_R | None | float64 | Yes |
| DA_DC_S | None | float64 | Yes |
| DA_HB_BUSAVG | None | float64 | Yes |
| DA_HB_HOUSTON | None | float64 | Yes |
| DA_HB_HUBAVG | None | float64 | Yes |
| DA_HB_NORTH | None | float64 | Yes |
| DA_HB_PAN | None | float64 | Yes |
| DA_HB_SOUTH | None | float64 | Yes |
| DA_HB_WEST | None | float64 | Yes |
| DA_LZ_AEN | None | float64 | Yes |
| DA_LZ_CPS | None | float64 | Yes |
| DA_LZ_HOUSTON | None | float64 | Yes |
| DA_LZ_LCRA | None | float64 | Yes |
| DA_LZ_NORTH | None | float64 | Yes |
| DA_LZ_RAYBN | None | float64 | Yes |
| DA_LZ_SOUTH | None | float64 | Yes |
| DA_LZ_WEST | None | float64 | Yes |
| RT_DC_E | None | float64 | Yes |
| RT_DC_L | None | float64 | Yes |
| RT_DC_N | None | float64 | Yes |
| RT_DC_R | None | float64 | Yes |
| RT_DC_S | None | float64 | Yes |
| RT_HB_BUSAVG | None | float64 | Yes |
| RT_HB_HOUSTON | None | float64 | Yes |
| RT_HB_HUBAVG | None | float64 | Yes |
| RT_HB_NORTH | None | float64 | Yes |
| RT_HB_PAN | None | float64 | Yes |
| RT_HB_SOUTH | None | float64 | Yes |
| RT_HB_WEST | None | float64 | Yes |
| RT_LZ_AEN | None | float64 | Yes |
| RT_LZ_CPS | None | float64 | Yes |
| RT_LZ_HOUSTON | None | float64 | Yes |
| RT_LZ_LCRA | None | float64 | Yes |
| RT_LZ_NORTH | None | float64 | Yes |
| RT_LZ_RAYBN | None | float64 | Yes |
| RT_LZ_SOUTH | None | float64 | Yes |
| RT_LZ_WEST | None | float64 | Yes |


### DA_AS_RT_combined_2023_01.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_RT_combined/DA_AS_RT_combined_2023_01.parquet`

**Size**: 204.5 KB

**Rows**: 743

**Last Modified**: 2025-08-18T16:05:23.448077


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DA_DC_E | None | float64 | Yes |
| DA_DC_L | None | float64 | Yes |
| DA_DC_N | None | float64 | Yes |
| DA_DC_R | None | float64 | Yes |
| DA_DC_S | None | float64 | Yes |
| DA_HB_BUSAVG | None | float64 | Yes |
| DA_HB_HOUSTON | None | float64 | Yes |
| DA_HB_HUBAVG | None | float64 | Yes |
| DA_HB_NORTH | None | float64 | Yes |
| DA_HB_PAN | None | float64 | Yes |
| DA_HB_SOUTH | None | float64 | Yes |
| DA_HB_WEST | None | float64 | Yes |
| DA_LZ_AEN | None | float64 | Yes |
| DA_LZ_CPS | None | float64 | Yes |
| DA_LZ_HOUSTON | None | float64 | Yes |
| DA_LZ_LCRA | None | float64 | Yes |
| DA_LZ_NORTH | None | float64 | Yes |
| DA_LZ_RAYBN | None | float64 | Yes |
| DA_LZ_SOUTH | None | float64 | Yes |
| DA_LZ_WEST | None | float64 | Yes |
| AS_ECRS | None | float64 | Yes |
| AS_NSPIN | None | float64 | Yes |
| AS_REGDN | None | float64 | Yes |
| AS_REGUP | None | float64 | Yes |
| AS_RRS | None | float64 | Yes |
| RT_DC_E | None | float64 | Yes |
| RT_DC_L | None | float64 | Yes |
| RT_DC_N | None | float64 | Yes |
| RT_DC_R | None | float64 | Yes |
| RT_DC_S | None | float64 | Yes |
| RT_HB_BUSAVG | None | float64 | Yes |
| RT_HB_HOUSTON | None | float64 | Yes |
| RT_HB_HUBAVG | None | float64 | Yes |
| RT_HB_NORTH | None | float64 | Yes |
| RT_HB_PAN | None | float64 | Yes |
| RT_HB_SOUTH | None | float64 | Yes |
| RT_HB_WEST | None | float64 | Yes |
| RT_LZ_AEN | None | float64 | Yes |
| RT_LZ_CPS | None | float64 | Yes |
| RT_LZ_HOUSTON | None | float64 | Yes |
| RT_LZ_LCRA | None | float64 | Yes |
| RT_LZ_NORTH | None | float64 | Yes |
| RT_LZ_RAYBN | None | float64 | Yes |
| RT_LZ_SOUTH | None | float64 | Yes |
| RT_LZ_WEST | None | float64 | Yes |


### DA_AS_RT_combined_2023_02.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_RT_combined/DA_AS_RT_combined_2023_02.parquet`

**Size**: 190.9 KB

**Rows**: 672

**Last Modified**: 2025-08-18T16:05:23.451077


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DA_DC_E | None | float64 | Yes |
| DA_DC_L | None | float64 | Yes |
| DA_DC_N | None | float64 | Yes |
| DA_DC_R | None | float64 | Yes |
| DA_DC_S | None | float64 | Yes |
| DA_HB_BUSAVG | None | float64 | Yes |
| DA_HB_HOUSTON | None | float64 | Yes |
| DA_HB_HUBAVG | None | float64 | Yes |
| DA_HB_NORTH | None | float64 | Yes |
| DA_HB_PAN | None | float64 | Yes |
| DA_HB_SOUTH | None | float64 | Yes |
| DA_HB_WEST | None | float64 | Yes |
| DA_LZ_AEN | None | float64 | Yes |
| DA_LZ_CPS | None | float64 | Yes |
| DA_LZ_HOUSTON | None | float64 | Yes |
| DA_LZ_LCRA | None | float64 | Yes |
| DA_LZ_NORTH | None | float64 | Yes |
| DA_LZ_RAYBN | None | float64 | Yes |
| DA_LZ_SOUTH | None | float64 | Yes |
| DA_LZ_WEST | None | float64 | Yes |
| AS_ECRS | None | float64 | Yes |
| AS_NSPIN | None | float64 | Yes |
| AS_REGDN | None | float64 | Yes |
| AS_REGUP | None | float64 | Yes |
| AS_RRS | None | float64 | Yes |
| RT_DC_E | None | float64 | Yes |
| RT_DC_L | None | float64 | Yes |
| RT_DC_N | None | float64 | Yes |
| RT_DC_R | None | float64 | Yes |
| RT_DC_S | None | float64 | Yes |
| RT_HB_BUSAVG | None | float64 | Yes |
| RT_HB_HOUSTON | None | float64 | Yes |
| RT_HB_HUBAVG | None | float64 | Yes |
| RT_HB_NORTH | None | float64 | Yes |
| RT_HB_PAN | None | float64 | Yes |
| RT_HB_SOUTH | None | float64 | Yes |
| RT_HB_WEST | None | float64 | Yes |
| RT_LZ_AEN | None | float64 | Yes |
| RT_LZ_CPS | None | float64 | Yes |
| RT_LZ_HOUSTON | None | float64 | Yes |
| RT_LZ_LCRA | None | float64 | Yes |
| RT_LZ_NORTH | None | float64 | Yes |
| RT_LZ_RAYBN | None | float64 | Yes |
| RT_LZ_SOUTH | None | float64 | Yes |
| RT_LZ_WEST | None | float64 | Yes |


### DA_AS_RT_combined_2023_03.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_RT_combined/DA_AS_RT_combined_2023_03.parquet`

**Size**: 208.9 KB

**Rows**: 743

**Last Modified**: 2025-08-18T16:05:23.454077


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DA_DC_E | None | float64 | Yes |
| DA_DC_L | None | float64 | Yes |
| DA_DC_N | None | float64 | Yes |
| DA_DC_R | None | float64 | Yes |
| DA_DC_S | None | float64 | Yes |
| DA_HB_BUSAVG | None | float64 | Yes |
| DA_HB_HOUSTON | None | float64 | Yes |
| DA_HB_HUBAVG | None | float64 | Yes |
| DA_HB_NORTH | None | float64 | Yes |
| DA_HB_PAN | None | float64 | Yes |
| DA_HB_SOUTH | None | float64 | Yes |
| DA_HB_WEST | None | float64 | Yes |
| DA_LZ_AEN | None | float64 | Yes |
| DA_LZ_CPS | None | float64 | Yes |
| DA_LZ_HOUSTON | None | float64 | Yes |
| DA_LZ_LCRA | None | float64 | Yes |
| DA_LZ_NORTH | None | float64 | Yes |
| DA_LZ_RAYBN | None | float64 | Yes |
| DA_LZ_SOUTH | None | float64 | Yes |
| DA_LZ_WEST | None | float64 | Yes |
| AS_ECRS | None | float64 | Yes |
| AS_NSPIN | None | float64 | Yes |
| AS_REGDN | None | float64 | Yes |
| AS_REGUP | None | float64 | Yes |
| AS_RRS | None | float64 | Yes |
| RT_DC_E | None | float64 | Yes |
| RT_DC_L | None | float64 | Yes |
| RT_DC_N | None | float64 | Yes |
| RT_DC_R | None | float64 | Yes |
| RT_DC_S | None | float64 | Yes |
| RT_HB_BUSAVG | None | float64 | Yes |
| RT_HB_HOUSTON | None | float64 | Yes |
| RT_HB_HUBAVG | None | float64 | Yes |
| RT_HB_NORTH | None | float64 | Yes |
| RT_HB_PAN | None | float64 | Yes |
| RT_HB_SOUTH | None | float64 | Yes |
| RT_HB_WEST | None | float64 | Yes |
| RT_LZ_AEN | None | float64 | Yes |
| RT_LZ_CPS | None | float64 | Yes |
| RT_LZ_HOUSTON | None | float64 | Yes |
| RT_LZ_LCRA | None | float64 | Yes |
| RT_LZ_NORTH | None | float64 | Yes |
| RT_LZ_RAYBN | None | float64 | Yes |
| RT_LZ_SOUTH | None | float64 | Yes |
| RT_LZ_WEST | None | float64 | Yes |


### DA_AS_RT_combined_2023_04.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_RT_combined/DA_AS_RT_combined_2023_04.parquet`

**Size**: 202.8 KB

**Rows**: 720

**Last Modified**: 2025-08-18T16:05:23.457077


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DA_DC_E | None | float64 | Yes |
| DA_DC_L | None | float64 | Yes |
| DA_DC_N | None | float64 | Yes |
| DA_DC_R | None | float64 | Yes |
| DA_DC_S | None | float64 | Yes |
| DA_HB_BUSAVG | None | float64 | Yes |
| DA_HB_HOUSTON | None | float64 | Yes |
| DA_HB_HUBAVG | None | float64 | Yes |
| DA_HB_NORTH | None | float64 | Yes |
| DA_HB_PAN | None | float64 | Yes |
| DA_HB_SOUTH | None | float64 | Yes |
| DA_HB_WEST | None | float64 | Yes |
| DA_LZ_AEN | None | float64 | Yes |
| DA_LZ_CPS | None | float64 | Yes |
| DA_LZ_HOUSTON | None | float64 | Yes |
| DA_LZ_LCRA | None | float64 | Yes |
| DA_LZ_NORTH | None | float64 | Yes |
| DA_LZ_RAYBN | None | float64 | Yes |
| DA_LZ_SOUTH | None | float64 | Yes |
| DA_LZ_WEST | None | float64 | Yes |
| AS_ECRS | None | float64 | Yes |
| AS_NSPIN | None | float64 | Yes |
| AS_REGDN | None | float64 | Yes |
| AS_REGUP | None | float64 | Yes |
| AS_RRS | None | float64 | Yes |
| RT_DC_E | None | float64 | Yes |
| RT_DC_L | None | float64 | Yes |
| RT_DC_N | None | float64 | Yes |
| RT_DC_R | None | float64 | Yes |
| RT_DC_S | None | float64 | Yes |
| RT_HB_BUSAVG | None | float64 | Yes |
| RT_HB_HOUSTON | None | float64 | Yes |
| RT_HB_HUBAVG | None | float64 | Yes |
| RT_HB_NORTH | None | float64 | Yes |
| RT_HB_PAN | None | float64 | Yes |
| RT_HB_SOUTH | None | float64 | Yes |
| RT_HB_WEST | None | float64 | Yes |
| RT_LZ_AEN | None | float64 | Yes |
| RT_LZ_CPS | None | float64 | Yes |
| RT_LZ_HOUSTON | None | float64 | Yes |
| RT_LZ_LCRA | None | float64 | Yes |
| RT_LZ_NORTH | None | float64 | Yes |
| RT_LZ_RAYBN | None | float64 | Yes |
| RT_LZ_SOUTH | None | float64 | Yes |
| RT_LZ_WEST | None | float64 | Yes |


### DA_AS_RT_combined_2023_05.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_RT_combined/DA_AS_RT_combined_2023_05.parquet`

**Size**: 205.1 KB

**Rows**: 744

**Last Modified**: 2025-08-18T16:05:23.460078


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DA_DC_E | None | float64 | Yes |
| DA_DC_L | None | float64 | Yes |
| DA_DC_N | None | float64 | Yes |
| DA_DC_R | None | float64 | Yes |
| DA_DC_S | None | float64 | Yes |
| DA_HB_BUSAVG | None | float64 | Yes |
| DA_HB_HOUSTON | None | float64 | Yes |
| DA_HB_HUBAVG | None | float64 | Yes |
| DA_HB_NORTH | None | float64 | Yes |
| DA_HB_PAN | None | float64 | Yes |
| DA_HB_SOUTH | None | float64 | Yes |
| DA_HB_WEST | None | float64 | Yes |
| DA_LZ_AEN | None | float64 | Yes |
| DA_LZ_CPS | None | float64 | Yes |
| DA_LZ_HOUSTON | None | float64 | Yes |
| DA_LZ_LCRA | None | float64 | Yes |
| DA_LZ_NORTH | None | float64 | Yes |
| DA_LZ_RAYBN | None | float64 | Yes |
| DA_LZ_SOUTH | None | float64 | Yes |
| DA_LZ_WEST | None | float64 | Yes |
| AS_ECRS | None | float64 | Yes |
| AS_NSPIN | None | float64 | Yes |
| AS_REGDN | None | float64 | Yes |
| AS_REGUP | None | float64 | Yes |
| AS_RRS | None | float64 | Yes |
| RT_DC_E | None | float64 | Yes |
| RT_DC_L | None | float64 | Yes |
| RT_DC_N | None | float64 | Yes |
| RT_DC_R | None | float64 | Yes |
| RT_DC_S | None | float64 | Yes |
| RT_HB_BUSAVG | None | float64 | Yes |
| RT_HB_HOUSTON | None | float64 | Yes |
| RT_HB_HUBAVG | None | float64 | Yes |
| RT_HB_NORTH | None | float64 | Yes |
| RT_HB_PAN | None | float64 | Yes |
| RT_HB_SOUTH | None | float64 | Yes |
| RT_HB_WEST | None | float64 | Yes |
| RT_LZ_AEN | None | float64 | Yes |
| RT_LZ_CPS | None | float64 | Yes |
| RT_LZ_HOUSTON | None | float64 | Yes |
| RT_LZ_LCRA | None | float64 | Yes |
| RT_LZ_NORTH | None | float64 | Yes |
| RT_LZ_RAYBN | None | float64 | Yes |
| RT_LZ_SOUTH | None | float64 | Yes |
| RT_LZ_WEST | None | float64 | Yes |


### DA_AS_RT_combined_2023_06.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_RT_combined/DA_AS_RT_combined_2023_06.parquet`

**Size**: 206.3 KB

**Rows**: 720

**Last Modified**: 2025-08-18T16:05:23.463078


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DA_DC_E | None | float64 | Yes |
| DA_DC_L | None | float64 | Yes |
| DA_DC_N | None | float64 | Yes |
| DA_DC_R | None | float64 | Yes |
| DA_DC_S | None | float64 | Yes |
| DA_HB_BUSAVG | None | float64 | Yes |
| DA_HB_HOUSTON | None | float64 | Yes |
| DA_HB_HUBAVG | None | float64 | Yes |
| DA_HB_NORTH | None | float64 | Yes |
| DA_HB_PAN | None | float64 | Yes |
| DA_HB_SOUTH | None | float64 | Yes |
| DA_HB_WEST | None | float64 | Yes |
| DA_LZ_AEN | None | float64 | Yes |
| DA_LZ_CPS | None | float64 | Yes |
| DA_LZ_HOUSTON | None | float64 | Yes |
| DA_LZ_LCRA | None | float64 | Yes |
| DA_LZ_NORTH | None | float64 | Yes |
| DA_LZ_RAYBN | None | float64 | Yes |
| DA_LZ_SOUTH | None | float64 | Yes |
| DA_LZ_WEST | None | float64 | Yes |
| AS_ECRS | None | float64 | Yes |
| AS_NSPIN | None | float64 | Yes |
| AS_REGDN | None | float64 | Yes |
| AS_REGUP | None | float64 | Yes |
| AS_RRS | None | float64 | Yes |
| RT_DC_E | None | float64 | Yes |
| RT_DC_L | None | float64 | Yes |
| RT_DC_N | None | float64 | Yes |
| RT_DC_R | None | float64 | Yes |
| RT_DC_S | None | float64 | Yes |
| RT_HB_BUSAVG | None | float64 | Yes |
| RT_HB_HOUSTON | None | float64 | Yes |
| RT_HB_HUBAVG | None | float64 | Yes |
| RT_HB_NORTH | None | float64 | Yes |
| RT_HB_PAN | None | float64 | Yes |
| RT_HB_SOUTH | None | float64 | Yes |
| RT_HB_WEST | None | float64 | Yes |
| RT_LZ_AEN | None | float64 | Yes |
| RT_LZ_CPS | None | float64 | Yes |
| RT_LZ_HOUSTON | None | float64 | Yes |
| RT_LZ_LCRA | None | float64 | Yes |
| RT_LZ_NORTH | None | float64 | Yes |
| RT_LZ_RAYBN | None | float64 | Yes |
| RT_LZ_SOUTH | None | float64 | Yes |
| RT_LZ_WEST | None | float64 | Yes |


### DA_AS_RT_combined_2023_07.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_RT_combined/DA_AS_RT_combined_2023_07.parquet`

**Size**: 205.8 KB

**Rows**: 744

**Last Modified**: 2025-08-18T16:05:23.466078


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DA_DC_E | None | float64 | Yes |
| DA_DC_L | None | float64 | Yes |
| DA_DC_N | None | float64 | Yes |
| DA_DC_R | None | float64 | Yes |
| DA_DC_S | None | float64 | Yes |
| DA_HB_BUSAVG | None | float64 | Yes |
| DA_HB_HOUSTON | None | float64 | Yes |
| DA_HB_HUBAVG | None | float64 | Yes |
| DA_HB_NORTH | None | float64 | Yes |
| DA_HB_PAN | None | float64 | Yes |
| DA_HB_SOUTH | None | float64 | Yes |
| DA_HB_WEST | None | float64 | Yes |
| DA_LZ_AEN | None | float64 | Yes |
| DA_LZ_CPS | None | float64 | Yes |
| DA_LZ_HOUSTON | None | float64 | Yes |
| DA_LZ_LCRA | None | float64 | Yes |
| DA_LZ_NORTH | None | float64 | Yes |
| DA_LZ_RAYBN | None | float64 | Yes |
| DA_LZ_SOUTH | None | float64 | Yes |
| DA_LZ_WEST | None | float64 | Yes |
| AS_ECRS | None | float64 | Yes |
| AS_NSPIN | None | float64 | Yes |
| AS_REGDN | None | float64 | Yes |
| AS_REGUP | None | float64 | Yes |
| AS_RRS | None | float64 | Yes |
| RT_DC_E | None | float64 | Yes |
| RT_DC_L | None | float64 | Yes |
| RT_DC_N | None | float64 | Yes |
| RT_DC_R | None | float64 | Yes |
| RT_DC_S | None | float64 | Yes |
| RT_HB_BUSAVG | None | float64 | Yes |
| RT_HB_HOUSTON | None | float64 | Yes |
| RT_HB_HUBAVG | None | float64 | Yes |
| RT_HB_NORTH | None | float64 | Yes |
| RT_HB_PAN | None | float64 | Yes |
| RT_HB_SOUTH | None | float64 | Yes |
| RT_HB_WEST | None | float64 | Yes |
| RT_LZ_AEN | None | float64 | Yes |
| RT_LZ_CPS | None | float64 | Yes |
| RT_LZ_HOUSTON | None | float64 | Yes |
| RT_LZ_LCRA | None | float64 | Yes |
| RT_LZ_NORTH | None | float64 | Yes |
| RT_LZ_RAYBN | None | float64 | Yes |
| RT_LZ_SOUTH | None | float64 | Yes |
| RT_LZ_WEST | None | float64 | Yes |


### DA_AS_RT_combined_2023_08.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_RT_combined/DA_AS_RT_combined_2023_08.parquet`

**Size**: 211.9 KB

**Rows**: 744

**Last Modified**: 2025-08-18T16:05:23.470078


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DA_DC_E | None | float64 | Yes |
| DA_DC_L | None | float64 | Yes |
| DA_DC_N | None | float64 | Yes |
| DA_DC_R | None | float64 | Yes |
| DA_DC_S | None | float64 | Yes |
| DA_HB_BUSAVG | None | float64 | Yes |
| DA_HB_HOUSTON | None | float64 | Yes |
| DA_HB_HUBAVG | None | float64 | Yes |
| DA_HB_NORTH | None | float64 | Yes |
| DA_HB_PAN | None | float64 | Yes |
| DA_HB_SOUTH | None | float64 | Yes |
| DA_HB_WEST | None | float64 | Yes |
| DA_LZ_AEN | None | float64 | Yes |
| DA_LZ_CPS | None | float64 | Yes |
| DA_LZ_HOUSTON | None | float64 | Yes |
| DA_LZ_LCRA | None | float64 | Yes |
| DA_LZ_NORTH | None | float64 | Yes |
| DA_LZ_RAYBN | None | float64 | Yes |
| DA_LZ_SOUTH | None | float64 | Yes |
| DA_LZ_WEST | None | float64 | Yes |
| AS_ECRS | None | float64 | Yes |
| AS_NSPIN | None | float64 | Yes |
| AS_REGDN | None | float64 | Yes |
| AS_REGUP | None | float64 | Yes |
| AS_RRS | None | float64 | Yes |
| RT_DC_E | None | float64 | Yes |
| RT_DC_L | None | float64 | Yes |
| RT_DC_N | None | float64 | Yes |
| RT_DC_R | None | float64 | Yes |
| RT_DC_S | None | float64 | Yes |
| RT_HB_BUSAVG | None | float64 | Yes |
| RT_HB_HOUSTON | None | float64 | Yes |
| RT_HB_HUBAVG | None | float64 | Yes |
| RT_HB_NORTH | None | float64 | Yes |
| RT_HB_PAN | None | float64 | Yes |
| RT_HB_SOUTH | None | float64 | Yes |
| RT_HB_WEST | None | float64 | Yes |
| RT_LZ_AEN | None | float64 | Yes |
| RT_LZ_CPS | None | float64 | Yes |
| RT_LZ_HOUSTON | None | float64 | Yes |
| RT_LZ_LCRA | None | float64 | Yes |
| RT_LZ_NORTH | None | float64 | Yes |
| RT_LZ_RAYBN | None | float64 | Yes |
| RT_LZ_SOUTH | None | float64 | Yes |
| RT_LZ_WEST | None | float64 | Yes |


### DA_AS_RT_combined_2023_09.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_RT_combined/DA_AS_RT_combined_2023_09.parquet`

**Size**: 199.3 KB

**Rows**: 720

**Last Modified**: 2025-08-18T16:05:23.473078


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DA_DC_E | None | float64 | Yes |
| DA_DC_L | None | float64 | Yes |
| DA_DC_N | None | float64 | Yes |
| DA_DC_R | None | float64 | Yes |
| DA_DC_S | None | float64 | Yes |
| DA_HB_BUSAVG | None | float64 | Yes |
| DA_HB_HOUSTON | None | float64 | Yes |
| DA_HB_HUBAVG | None | float64 | Yes |
| DA_HB_NORTH | None | float64 | Yes |
| DA_HB_PAN | None | float64 | Yes |
| DA_HB_SOUTH | None | float64 | Yes |
| DA_HB_WEST | None | float64 | Yes |
| DA_LZ_AEN | None | float64 | Yes |
| DA_LZ_CPS | None | float64 | Yes |
| DA_LZ_HOUSTON | None | float64 | Yes |
| DA_LZ_LCRA | None | float64 | Yes |
| DA_LZ_NORTH | None | float64 | Yes |
| DA_LZ_RAYBN | None | float64 | Yes |
| DA_LZ_SOUTH | None | float64 | Yes |
| DA_LZ_WEST | None | float64 | Yes |
| AS_ECRS | None | float64 | Yes |
| AS_NSPIN | None | float64 | Yes |
| AS_REGDN | None | float64 | Yes |
| AS_REGUP | None | float64 | Yes |
| AS_RRS | None | float64 | Yes |
| RT_DC_E | None | float64 | Yes |
| RT_DC_L | None | float64 | Yes |
| RT_DC_N | None | float64 | Yes |
| RT_DC_R | None | float64 | Yes |
| RT_DC_S | None | float64 | Yes |
| RT_HB_BUSAVG | None | float64 | Yes |
| RT_HB_HOUSTON | None | float64 | Yes |
| RT_HB_HUBAVG | None | float64 | Yes |
| RT_HB_NORTH | None | float64 | Yes |
| RT_HB_PAN | None | float64 | Yes |
| RT_HB_SOUTH | None | float64 | Yes |
| RT_HB_WEST | None | float64 | Yes |
| RT_LZ_AEN | None | float64 | Yes |
| RT_LZ_CPS | None | float64 | Yes |
| RT_LZ_HOUSTON | None | float64 | Yes |
| RT_LZ_LCRA | None | float64 | Yes |
| RT_LZ_NORTH | None | float64 | Yes |
| RT_LZ_RAYBN | None | float64 | Yes |
| RT_LZ_SOUTH | None | float64 | Yes |
| RT_LZ_WEST | None | float64 | Yes |


### DA_AS_RT_combined_2023_10.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_RT_combined/DA_AS_RT_combined_2023_10.parquet`

**Size**: 201.1 KB

**Rows**: 744

**Last Modified**: 2025-08-18T16:05:23.476078


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DA_DC_E | None | float64 | Yes |
| DA_DC_L | None | float64 | Yes |
| DA_DC_N | None | float64 | Yes |
| DA_DC_R | None | float64 | Yes |
| DA_DC_S | None | float64 | Yes |
| DA_HB_BUSAVG | None | float64 | Yes |
| DA_HB_HOUSTON | None | float64 | Yes |
| DA_HB_HUBAVG | None | float64 | Yes |
| DA_HB_NORTH | None | float64 | Yes |
| DA_HB_PAN | None | float64 | Yes |
| DA_HB_SOUTH | None | float64 | Yes |
| DA_HB_WEST | None | float64 | Yes |
| DA_LZ_AEN | None | float64 | Yes |
| DA_LZ_CPS | None | float64 | Yes |
| DA_LZ_HOUSTON | None | float64 | Yes |
| DA_LZ_LCRA | None | float64 | Yes |
| DA_LZ_NORTH | None | float64 | Yes |
| DA_LZ_RAYBN | None | float64 | Yes |
| DA_LZ_SOUTH | None | float64 | Yes |
| DA_LZ_WEST | None | float64 | Yes |
| AS_ECRS | None | float64 | Yes |
| AS_NSPIN | None | float64 | Yes |
| AS_REGDN | None | float64 | Yes |
| AS_REGUP | None | float64 | Yes |
| AS_RRS | None | float64 | Yes |
| RT_DC_E | None | float64 | Yes |
| RT_DC_L | None | float64 | Yes |
| RT_DC_N | None | float64 | Yes |
| RT_DC_R | None | float64 | Yes |
| RT_DC_S | None | float64 | Yes |
| RT_HB_BUSAVG | None | float64 | Yes |
| RT_HB_HOUSTON | None | float64 | Yes |
| RT_HB_HUBAVG | None | float64 | Yes |
| RT_HB_NORTH | None | float64 | Yes |
| RT_HB_PAN | None | float64 | Yes |
| RT_HB_SOUTH | None | float64 | Yes |
| RT_HB_WEST | None | float64 | Yes |
| RT_LZ_AEN | None | float64 | Yes |
| RT_LZ_CPS | None | float64 | Yes |
| RT_LZ_HOUSTON | None | float64 | Yes |
| RT_LZ_LCRA | None | float64 | Yes |
| RT_LZ_NORTH | None | float64 | Yes |
| RT_LZ_RAYBN | None | float64 | Yes |
| RT_LZ_SOUTH | None | float64 | Yes |
| RT_LZ_WEST | None | float64 | Yes |


### DA_AS_RT_combined_2023_11.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_RT_combined/DA_AS_RT_combined_2023_11.parquet`

**Size**: 195.9 KB

**Rows**: 720

**Last Modified**: 2025-08-18T16:05:23.479078


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DA_DC_E | None | float64 | Yes |
| DA_DC_L | None | float64 | Yes |
| DA_DC_N | None | float64 | Yes |
| DA_DC_R | None | float64 | Yes |
| DA_DC_S | None | float64 | Yes |
| DA_HB_BUSAVG | None | float64 | Yes |
| DA_HB_HOUSTON | None | float64 | Yes |
| DA_HB_HUBAVG | None | float64 | Yes |
| DA_HB_NORTH | None | float64 | Yes |
| DA_HB_PAN | None | float64 | Yes |
| DA_HB_SOUTH | None | float64 | Yes |
| DA_HB_WEST | None | float64 | Yes |
| DA_LZ_AEN | None | float64 | Yes |
| DA_LZ_CPS | None | float64 | Yes |
| DA_LZ_HOUSTON | None | float64 | Yes |
| DA_LZ_LCRA | None | float64 | Yes |
| DA_LZ_NORTH | None | float64 | Yes |
| DA_LZ_RAYBN | None | float64 | Yes |
| DA_LZ_SOUTH | None | float64 | Yes |
| DA_LZ_WEST | None | float64 | Yes |
| AS_ECRS | None | float64 | Yes |
| AS_NSPIN | None | float64 | Yes |
| AS_REGDN | None | float64 | Yes |
| AS_REGUP | None | float64 | Yes |
| AS_RRS | None | float64 | Yes |
| RT_DC_E | None | float64 | Yes |
| RT_DC_L | None | float64 | Yes |
| RT_DC_N | None | float64 | Yes |
| RT_DC_R | None | float64 | Yes |
| RT_DC_S | None | float64 | Yes |
| RT_HB_BUSAVG | None | float64 | Yes |
| RT_HB_HOUSTON | None | float64 | Yes |
| RT_HB_HUBAVG | None | float64 | Yes |
| RT_HB_NORTH | None | float64 | Yes |
| RT_HB_PAN | None | float64 | Yes |
| RT_HB_SOUTH | None | float64 | Yes |
| RT_HB_WEST | None | float64 | Yes |
| RT_LZ_AEN | None | float64 | Yes |
| RT_LZ_CPS | None | float64 | Yes |
| RT_LZ_HOUSTON | None | float64 | Yes |
| RT_LZ_LCRA | None | float64 | Yes |
| RT_LZ_NORTH | None | float64 | Yes |
| RT_LZ_RAYBN | None | float64 | Yes |
| RT_LZ_SOUTH | None | float64 | Yes |
| RT_LZ_WEST | None | float64 | Yes |


### DA_AS_RT_combined_2023_12.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_RT_combined/DA_AS_RT_combined_2023_12.parquet`

**Size**: 195.4 KB

**Rows**: 744

**Last Modified**: 2025-08-18T16:05:23.482079


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DA_DC_E | None | float64 | Yes |
| DA_DC_L | None | float64 | Yes |
| DA_DC_N | None | float64 | Yes |
| DA_DC_R | None | float64 | Yes |
| DA_DC_S | None | float64 | Yes |
| DA_HB_BUSAVG | None | float64 | Yes |
| DA_HB_HOUSTON | None | float64 | Yes |
| DA_HB_HUBAVG | None | float64 | Yes |
| DA_HB_NORTH | None | float64 | Yes |
| DA_HB_PAN | None | float64 | Yes |
| DA_HB_SOUTH | None | float64 | Yes |
| DA_HB_WEST | None | float64 | Yes |
| DA_LZ_AEN | None | float64 | Yes |
| DA_LZ_CPS | None | float64 | Yes |
| DA_LZ_HOUSTON | None | float64 | Yes |
| DA_LZ_LCRA | None | float64 | Yes |
| DA_LZ_NORTH | None | float64 | Yes |
| DA_LZ_RAYBN | None | float64 | Yes |
| DA_LZ_SOUTH | None | float64 | Yes |
| DA_LZ_WEST | None | float64 | Yes |
| AS_ECRS | None | float64 | Yes |
| AS_NSPIN | None | float64 | Yes |
| AS_REGDN | None | float64 | Yes |
| AS_REGUP | None | float64 | Yes |
| AS_RRS | None | float64 | Yes |
| RT_DC_E | None | float64 | Yes |
| RT_DC_L | None | float64 | Yes |
| RT_DC_N | None | float64 | Yes |
| RT_DC_R | None | float64 | Yes |
| RT_DC_S | None | float64 | Yes |
| RT_HB_BUSAVG | None | float64 | Yes |
| RT_HB_HOUSTON | None | float64 | Yes |
| RT_HB_HUBAVG | None | float64 | Yes |
| RT_HB_NORTH | None | float64 | Yes |
| RT_HB_PAN | None | float64 | Yes |
| RT_HB_SOUTH | None | float64 | Yes |
| RT_HB_WEST | None | float64 | Yes |
| RT_LZ_AEN | None | float64 | Yes |
| RT_LZ_CPS | None | float64 | Yes |
| RT_LZ_HOUSTON | None | float64 | Yes |
| RT_LZ_LCRA | None | float64 | Yes |
| RT_LZ_NORTH | None | float64 | Yes |
| RT_LZ_RAYBN | None | float64 | Yes |
| RT_LZ_SOUTH | None | float64 | Yes |
| RT_LZ_WEST | None | float64 | Yes |


### DA_AS_RT_combined_2024_01.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_RT_combined/DA_AS_RT_combined_2024_01.parquet`

**Size**: 148.4 KB

**Rows**: 724

**Last Modified**: 2025-08-18T16:05:23.522080


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DA_DC_E | None | float64 | Yes |
| DA_DC_L | None | float64 | Yes |
| DA_DC_N | None | float64 | Yes |
| DA_DC_R | None | float64 | Yes |
| DA_DC_S | None | float64 | Yes |
| DA_HB_BUSAVG | None | float64 | Yes |
| DA_HB_HOUSTON | None | float64 | Yes |
| DA_HB_HUBAVG | None | float64 | Yes |
| DA_HB_NORTH | None | float64 | Yes |
| DA_HB_PAN | None | float64 | Yes |
| DA_HB_SOUTH | None | float64 | Yes |
| DA_HB_WEST | None | float64 | Yes |
| DA_LZ_AEN | None | float64 | Yes |
| DA_LZ_CPS | None | float64 | Yes |
| DA_LZ_HOUSTON | None | float64 | Yes |
| DA_LZ_LCRA | None | float64 | Yes |
| DA_LZ_NORTH | None | float64 | Yes |
| DA_LZ_RAYBN | None | float64 | Yes |
| DA_LZ_SOUTH | None | float64 | Yes |
| DA_LZ_WEST | None | float64 | Yes |
| AS_ECRS | None | float64 | Yes |
| AS_NSPIN | None | float64 | Yes |
| AS_REGDN | None | float64 | Yes |
| AS_REGUP | None | float64 | Yes |
| AS_RRS | None | float64 | Yes |
| RT_DC_E | None | float64 | Yes |
| RT_DC_L | None | float64 | Yes |
| RT_DC_N | None | float64 | Yes |
| RT_DC_R | None | float64 | Yes |
| RT_DC_S | None | float64 | Yes |
| RT_HB_BUSAVG | None | float64 | Yes |
| RT_HB_HOUSTON | None | float64 | Yes |
| RT_HB_HUBAVG | None | float64 | Yes |
| RT_HB_NORTH | None | float64 | Yes |
| RT_HB_PAN | None | float64 | Yes |
| RT_HB_SOUTH | None | float64 | Yes |
| RT_HB_WEST | None | float64 | Yes |
| RT_LZ_AEN | None | float64 | Yes |
| RT_LZ_CPS | None | float64 | Yes |
| RT_LZ_HOUSTON | None | float64 | Yes |
| RT_LZ_LCRA | None | float64 | Yes |
| RT_LZ_NORTH | None | float64 | Yes |
| RT_LZ_RAYBN | None | float64 | Yes |
| RT_LZ_SOUTH | None | float64 | Yes |
| RT_LZ_WEST | None | float64 | Yes |


### DA_AS_RT_combined_2024_02.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_RT_combined/DA_AS_RT_combined_2024_02.parquet`

**Size**: 144.7 KB

**Rows**: 679

**Last Modified**: 2025-08-18T16:05:23.524081


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DA_DC_E | None | float64 | Yes |
| DA_DC_L | None | float64 | Yes |
| DA_DC_N | None | float64 | Yes |
| DA_DC_R | None | float64 | Yes |
| DA_DC_S | None | float64 | Yes |
| DA_HB_BUSAVG | None | float64 | Yes |
| DA_HB_HOUSTON | None | float64 | Yes |
| DA_HB_HUBAVG | None | float64 | Yes |
| DA_HB_NORTH | None | float64 | Yes |
| DA_HB_PAN | None | float64 | Yes |
| DA_HB_SOUTH | None | float64 | Yes |
| DA_HB_WEST | None | float64 | Yes |
| DA_LZ_AEN | None | float64 | Yes |
| DA_LZ_CPS | None | float64 | Yes |
| DA_LZ_HOUSTON | None | float64 | Yes |
| DA_LZ_LCRA | None | float64 | Yes |
| DA_LZ_NORTH | None | float64 | Yes |
| DA_LZ_RAYBN | None | float64 | Yes |
| DA_LZ_SOUTH | None | float64 | Yes |
| DA_LZ_WEST | None | float64 | Yes |
| AS_ECRS | None | float64 | Yes |
| AS_NSPIN | None | float64 | Yes |
| AS_REGDN | None | float64 | Yes |
| AS_REGUP | None | float64 | Yes |
| AS_RRS | None | float64 | Yes |
| RT_DC_E | None | float64 | Yes |
| RT_DC_L | None | float64 | Yes |
| RT_DC_N | None | float64 | Yes |
| RT_DC_R | None | float64 | Yes |
| RT_DC_S | None | float64 | Yes |
| RT_HB_BUSAVG | None | float64 | Yes |
| RT_HB_HOUSTON | None | float64 | Yes |
| RT_HB_HUBAVG | None | float64 | Yes |
| RT_HB_NORTH | None | float64 | Yes |
| RT_HB_PAN | None | float64 | Yes |
| RT_HB_SOUTH | None | float64 | Yes |
| RT_HB_WEST | None | float64 | Yes |
| RT_LZ_AEN | None | float64 | Yes |
| RT_LZ_CPS | None | float64 | Yes |
| RT_LZ_HOUSTON | None | float64 | Yes |
| RT_LZ_LCRA | None | float64 | Yes |
| RT_LZ_NORTH | None | float64 | Yes |
| RT_LZ_RAYBN | None | float64 | Yes |
| RT_LZ_SOUTH | None | float64 | Yes |
| RT_LZ_WEST | None | float64 | Yes |


### DA_AS_RT_combined_2024_03.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_RT_combined/DA_AS_RT_combined_2024_03.parquet`

**Size**: 151.7 KB

**Rows**: 725

**Last Modified**: 2025-08-18T16:05:23.527081


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DA_DC_E | None | float64 | Yes |
| DA_DC_L | None | float64 | Yes |
| DA_DC_N | None | float64 | Yes |
| DA_DC_R | None | float64 | Yes |
| DA_DC_S | None | float64 | Yes |
| DA_HB_BUSAVG | None | float64 | Yes |
| DA_HB_HOUSTON | None | float64 | Yes |
| DA_HB_HUBAVG | None | float64 | Yes |
| DA_HB_NORTH | None | float64 | Yes |
| DA_HB_PAN | None | float64 | Yes |
| DA_HB_SOUTH | None | float64 | Yes |
| DA_HB_WEST | None | float64 | Yes |
| DA_LZ_AEN | None | float64 | Yes |
| DA_LZ_CPS | None | float64 | Yes |
| DA_LZ_HOUSTON | None | float64 | Yes |
| DA_LZ_LCRA | None | float64 | Yes |
| DA_LZ_NORTH | None | float64 | Yes |
| DA_LZ_RAYBN | None | float64 | Yes |
| DA_LZ_SOUTH | None | float64 | Yes |
| DA_LZ_WEST | None | float64 | Yes |
| AS_ECRS | None | float64 | Yes |
| AS_NSPIN | None | float64 | Yes |
| AS_REGDN | None | float64 | Yes |
| AS_REGUP | None | float64 | Yes |
| AS_RRS | None | float64 | Yes |
| RT_DC_E | None | float64 | Yes |
| RT_DC_L | None | float64 | Yes |
| RT_DC_N | None | float64 | Yes |
| RT_DC_R | None | float64 | Yes |
| RT_DC_S | None | float64 | Yes |
| RT_HB_BUSAVG | None | float64 | Yes |
| RT_HB_HOUSTON | None | float64 | Yes |
| RT_HB_HUBAVG | None | float64 | Yes |
| RT_HB_NORTH | None | float64 | Yes |
| RT_HB_PAN | None | float64 | Yes |
| RT_HB_SOUTH | None | float64 | Yes |
| RT_HB_WEST | None | float64 | Yes |
| RT_LZ_AEN | None | float64 | Yes |
| RT_LZ_CPS | None | float64 | Yes |
| RT_LZ_HOUSTON | None | float64 | Yes |
| RT_LZ_LCRA | None | float64 | Yes |
| RT_LZ_NORTH | None | float64 | Yes |
| RT_LZ_RAYBN | None | float64 | Yes |
| RT_LZ_SOUTH | None | float64 | Yes |
| RT_LZ_WEST | None | float64 | Yes |


### DA_AS_RT_combined_2024_04.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_RT_combined/DA_AS_RT_combined_2024_04.parquet`

**Size**: 148.6 KB

**Rows**: 702

**Last Modified**: 2025-08-18T16:05:23.530081


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DA_DC_E | None | float64 | Yes |
| DA_DC_L | None | float64 | Yes |
| DA_DC_N | None | float64 | Yes |
| DA_DC_R | None | float64 | Yes |
| DA_DC_S | None | float64 | Yes |
| DA_HB_BUSAVG | None | float64 | Yes |
| DA_HB_HOUSTON | None | float64 | Yes |
| DA_HB_HUBAVG | None | float64 | Yes |
| DA_HB_NORTH | None | float64 | Yes |
| DA_HB_PAN | None | float64 | Yes |
| DA_HB_SOUTH | None | float64 | Yes |
| DA_HB_WEST | None | float64 | Yes |
| DA_LZ_AEN | None | float64 | Yes |
| DA_LZ_CPS | None | float64 | Yes |
| DA_LZ_HOUSTON | None | float64 | Yes |
| DA_LZ_LCRA | None | float64 | Yes |
| DA_LZ_NORTH | None | float64 | Yes |
| DA_LZ_RAYBN | None | float64 | Yes |
| DA_LZ_SOUTH | None | float64 | Yes |
| DA_LZ_WEST | None | float64 | Yes |
| AS_ECRS | None | float64 | Yes |
| AS_NSPIN | None | float64 | Yes |
| AS_REGDN | None | float64 | Yes |
| AS_REGUP | None | float64 | Yes |
| AS_RRS | None | float64 | Yes |
| RT_DC_E | None | float64 | Yes |
| RT_DC_L | None | float64 | Yes |
| RT_DC_N | None | float64 | Yes |
| RT_DC_R | None | float64 | Yes |
| RT_DC_S | None | float64 | Yes |
| RT_HB_BUSAVG | None | float64 | Yes |
| RT_HB_HOUSTON | None | float64 | Yes |
| RT_HB_HUBAVG | None | float64 | Yes |
| RT_HB_NORTH | None | float64 | Yes |
| RT_HB_PAN | None | float64 | Yes |
| RT_HB_SOUTH | None | float64 | Yes |
| RT_HB_WEST | None | float64 | Yes |
| RT_LZ_AEN | None | float64 | Yes |
| RT_LZ_CPS | None | float64 | Yes |
| RT_LZ_HOUSTON | None | float64 | Yes |
| RT_LZ_LCRA | None | float64 | Yes |
| RT_LZ_NORTH | None | float64 | Yes |
| RT_LZ_RAYBN | None | float64 | Yes |
| RT_LZ_SOUTH | None | float64 | Yes |
| RT_LZ_WEST | None | float64 | Yes |


### DA_AS_RT_combined_2024_05.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_RT_combined/DA_AS_RT_combined_2024_05.parquet`

**Size**: 151.3 KB

**Rows**: 725

**Last Modified**: 2025-08-18T16:05:23.533081


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DA_DC_E | None | float64 | Yes |
| DA_DC_L | None | float64 | Yes |
| DA_DC_N | None | float64 | Yes |
| DA_DC_R | None | float64 | Yes |
| DA_DC_S | None | float64 | Yes |
| DA_HB_BUSAVG | None | float64 | Yes |
| DA_HB_HOUSTON | None | float64 | Yes |
| DA_HB_HUBAVG | None | float64 | Yes |
| DA_HB_NORTH | None | float64 | Yes |
| DA_HB_PAN | None | float64 | Yes |
| DA_HB_SOUTH | None | float64 | Yes |
| DA_HB_WEST | None | float64 | Yes |
| DA_LZ_AEN | None | float64 | Yes |
| DA_LZ_CPS | None | float64 | Yes |
| DA_LZ_HOUSTON | None | float64 | Yes |
| DA_LZ_LCRA | None | float64 | Yes |
| DA_LZ_NORTH | None | float64 | Yes |
| DA_LZ_RAYBN | None | float64 | Yes |
| DA_LZ_SOUTH | None | float64 | Yes |
| DA_LZ_WEST | None | float64 | Yes |
| AS_ECRS | None | float64 | Yes |
| AS_NSPIN | None | float64 | Yes |
| AS_REGDN | None | float64 | Yes |
| AS_REGUP | None | float64 | Yes |
| AS_RRS | None | float64 | Yes |
| RT_DC_E | None | float64 | Yes |
| RT_DC_L | None | float64 | Yes |
| RT_DC_N | None | float64 | Yes |
| RT_DC_R | None | float64 | Yes |
| RT_DC_S | None | float64 | Yes |
| RT_HB_BUSAVG | None | float64 | Yes |
| RT_HB_HOUSTON | None | float64 | Yes |
| RT_HB_HUBAVG | None | float64 | Yes |
| RT_HB_NORTH | None | float64 | Yes |
| RT_HB_PAN | None | float64 | Yes |
| RT_HB_SOUTH | None | float64 | Yes |
| RT_HB_WEST | None | float64 | Yes |
| RT_LZ_AEN | None | float64 | Yes |
| RT_LZ_CPS | None | float64 | Yes |
| RT_LZ_HOUSTON | None | float64 | Yes |
| RT_LZ_LCRA | None | float64 | Yes |
| RT_LZ_NORTH | None | float64 | Yes |
| RT_LZ_RAYBN | None | float64 | Yes |
| RT_LZ_SOUTH | None | float64 | Yes |
| RT_LZ_WEST | None | float64 | Yes |


### DA_AS_RT_combined_2024_06.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_RT_combined/DA_AS_RT_combined_2024_06.parquet`

**Size**: 148.4 KB

**Rows**: 702

**Last Modified**: 2025-08-18T16:05:23.535081


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DA_DC_E | None | float64 | Yes |
| DA_DC_L | None | float64 | Yes |
| DA_DC_N | None | float64 | Yes |
| DA_DC_R | None | float64 | Yes |
| DA_DC_S | None | float64 | Yes |
| DA_HB_BUSAVG | None | float64 | Yes |
| DA_HB_HOUSTON | None | float64 | Yes |
| DA_HB_HUBAVG | None | float64 | Yes |
| DA_HB_NORTH | None | float64 | Yes |
| DA_HB_PAN | None | float64 | Yes |
| DA_HB_SOUTH | None | float64 | Yes |
| DA_HB_WEST | None | float64 | Yes |
| DA_LZ_AEN | None | float64 | Yes |
| DA_LZ_CPS | None | float64 | Yes |
| DA_LZ_HOUSTON | None | float64 | Yes |
| DA_LZ_LCRA | None | float64 | Yes |
| DA_LZ_NORTH | None | float64 | Yes |
| DA_LZ_RAYBN | None | float64 | Yes |
| DA_LZ_SOUTH | None | float64 | Yes |
| DA_LZ_WEST | None | float64 | Yes |
| AS_ECRS | None | float64 | Yes |
| AS_NSPIN | None | float64 | Yes |
| AS_REGDN | None | float64 | Yes |
| AS_REGUP | None | float64 | Yes |
| AS_RRS | None | float64 | Yes |
| RT_DC_E | None | float64 | Yes |
| RT_DC_L | None | float64 | Yes |
| RT_DC_N | None | float64 | Yes |
| RT_DC_R | None | float64 | Yes |
| RT_DC_S | None | float64 | Yes |
| RT_HB_BUSAVG | None | float64 | Yes |
| RT_HB_HOUSTON | None | float64 | Yes |
| RT_HB_HUBAVG | None | float64 | Yes |
| RT_HB_NORTH | None | float64 | Yes |
| RT_HB_PAN | None | float64 | Yes |
| RT_HB_SOUTH | None | float64 | Yes |
| RT_HB_WEST | None | float64 | Yes |
| RT_LZ_AEN | None | float64 | Yes |
| RT_LZ_CPS | None | float64 | Yes |
| RT_LZ_HOUSTON | None | float64 | Yes |
| RT_LZ_LCRA | None | float64 | Yes |
| RT_LZ_NORTH | None | float64 | Yes |
| RT_LZ_RAYBN | None | float64 | Yes |
| RT_LZ_SOUTH | None | float64 | Yes |
| RT_LZ_WEST | None | float64 | Yes |


### DA_AS_RT_combined_2024_07.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_RT_combined/DA_AS_RT_combined_2024_07.parquet`

**Size**: 149.0 KB

**Rows**: 725

**Last Modified**: 2025-08-18T16:05:23.538081


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DA_DC_E | None | float64 | Yes |
| DA_DC_L | None | float64 | Yes |
| DA_DC_N | None | float64 | Yes |
| DA_DC_R | None | float64 | Yes |
| DA_DC_S | None | float64 | Yes |
| DA_HB_BUSAVG | None | float64 | Yes |
| DA_HB_HOUSTON | None | float64 | Yes |
| DA_HB_HUBAVG | None | float64 | Yes |
| DA_HB_NORTH | None | float64 | Yes |
| DA_HB_PAN | None | float64 | Yes |
| DA_HB_SOUTH | None | float64 | Yes |
| DA_HB_WEST | None | float64 | Yes |
| DA_LZ_AEN | None | float64 | Yes |
| DA_LZ_CPS | None | float64 | Yes |
| DA_LZ_HOUSTON | None | float64 | Yes |
| DA_LZ_LCRA | None | float64 | Yes |
| DA_LZ_NORTH | None | float64 | Yes |
| DA_LZ_RAYBN | None | float64 | Yes |
| DA_LZ_SOUTH | None | float64 | Yes |
| DA_LZ_WEST | None | float64 | Yes |
| AS_ECRS | None | float64 | Yes |
| AS_NSPIN | None | float64 | Yes |
| AS_REGDN | None | float64 | Yes |
| AS_REGUP | None | float64 | Yes |
| AS_RRS | None | float64 | Yes |
| RT_DC_E | None | float64 | Yes |
| RT_DC_L | None | float64 | Yes |
| RT_DC_N | None | float64 | Yes |
| RT_DC_R | None | float64 | Yes |
| RT_DC_S | None | float64 | Yes |
| RT_HB_BUSAVG | None | float64 | Yes |
| RT_HB_HOUSTON | None | float64 | Yes |
| RT_HB_HUBAVG | None | float64 | Yes |
| RT_HB_NORTH | None | float64 | Yes |
| RT_HB_PAN | None | float64 | Yes |
| RT_HB_SOUTH | None | float64 | Yes |
| RT_HB_WEST | None | float64 | Yes |
| RT_LZ_AEN | None | float64 | Yes |
| RT_LZ_CPS | None | float64 | Yes |
| RT_LZ_HOUSTON | None | float64 | Yes |
| RT_LZ_LCRA | None | float64 | Yes |
| RT_LZ_NORTH | None | float64 | Yes |
| RT_LZ_RAYBN | None | float64 | Yes |
| RT_LZ_SOUTH | None | float64 | Yes |
| RT_LZ_WEST | None | float64 | Yes |


### DA_AS_RT_combined_2024_08.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_RT_combined/DA_AS_RT_combined_2024_08.parquet`

**Size**: 149.4 KB

**Rows**: 725

**Last Modified**: 2025-08-18T16:05:23.541081


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DA_DC_E | None | float64 | Yes |
| DA_DC_L | None | float64 | Yes |
| DA_DC_N | None | float64 | Yes |
| DA_DC_R | None | float64 | Yes |
| DA_DC_S | None | float64 | Yes |
| DA_HB_BUSAVG | None | float64 | Yes |
| DA_HB_HOUSTON | None | float64 | Yes |
| DA_HB_HUBAVG | None | float64 | Yes |
| DA_HB_NORTH | None | float64 | Yes |
| DA_HB_PAN | None | float64 | Yes |
| DA_HB_SOUTH | None | float64 | Yes |
| DA_HB_WEST | None | float64 | Yes |
| DA_LZ_AEN | None | float64 | Yes |
| DA_LZ_CPS | None | float64 | Yes |
| DA_LZ_HOUSTON | None | float64 | Yes |
| DA_LZ_LCRA | None | float64 | Yes |
| DA_LZ_NORTH | None | float64 | Yes |
| DA_LZ_RAYBN | None | float64 | Yes |
| DA_LZ_SOUTH | None | float64 | Yes |
| DA_LZ_WEST | None | float64 | Yes |
| AS_ECRS | None | float64 | Yes |
| AS_NSPIN | None | float64 | Yes |
| AS_REGDN | None | float64 | Yes |
| AS_REGUP | None | float64 | Yes |
| AS_RRS | None | float64 | Yes |
| RT_DC_E | None | float64 | Yes |
| RT_DC_L | None | float64 | Yes |
| RT_DC_N | None | float64 | Yes |
| RT_DC_R | None | float64 | Yes |
| RT_DC_S | None | float64 | Yes |
| RT_HB_BUSAVG | None | float64 | Yes |
| RT_HB_HOUSTON | None | float64 | Yes |
| RT_HB_HUBAVG | None | float64 | Yes |
| RT_HB_NORTH | None | float64 | Yes |
| RT_HB_PAN | None | float64 | Yes |
| RT_HB_SOUTH | None | float64 | Yes |
| RT_HB_WEST | None | float64 | Yes |
| RT_LZ_AEN | None | float64 | Yes |
| RT_LZ_CPS | None | float64 | Yes |
| RT_LZ_HOUSTON | None | float64 | Yes |
| RT_LZ_LCRA | None | float64 | Yes |
| RT_LZ_NORTH | None | float64 | Yes |
| RT_LZ_RAYBN | None | float64 | Yes |
| RT_LZ_SOUTH | None | float64 | Yes |
| RT_LZ_WEST | None | float64 | Yes |


### DA_AS_RT_combined_2024_09.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_RT_combined/DA_AS_RT_combined_2024_09.parquet`

**Size**: 146.1 KB

**Rows**: 702

**Last Modified**: 2025-08-18T16:05:23.544081


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DA_DC_E | None | float64 | Yes |
| DA_DC_L | None | float64 | Yes |
| DA_DC_N | None | float64 | Yes |
| DA_DC_R | None | float64 | Yes |
| DA_DC_S | None | float64 | Yes |
| DA_HB_BUSAVG | None | float64 | Yes |
| DA_HB_HOUSTON | None | float64 | Yes |
| DA_HB_HUBAVG | None | float64 | Yes |
| DA_HB_NORTH | None | float64 | Yes |
| DA_HB_PAN | None | float64 | Yes |
| DA_HB_SOUTH | None | float64 | Yes |
| DA_HB_WEST | None | float64 | Yes |
| DA_LZ_AEN | None | float64 | Yes |
| DA_LZ_CPS | None | float64 | Yes |
| DA_LZ_HOUSTON | None | float64 | Yes |
| DA_LZ_LCRA | None | float64 | Yes |
| DA_LZ_NORTH | None | float64 | Yes |
| DA_LZ_RAYBN | None | float64 | Yes |
| DA_LZ_SOUTH | None | float64 | Yes |
| DA_LZ_WEST | None | float64 | Yes |
| AS_ECRS | None | float64 | Yes |
| AS_NSPIN | None | float64 | Yes |
| AS_REGDN | None | float64 | Yes |
| AS_REGUP | None | float64 | Yes |
| AS_RRS | None | float64 | Yes |
| RT_DC_E | None | float64 | Yes |
| RT_DC_L | None | float64 | Yes |
| RT_DC_N | None | float64 | Yes |
| RT_DC_R | None | float64 | Yes |
| RT_DC_S | None | float64 | Yes |
| RT_HB_BUSAVG | None | float64 | Yes |
| RT_HB_HOUSTON | None | float64 | Yes |
| RT_HB_HUBAVG | None | float64 | Yes |
| RT_HB_NORTH | None | float64 | Yes |
| RT_HB_PAN | None | float64 | Yes |
| RT_HB_SOUTH | None | float64 | Yes |
| RT_HB_WEST | None | float64 | Yes |
| RT_LZ_AEN | None | float64 | Yes |
| RT_LZ_CPS | None | float64 | Yes |
| RT_LZ_HOUSTON | None | float64 | Yes |
| RT_LZ_LCRA | None | float64 | Yes |
| RT_LZ_NORTH | None | float64 | Yes |
| RT_LZ_RAYBN | None | float64 | Yes |
| RT_LZ_SOUTH | None | float64 | Yes |
| RT_LZ_WEST | None | float64 | Yes |


### DA_AS_RT_combined_2024_10.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_RT_combined/DA_AS_RT_combined_2024_10.parquet`

**Size**: 150.8 KB

**Rows**: 725

**Last Modified**: 2025-08-18T16:05:23.547081


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DA_DC_E | None | float64 | Yes |
| DA_DC_L | None | float64 | Yes |
| DA_DC_N | None | float64 | Yes |
| DA_DC_R | None | float64 | Yes |
| DA_DC_S | None | float64 | Yes |
| DA_HB_BUSAVG | None | float64 | Yes |
| DA_HB_HOUSTON | None | float64 | Yes |
| DA_HB_HUBAVG | None | float64 | Yes |
| DA_HB_NORTH | None | float64 | Yes |
| DA_HB_PAN | None | float64 | Yes |
| DA_HB_SOUTH | None | float64 | Yes |
| DA_HB_WEST | None | float64 | Yes |
| DA_LZ_AEN | None | float64 | Yes |
| DA_LZ_CPS | None | float64 | Yes |
| DA_LZ_HOUSTON | None | float64 | Yes |
| DA_LZ_LCRA | None | float64 | Yes |
| DA_LZ_NORTH | None | float64 | Yes |
| DA_LZ_RAYBN | None | float64 | Yes |
| DA_LZ_SOUTH | None | float64 | Yes |
| DA_LZ_WEST | None | float64 | Yes |
| AS_ECRS | None | float64 | Yes |
| AS_NSPIN | None | float64 | Yes |
| AS_REGDN | None | float64 | Yes |
| AS_REGUP | None | float64 | Yes |
| AS_RRS | None | float64 | Yes |
| RT_DC_E | None | float64 | Yes |
| RT_DC_L | None | float64 | Yes |
| RT_DC_N | None | float64 | Yes |
| RT_DC_R | None | float64 | Yes |
| RT_DC_S | None | float64 | Yes |
| RT_HB_BUSAVG | None | float64 | Yes |
| RT_HB_HOUSTON | None | float64 | Yes |
| RT_HB_HUBAVG | None | float64 | Yes |
| RT_HB_NORTH | None | float64 | Yes |
| RT_HB_PAN | None | float64 | Yes |
| RT_HB_SOUTH | None | float64 | Yes |
| RT_HB_WEST | None | float64 | Yes |
| RT_LZ_AEN | None | float64 | Yes |
| RT_LZ_CPS | None | float64 | Yes |
| RT_LZ_HOUSTON | None | float64 | Yes |
| RT_LZ_LCRA | None | float64 | Yes |
| RT_LZ_NORTH | None | float64 | Yes |
| RT_LZ_RAYBN | None | float64 | Yes |
| RT_LZ_SOUTH | None | float64 | Yes |
| RT_LZ_WEST | None | float64 | Yes |


### DA_AS_RT_combined_2024_11.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_RT_combined/DA_AS_RT_combined_2024_11.parquet`

**Size**: 148.2 KB

**Rows**: 702

**Last Modified**: 2025-08-18T16:05:23.550082


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DA_DC_E | None | float64 | Yes |
| DA_DC_L | None | float64 | Yes |
| DA_DC_N | None | float64 | Yes |
| DA_DC_R | None | float64 | Yes |
| DA_DC_S | None | float64 | Yes |
| DA_HB_BUSAVG | None | float64 | Yes |
| DA_HB_HOUSTON | None | float64 | Yes |
| DA_HB_HUBAVG | None | float64 | Yes |
| DA_HB_NORTH | None | float64 | Yes |
| DA_HB_PAN | None | float64 | Yes |
| DA_HB_SOUTH | None | float64 | Yes |
| DA_HB_WEST | None | float64 | Yes |
| DA_LZ_AEN | None | float64 | Yes |
| DA_LZ_CPS | None | float64 | Yes |
| DA_LZ_HOUSTON | None | float64 | Yes |
| DA_LZ_LCRA | None | float64 | Yes |
| DA_LZ_NORTH | None | float64 | Yes |
| DA_LZ_RAYBN | None | float64 | Yes |
| DA_LZ_SOUTH | None | float64 | Yes |
| DA_LZ_WEST | None | float64 | Yes |
| AS_ECRS | None | float64 | Yes |
| AS_NSPIN | None | float64 | Yes |
| AS_REGDN | None | float64 | Yes |
| AS_REGUP | None | float64 | Yes |
| AS_RRS | None | float64 | Yes |
| RT_DC_E | None | float64 | Yes |
| RT_DC_L | None | float64 | Yes |
| RT_DC_N | None | float64 | Yes |
| RT_DC_R | None | float64 | Yes |
| RT_DC_S | None | float64 | Yes |
| RT_HB_BUSAVG | None | float64 | Yes |
| RT_HB_HOUSTON | None | float64 | Yes |
| RT_HB_HUBAVG | None | float64 | Yes |
| RT_HB_NORTH | None | float64 | Yes |
| RT_HB_PAN | None | float64 | Yes |
| RT_HB_SOUTH | None | float64 | Yes |
| RT_HB_WEST | None | float64 | Yes |
| RT_LZ_AEN | None | float64 | Yes |
| RT_LZ_CPS | None | float64 | Yes |
| RT_LZ_HOUSTON | None | float64 | Yes |
| RT_LZ_LCRA | None | float64 | Yes |
| RT_LZ_NORTH | None | float64 | Yes |
| RT_LZ_RAYBN | None | float64 | Yes |
| RT_LZ_SOUTH | None | float64 | Yes |
| RT_LZ_WEST | None | float64 | Yes |


### DA_AS_RT_combined_2024_12.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_RT_combined/DA_AS_RT_combined_2024_12.parquet`

**Size**: 150.5 KB

**Rows**: 725

**Last Modified**: 2025-08-18T16:05:23.552082


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DA_DC_E | None | float64 | Yes |
| DA_DC_L | None | float64 | Yes |
| DA_DC_N | None | float64 | Yes |
| DA_DC_R | None | float64 | Yes |
| DA_DC_S | None | float64 | Yes |
| DA_HB_BUSAVG | None | float64 | Yes |
| DA_HB_HOUSTON | None | float64 | Yes |
| DA_HB_HUBAVG | None | float64 | Yes |
| DA_HB_NORTH | None | float64 | Yes |
| DA_HB_PAN | None | float64 | Yes |
| DA_HB_SOUTH | None | float64 | Yes |
| DA_HB_WEST | None | float64 | Yes |
| DA_LZ_AEN | None | float64 | Yes |
| DA_LZ_CPS | None | float64 | Yes |
| DA_LZ_HOUSTON | None | float64 | Yes |
| DA_LZ_LCRA | None | float64 | Yes |
| DA_LZ_NORTH | None | float64 | Yes |
| DA_LZ_RAYBN | None | float64 | Yes |
| DA_LZ_SOUTH | None | float64 | Yes |
| DA_LZ_WEST | None | float64 | Yes |
| AS_ECRS | None | float64 | Yes |
| AS_NSPIN | None | float64 | Yes |
| AS_REGDN | None | float64 | Yes |
| AS_REGUP | None | float64 | Yes |
| AS_RRS | None | float64 | Yes |
| RT_DC_E | None | float64 | Yes |
| RT_DC_L | None | float64 | Yes |
| RT_DC_N | None | float64 | Yes |
| RT_DC_R | None | float64 | Yes |
| RT_DC_S | None | float64 | Yes |
| RT_HB_BUSAVG | None | float64 | Yes |
| RT_HB_HOUSTON | None | float64 | Yes |
| RT_HB_HUBAVG | None | float64 | Yes |
| RT_HB_NORTH | None | float64 | Yes |
| RT_HB_PAN | None | float64 | Yes |
| RT_HB_SOUTH | None | float64 | Yes |
| RT_HB_WEST | None | float64 | Yes |
| RT_LZ_AEN | None | float64 | Yes |
| RT_LZ_CPS | None | float64 | Yes |
| RT_LZ_HOUSTON | None | float64 | Yes |
| RT_LZ_LCRA | None | float64 | Yes |
| RT_LZ_NORTH | None | float64 | Yes |
| RT_LZ_RAYBN | None | float64 | Yes |
| RT_LZ_SOUTH | None | float64 | Yes |
| RT_LZ_WEST | None | float64 | Yes |


### DA_AS_combined_2023_01.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_combined/DA_AS_combined_2023_01.parquet`

**Size**: 105.3 KB

**Rows**: 720

**Last Modified**: 2025-08-18T16:05:22.244020


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DC_E | None | float64 | Yes |
| DC_L | None | float64 | Yes |
| DC_N | None | float64 | Yes |
| DC_R | None | float64 | Yes |
| DC_S | None | float64 | Yes |
| HB_BUSAVG | None | float64 | Yes |
| HB_HOUSTON | None | float64 | Yes |
| HB_HUBAVG | None | float64 | Yes |
| HB_NORTH | None | float64 | Yes |
| HB_PAN | None | float64 | Yes |
| HB_SOUTH | None | float64 | Yes |
| HB_WEST | None | float64 | Yes |
| LZ_AEN | None | float64 | Yes |
| LZ_CPS | None | float64 | Yes |
| LZ_HOUSTON | None | float64 | Yes |
| LZ_LCRA | None | float64 | Yes |
| LZ_NORTH | None | float64 | Yes |
| LZ_RAYBN | None | float64 | Yes |
| LZ_SOUTH | None | float64 | Yes |
| LZ_WEST | None | float64 | Yes |
| ECRS | None | float64 | Yes |
| NSPIN | None | float64 | Yes |
| REGDN | None | float64 | Yes |
| REGUP | None | float64 | Yes |
| RRS | None | float64 | Yes |


### DA_AS_combined_2023_02.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_combined/DA_AS_combined_2023_02.parquet`

**Size**: 99.7 KB

**Rows**: 672

**Last Modified**: 2025-08-18T16:05:22.246020


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DC_E | None | float64 | Yes |
| DC_L | None | float64 | Yes |
| DC_N | None | float64 | Yes |
| DC_R | None | float64 | Yes |
| DC_S | None | float64 | Yes |
| HB_BUSAVG | None | float64 | Yes |
| HB_HOUSTON | None | float64 | Yes |
| HB_HUBAVG | None | float64 | Yes |
| HB_NORTH | None | float64 | Yes |
| HB_PAN | None | float64 | Yes |
| HB_SOUTH | None | float64 | Yes |
| HB_WEST | None | float64 | Yes |
| LZ_AEN | None | float64 | Yes |
| LZ_CPS | None | float64 | Yes |
| LZ_HOUSTON | None | float64 | Yes |
| LZ_LCRA | None | float64 | Yes |
| LZ_NORTH | None | float64 | Yes |
| LZ_RAYBN | None | float64 | Yes |
| LZ_SOUTH | None | float64 | Yes |
| LZ_WEST | None | float64 | Yes |
| ECRS | None | float64 | Yes |
| NSPIN | None | float64 | Yes |
| REGDN | None | float64 | Yes |
| REGUP | None | float64 | Yes |
| RRS | None | float64 | Yes |


### DA_AS_combined_2023_03.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_combined/DA_AS_combined_2023_03.parquet`

**Size**: 110.4 KB

**Rows**: 743

**Last Modified**: 2025-08-18T16:05:22.248020


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DC_E | None | float64 | Yes |
| DC_L | None | float64 | Yes |
| DC_N | None | float64 | Yes |
| DC_R | None | float64 | Yes |
| DC_S | None | float64 | Yes |
| HB_BUSAVG | None | float64 | Yes |
| HB_HOUSTON | None | float64 | Yes |
| HB_HUBAVG | None | float64 | Yes |
| HB_NORTH | None | float64 | Yes |
| HB_PAN | None | float64 | Yes |
| HB_SOUTH | None | float64 | Yes |
| HB_WEST | None | float64 | Yes |
| LZ_AEN | None | float64 | Yes |
| LZ_CPS | None | float64 | Yes |
| LZ_HOUSTON | None | float64 | Yes |
| LZ_LCRA | None | float64 | Yes |
| LZ_NORTH | None | float64 | Yes |
| LZ_RAYBN | None | float64 | Yes |
| LZ_SOUTH | None | float64 | Yes |
| LZ_WEST | None | float64 | Yes |
| ECRS | None | float64 | Yes |
| NSPIN | None | float64 | Yes |
| REGDN | None | float64 | Yes |
| REGUP | None | float64 | Yes |
| RRS | None | float64 | Yes |


### DA_AS_combined_2023_04.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_combined/DA_AS_combined_2023_04.parquet`

**Size**: 107.1 KB

**Rows**: 720

**Last Modified**: 2025-08-18T16:05:22.250021


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DC_E | None | float64 | Yes |
| DC_L | None | float64 | Yes |
| DC_N | None | float64 | Yes |
| DC_R | None | float64 | Yes |
| DC_S | None | float64 | Yes |
| HB_BUSAVG | None | float64 | Yes |
| HB_HOUSTON | None | float64 | Yes |
| HB_HUBAVG | None | float64 | Yes |
| HB_NORTH | None | float64 | Yes |
| HB_PAN | None | float64 | Yes |
| HB_SOUTH | None | float64 | Yes |
| HB_WEST | None | float64 | Yes |
| LZ_AEN | None | float64 | Yes |
| LZ_CPS | None | float64 | Yes |
| LZ_HOUSTON | None | float64 | Yes |
| LZ_LCRA | None | float64 | Yes |
| LZ_NORTH | None | float64 | Yes |
| LZ_RAYBN | None | float64 | Yes |
| LZ_SOUTH | None | float64 | Yes |
| LZ_WEST | None | float64 | Yes |
| ECRS | None | float64 | Yes |
| NSPIN | None | float64 | Yes |
| REGDN | None | float64 | Yes |
| REGUP | None | float64 | Yes |
| RRS | None | float64 | Yes |


### DA_AS_combined_2023_05.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_combined/DA_AS_combined_2023_05.parquet`

**Size**: 108.8 KB

**Rows**: 744

**Last Modified**: 2025-08-18T16:05:22.252021


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DC_E | None | float64 | Yes |
| DC_L | None | float64 | Yes |
| DC_N | None | float64 | Yes |
| DC_R | None | float64 | Yes |
| DC_S | None | float64 | Yes |
| HB_BUSAVG | None | float64 | Yes |
| HB_HOUSTON | None | float64 | Yes |
| HB_HUBAVG | None | float64 | Yes |
| HB_NORTH | None | float64 | Yes |
| HB_PAN | None | float64 | Yes |
| HB_SOUTH | None | float64 | Yes |
| HB_WEST | None | float64 | Yes |
| LZ_AEN | None | float64 | Yes |
| LZ_CPS | None | float64 | Yes |
| LZ_HOUSTON | None | float64 | Yes |
| LZ_LCRA | None | float64 | Yes |
| LZ_NORTH | None | float64 | Yes |
| LZ_RAYBN | None | float64 | Yes |
| LZ_SOUTH | None | float64 | Yes |
| LZ_WEST | None | float64 | Yes |
| ECRS | None | float64 | Yes |
| NSPIN | None | float64 | Yes |
| REGDN | None | float64 | Yes |
| REGUP | None | float64 | Yes |
| RRS | None | float64 | Yes |


### DA_AS_combined_2023_06.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_combined/DA_AS_combined_2023_06.parquet`

**Size**: 111.5 KB

**Rows**: 720

**Last Modified**: 2025-08-18T16:05:22.254021


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DC_E | None | float64 | Yes |
| DC_L | None | float64 | Yes |
| DC_N | None | float64 | Yes |
| DC_R | None | float64 | Yes |
| DC_S | None | float64 | Yes |
| HB_BUSAVG | None | float64 | Yes |
| HB_HOUSTON | None | float64 | Yes |
| HB_HUBAVG | None | float64 | Yes |
| HB_NORTH | None | float64 | Yes |
| HB_PAN | None | float64 | Yes |
| HB_SOUTH | None | float64 | Yes |
| HB_WEST | None | float64 | Yes |
| LZ_AEN | None | float64 | Yes |
| LZ_CPS | None | float64 | Yes |
| LZ_HOUSTON | None | float64 | Yes |
| LZ_LCRA | None | float64 | Yes |
| LZ_NORTH | None | float64 | Yes |
| LZ_RAYBN | None | float64 | Yes |
| LZ_SOUTH | None | float64 | Yes |
| LZ_WEST | None | float64 | Yes |
| ECRS | None | float64 | Yes |
| NSPIN | None | float64 | Yes |
| REGDN | None | float64 | Yes |
| REGUP | None | float64 | Yes |
| RRS | None | float64 | Yes |


### DA_AS_combined_2023_07.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_combined/DA_AS_combined_2023_07.parquet`

**Size**: 111.9 KB

**Rows**: 744

**Last Modified**: 2025-08-18T16:05:22.255021


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DC_E | None | float64 | Yes |
| DC_L | None | float64 | Yes |
| DC_N | None | float64 | Yes |
| DC_R | None | float64 | Yes |
| DC_S | None | float64 | Yes |
| HB_BUSAVG | None | float64 | Yes |
| HB_HOUSTON | None | float64 | Yes |
| HB_HUBAVG | None | float64 | Yes |
| HB_NORTH | None | float64 | Yes |
| HB_PAN | None | float64 | Yes |
| HB_SOUTH | None | float64 | Yes |
| HB_WEST | None | float64 | Yes |
| LZ_AEN | None | float64 | Yes |
| LZ_CPS | None | float64 | Yes |
| LZ_HOUSTON | None | float64 | Yes |
| LZ_LCRA | None | float64 | Yes |
| LZ_NORTH | None | float64 | Yes |
| LZ_RAYBN | None | float64 | Yes |
| LZ_SOUTH | None | float64 | Yes |
| LZ_WEST | None | float64 | Yes |
| ECRS | None | float64 | Yes |
| NSPIN | None | float64 | Yes |
| REGDN | None | float64 | Yes |
| REGUP | None | float64 | Yes |
| RRS | None | float64 | Yes |


### DA_AS_combined_2023_08.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_combined/DA_AS_combined_2023_08.parquet`

**Size**: 115.9 KB

**Rows**: 744

**Last Modified**: 2025-08-18T16:05:22.257021


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DC_E | None | float64 | Yes |
| DC_L | None | float64 | Yes |
| DC_N | None | float64 | Yes |
| DC_R | None | float64 | Yes |
| DC_S | None | float64 | Yes |
| HB_BUSAVG | None | float64 | Yes |
| HB_HOUSTON | None | float64 | Yes |
| HB_HUBAVG | None | float64 | Yes |
| HB_NORTH | None | float64 | Yes |
| HB_PAN | None | float64 | Yes |
| HB_SOUTH | None | float64 | Yes |
| HB_WEST | None | float64 | Yes |
| LZ_AEN | None | float64 | Yes |
| LZ_CPS | None | float64 | Yes |
| LZ_HOUSTON | None | float64 | Yes |
| LZ_LCRA | None | float64 | Yes |
| LZ_NORTH | None | float64 | Yes |
| LZ_RAYBN | None | float64 | Yes |
| LZ_SOUTH | None | float64 | Yes |
| LZ_WEST | None | float64 | Yes |
| ECRS | None | float64 | Yes |
| NSPIN | None | float64 | Yes |
| REGDN | None | float64 | Yes |
| REGUP | None | float64 | Yes |
| RRS | None | float64 | Yes |


### DA_AS_combined_2023_09.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_combined/DA_AS_combined_2023_09.parquet`

**Size**: 108.1 KB

**Rows**: 720

**Last Modified**: 2025-08-18T16:05:22.259021


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DC_E | None | float64 | Yes |
| DC_L | None | float64 | Yes |
| DC_N | None | float64 | Yes |
| DC_R | None | float64 | Yes |
| DC_S | None | float64 | Yes |
| HB_BUSAVG | None | float64 | Yes |
| HB_HOUSTON | None | float64 | Yes |
| HB_HUBAVG | None | float64 | Yes |
| HB_NORTH | None | float64 | Yes |
| HB_PAN | None | float64 | Yes |
| HB_SOUTH | None | float64 | Yes |
| HB_WEST | None | float64 | Yes |
| LZ_AEN | None | float64 | Yes |
| LZ_CPS | None | float64 | Yes |
| LZ_HOUSTON | None | float64 | Yes |
| LZ_LCRA | None | float64 | Yes |
| LZ_NORTH | None | float64 | Yes |
| LZ_RAYBN | None | float64 | Yes |
| LZ_SOUTH | None | float64 | Yes |
| LZ_WEST | None | float64 | Yes |
| ECRS | None | float64 | Yes |
| NSPIN | None | float64 | Yes |
| REGDN | None | float64 | Yes |
| REGUP | None | float64 | Yes |
| RRS | None | float64 | Yes |


### DA_AS_combined_2023_10.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_combined/DA_AS_combined_2023_10.parquet`

**Size**: 108.1 KB

**Rows**: 744

**Last Modified**: 2025-08-18T16:05:22.261021


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DC_E | None | float64 | Yes |
| DC_L | None | float64 | Yes |
| DC_N | None | float64 | Yes |
| DC_R | None | float64 | Yes |
| DC_S | None | float64 | Yes |
| HB_BUSAVG | None | float64 | Yes |
| HB_HOUSTON | None | float64 | Yes |
| HB_HUBAVG | None | float64 | Yes |
| HB_NORTH | None | float64 | Yes |
| HB_PAN | None | float64 | Yes |
| HB_SOUTH | None | float64 | Yes |
| HB_WEST | None | float64 | Yes |
| LZ_AEN | None | float64 | Yes |
| LZ_CPS | None | float64 | Yes |
| LZ_HOUSTON | None | float64 | Yes |
| LZ_LCRA | None | float64 | Yes |
| LZ_NORTH | None | float64 | Yes |
| LZ_RAYBN | None | float64 | Yes |
| LZ_SOUTH | None | float64 | Yes |
| LZ_WEST | None | float64 | Yes |
| ECRS | None | float64 | Yes |
| NSPIN | None | float64 | Yes |
| REGDN | None | float64 | Yes |
| REGUP | None | float64 | Yes |
| RRS | None | float64 | Yes |


### DA_AS_combined_2023_11.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_combined/DA_AS_combined_2023_11.parquet`

**Size**: 105.1 KB

**Rows**: 720

**Last Modified**: 2025-08-18T16:05:22.263021


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DC_E | None | float64 | Yes |
| DC_L | None | float64 | Yes |
| DC_N | None | float64 | Yes |
| DC_R | None | float64 | Yes |
| DC_S | None | float64 | Yes |
| HB_BUSAVG | None | float64 | Yes |
| HB_HOUSTON | None | float64 | Yes |
| HB_HUBAVG | None | float64 | Yes |
| HB_NORTH | None | float64 | Yes |
| HB_PAN | None | float64 | Yes |
| HB_SOUTH | None | float64 | Yes |
| HB_WEST | None | float64 | Yes |
| LZ_AEN | None | float64 | Yes |
| LZ_CPS | None | float64 | Yes |
| LZ_HOUSTON | None | float64 | Yes |
| LZ_LCRA | None | float64 | Yes |
| LZ_NORTH | None | float64 | Yes |
| LZ_RAYBN | None | float64 | Yes |
| LZ_SOUTH | None | float64 | Yes |
| LZ_WEST | None | float64 | Yes |
| ECRS | None | float64 | Yes |
| NSPIN | None | float64 | Yes |
| REGDN | None | float64 | Yes |
| REGUP | None | float64 | Yes |
| RRS | None | float64 | Yes |


### DA_AS_combined_2023_12.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_combined/DA_AS_combined_2023_12.parquet`

**Size**: 102.0 KB

**Rows**: 744

**Last Modified**: 2025-08-18T16:05:22.265021


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DC_E | None | float64 | Yes |
| DC_L | None | float64 | Yes |
| DC_N | None | float64 | Yes |
| DC_R | None | float64 | Yes |
| DC_S | None | float64 | Yes |
| HB_BUSAVG | None | float64 | Yes |
| HB_HOUSTON | None | float64 | Yes |
| HB_HUBAVG | None | float64 | Yes |
| HB_NORTH | None | float64 | Yes |
| HB_PAN | None | float64 | Yes |
| HB_SOUTH | None | float64 | Yes |
| HB_WEST | None | float64 | Yes |
| LZ_AEN | None | float64 | Yes |
| LZ_CPS | None | float64 | Yes |
| LZ_HOUSTON | None | float64 | Yes |
| LZ_LCRA | None | float64 | Yes |
| LZ_NORTH | None | float64 | Yes |
| LZ_RAYBN | None | float64 | Yes |
| LZ_SOUTH | None | float64 | Yes |
| LZ_WEST | None | float64 | Yes |
| ECRS | None | float64 | Yes |
| NSPIN | None | float64 | Yes |
| REGDN | None | float64 | Yes |
| REGUP | None | float64 | Yes |
| RRS | None | float64 | Yes |


### DA_AS_combined_2024_01.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_combined/DA_AS_combined_2024_01.parquet`

**Size**: 49.4 KB

**Rows**: 264

**Last Modified**: 2025-08-18T16:05:22.281022


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DC_E | None | float64 | Yes |
| DC_L | None | float64 | Yes |
| DC_N | None | float64 | Yes |
| DC_R | None | float64 | Yes |
| DC_S | None | float64 | Yes |
| HB_BUSAVG | None | float64 | Yes |
| HB_HOUSTON | None | float64 | Yes |
| HB_HUBAVG | None | float64 | Yes |
| HB_NORTH | None | float64 | Yes |
| HB_PAN | None | float64 | Yes |
| HB_SOUTH | None | float64 | Yes |
| HB_WEST | None | float64 | Yes |
| LZ_AEN | None | float64 | Yes |
| LZ_CPS | None | float64 | Yes |
| LZ_HOUSTON | None | float64 | Yes |
| LZ_LCRA | None | float64 | Yes |
| LZ_NORTH | None | float64 | Yes |
| LZ_RAYBN | None | float64 | Yes |
| LZ_SOUTH | None | float64 | Yes |
| LZ_WEST | None | float64 | Yes |
| ECRS | None | float64 | Yes |
| NSPIN | None | float64 | Yes |
| REGDN | None | float64 | Yes |
| REGUP | None | float64 | Yes |
| RRS | None | float64 | Yes |


### DA_AS_combined_2024_02.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_combined/DA_AS_combined_2024_02.parquet`

**Size**: 53.0 KB

**Rows**: 288

**Last Modified**: 2025-08-18T16:05:22.282022


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DC_E | None | float64 | Yes |
| DC_L | None | float64 | Yes |
| DC_N | None | float64 | Yes |
| DC_R | None | float64 | Yes |
| DC_S | None | float64 | Yes |
| HB_BUSAVG | None | float64 | Yes |
| HB_HOUSTON | None | float64 | Yes |
| HB_HUBAVG | None | float64 | Yes |
| HB_NORTH | None | float64 | Yes |
| HB_PAN | None | float64 | Yes |
| HB_SOUTH | None | float64 | Yes |
| HB_WEST | None | float64 | Yes |
| LZ_AEN | None | float64 | Yes |
| LZ_CPS | None | float64 | Yes |
| LZ_HOUSTON | None | float64 | Yes |
| LZ_LCRA | None | float64 | Yes |
| LZ_NORTH | None | float64 | Yes |
| LZ_RAYBN | None | float64 | Yes |
| LZ_SOUTH | None | float64 | Yes |
| LZ_WEST | None | float64 | Yes |
| ECRS | None | float64 | Yes |
| NSPIN | None | float64 | Yes |
| REGDN | None | float64 | Yes |
| REGUP | None | float64 | Yes |
| RRS | None | float64 | Yes |


### DA_AS_combined_2024_03.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_combined/DA_AS_combined_2024_03.parquet`

**Size**: 52.9 KB

**Rows**: 288

**Last Modified**: 2025-08-18T16:05:22.284022


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DC_E | None | float64 | Yes |
| DC_L | None | float64 | Yes |
| DC_N | None | float64 | Yes |
| DC_R | None | float64 | Yes |
| DC_S | None | float64 | Yes |
| HB_BUSAVG | None | float64 | Yes |
| HB_HOUSTON | None | float64 | Yes |
| HB_HUBAVG | None | float64 | Yes |
| HB_NORTH | None | float64 | Yes |
| HB_PAN | None | float64 | Yes |
| HB_SOUTH | None | float64 | Yes |
| HB_WEST | None | float64 | Yes |
| LZ_AEN | None | float64 | Yes |
| LZ_CPS | None | float64 | Yes |
| LZ_HOUSTON | None | float64 | Yes |
| LZ_LCRA | None | float64 | Yes |
| LZ_NORTH | None | float64 | Yes |
| LZ_RAYBN | None | float64 | Yes |
| LZ_SOUTH | None | float64 | Yes |
| LZ_WEST | None | float64 | Yes |
| ECRS | None | float64 | Yes |
| NSPIN | None | float64 | Yes |
| REGDN | None | float64 | Yes |
| REGUP | None | float64 | Yes |
| RRS | None | float64 | Yes |


### DA_AS_combined_2024_04.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_combined/DA_AS_combined_2024_04.parquet`

**Size**: 52.9 KB

**Rows**: 288

**Last Modified**: 2025-08-18T16:05:22.285022


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DC_E | None | float64 | Yes |
| DC_L | None | float64 | Yes |
| DC_N | None | float64 | Yes |
| DC_R | None | float64 | Yes |
| DC_S | None | float64 | Yes |
| HB_BUSAVG | None | float64 | Yes |
| HB_HOUSTON | None | float64 | Yes |
| HB_HUBAVG | None | float64 | Yes |
| HB_NORTH | None | float64 | Yes |
| HB_PAN | None | float64 | Yes |
| HB_SOUTH | None | float64 | Yes |
| HB_WEST | None | float64 | Yes |
| LZ_AEN | None | float64 | Yes |
| LZ_CPS | None | float64 | Yes |
| LZ_HOUSTON | None | float64 | Yes |
| LZ_LCRA | None | float64 | Yes |
| LZ_NORTH | None | float64 | Yes |
| LZ_RAYBN | None | float64 | Yes |
| LZ_SOUTH | None | float64 | Yes |
| LZ_WEST | None | float64 | Yes |
| ECRS | None | float64 | Yes |
| NSPIN | None | float64 | Yes |
| REGDN | None | float64 | Yes |
| REGUP | None | float64 | Yes |
| RRS | None | float64 | Yes |


### DA_AS_combined_2024_05.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_combined/DA_AS_combined_2024_05.parquet`

**Size**: 53.1 KB

**Rows**: 288

**Last Modified**: 2025-08-18T16:05:22.287022


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DC_E | None | float64 | Yes |
| DC_L | None | float64 | Yes |
| DC_N | None | float64 | Yes |
| DC_R | None | float64 | Yes |
| DC_S | None | float64 | Yes |
| HB_BUSAVG | None | float64 | Yes |
| HB_HOUSTON | None | float64 | Yes |
| HB_HUBAVG | None | float64 | Yes |
| HB_NORTH | None | float64 | Yes |
| HB_PAN | None | float64 | Yes |
| HB_SOUTH | None | float64 | Yes |
| HB_WEST | None | float64 | Yes |
| LZ_AEN | None | float64 | Yes |
| LZ_CPS | None | float64 | Yes |
| LZ_HOUSTON | None | float64 | Yes |
| LZ_LCRA | None | float64 | Yes |
| LZ_NORTH | None | float64 | Yes |
| LZ_RAYBN | None | float64 | Yes |
| LZ_SOUTH | None | float64 | Yes |
| LZ_WEST | None | float64 | Yes |
| ECRS | None | float64 | Yes |
| NSPIN | None | float64 | Yes |
| REGDN | None | float64 | Yes |
| REGUP | None | float64 | Yes |
| RRS | None | float64 | Yes |


### DA_AS_combined_2024_06.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_combined/DA_AS_combined_2024_06.parquet`

**Size**: 53.6 KB

**Rows**: 288

**Last Modified**: 2025-08-18T16:05:22.288022


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DC_E | None | float64 | Yes |
| DC_L | None | float64 | Yes |
| DC_N | None | float64 | Yes |
| DC_R | None | float64 | Yes |
| DC_S | None | float64 | Yes |
| HB_BUSAVG | None | float64 | Yes |
| HB_HOUSTON | None | float64 | Yes |
| HB_HUBAVG | None | float64 | Yes |
| HB_NORTH | None | float64 | Yes |
| HB_PAN | None | float64 | Yes |
| HB_SOUTH | None | float64 | Yes |
| HB_WEST | None | float64 | Yes |
| LZ_AEN | None | float64 | Yes |
| LZ_CPS | None | float64 | Yes |
| LZ_HOUSTON | None | float64 | Yes |
| LZ_LCRA | None | float64 | Yes |
| LZ_NORTH | None | float64 | Yes |
| LZ_RAYBN | None | float64 | Yes |
| LZ_SOUTH | None | float64 | Yes |
| LZ_WEST | None | float64 | Yes |
| ECRS | None | float64 | Yes |
| NSPIN | None | float64 | Yes |
| REGDN | None | float64 | Yes |
| REGUP | None | float64 | Yes |
| RRS | None | float64 | Yes |


### DA_AS_combined_2024_07.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_combined/DA_AS_combined_2024_07.parquet`

**Size**: 53.0 KB

**Rows**: 288

**Last Modified**: 2025-08-18T16:05:22.290022


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DC_E | None | float64 | Yes |
| DC_L | None | float64 | Yes |
| DC_N | None | float64 | Yes |
| DC_R | None | float64 | Yes |
| DC_S | None | float64 | Yes |
| HB_BUSAVG | None | float64 | Yes |
| HB_HOUSTON | None | float64 | Yes |
| HB_HUBAVG | None | float64 | Yes |
| HB_NORTH | None | float64 | Yes |
| HB_PAN | None | float64 | Yes |
| HB_SOUTH | None | float64 | Yes |
| HB_WEST | None | float64 | Yes |
| LZ_AEN | None | float64 | Yes |
| LZ_CPS | None | float64 | Yes |
| LZ_HOUSTON | None | float64 | Yes |
| LZ_LCRA | None | float64 | Yes |
| LZ_NORTH | None | float64 | Yes |
| LZ_RAYBN | None | float64 | Yes |
| LZ_SOUTH | None | float64 | Yes |
| LZ_WEST | None | float64 | Yes |
| ECRS | None | float64 | Yes |
| NSPIN | None | float64 | Yes |
| REGDN | None | float64 | Yes |
| REGUP | None | float64 | Yes |
| RRS | None | float64 | Yes |


### DA_AS_combined_2024_08.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_combined/DA_AS_combined_2024_08.parquet`

**Size**: 52.9 KB

**Rows**: 288

**Last Modified**: 2025-08-18T16:05:22.291023


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DC_E | None | float64 | Yes |
| DC_L | None | float64 | Yes |
| DC_N | None | float64 | Yes |
| DC_R | None | float64 | Yes |
| DC_S | None | float64 | Yes |
| HB_BUSAVG | None | float64 | Yes |
| HB_HOUSTON | None | float64 | Yes |
| HB_HUBAVG | None | float64 | Yes |
| HB_NORTH | None | float64 | Yes |
| HB_PAN | None | float64 | Yes |
| HB_SOUTH | None | float64 | Yes |
| HB_WEST | None | float64 | Yes |
| LZ_AEN | None | float64 | Yes |
| LZ_CPS | None | float64 | Yes |
| LZ_HOUSTON | None | float64 | Yes |
| LZ_LCRA | None | float64 | Yes |
| LZ_NORTH | None | float64 | Yes |
| LZ_RAYBN | None | float64 | Yes |
| LZ_SOUTH | None | float64 | Yes |
| LZ_WEST | None | float64 | Yes |
| ECRS | None | float64 | Yes |
| NSPIN | None | float64 | Yes |
| REGDN | None | float64 | Yes |
| REGUP | None | float64 | Yes |
| RRS | None | float64 | Yes |


### DA_AS_combined_2024_09.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_combined/DA_AS_combined_2024_09.parquet`

**Size**: 52.8 KB

**Rows**: 288

**Last Modified**: 2025-08-18T16:05:22.292022


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DC_E | None | float64 | Yes |
| DC_L | None | float64 | Yes |
| DC_N | None | float64 | Yes |
| DC_R | None | float64 | Yes |
| DC_S | None | float64 | Yes |
| HB_BUSAVG | None | float64 | Yes |
| HB_HOUSTON | None | float64 | Yes |
| HB_HUBAVG | None | float64 | Yes |
| HB_NORTH | None | float64 | Yes |
| HB_PAN | None | float64 | Yes |
| HB_SOUTH | None | float64 | Yes |
| HB_WEST | None | float64 | Yes |
| LZ_AEN | None | float64 | Yes |
| LZ_CPS | None | float64 | Yes |
| LZ_HOUSTON | None | float64 | Yes |
| LZ_LCRA | None | float64 | Yes |
| LZ_NORTH | None | float64 | Yes |
| LZ_RAYBN | None | float64 | Yes |
| LZ_SOUTH | None | float64 | Yes |
| LZ_WEST | None | float64 | Yes |
| ECRS | None | float64 | Yes |
| NSPIN | None | float64 | Yes |
| REGDN | None | float64 | Yes |
| REGUP | None | float64 | Yes |
| RRS | None | float64 | Yes |


### DA_AS_combined_2024_10.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_combined/DA_AS_combined_2024_10.parquet`

**Size**: 53.2 KB

**Rows**: 287

**Last Modified**: 2025-08-18T16:05:22.294023


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DC_E | None | float64 | Yes |
| DC_L | None | float64 | Yes |
| DC_N | None | float64 | Yes |
| DC_R | None | float64 | Yes |
| DC_S | None | float64 | Yes |
| HB_BUSAVG | None | float64 | Yes |
| HB_HOUSTON | None | float64 | Yes |
| HB_HUBAVG | None | float64 | Yes |
| HB_NORTH | None | float64 | Yes |
| HB_PAN | None | float64 | Yes |
| HB_SOUTH | None | float64 | Yes |
| HB_WEST | None | float64 | Yes |
| LZ_AEN | None | float64 | Yes |
| LZ_CPS | None | float64 | Yes |
| LZ_HOUSTON | None | float64 | Yes |
| LZ_LCRA | None | float64 | Yes |
| LZ_NORTH | None | float64 | Yes |
| LZ_RAYBN | None | float64 | Yes |
| LZ_SOUTH | None | float64 | Yes |
| LZ_WEST | None | float64 | Yes |
| ECRS | None | float64 | Yes |
| NSPIN | None | float64 | Yes |
| REGDN | None | float64 | Yes |
| REGUP | None | float64 | Yes |
| RRS | None | float64 | Yes |


### DA_AS_combined_2024_11.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_combined/DA_AS_combined_2024_11.parquet`

**Size**: 52.6 KB

**Rows**: 288

**Last Modified**: 2025-08-18T16:05:22.295023


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DC_E | None | float64 | Yes |
| DC_L | None | float64 | Yes |
| DC_N | None | float64 | Yes |
| DC_R | None | float64 | Yes |
| DC_S | None | float64 | Yes |
| HB_BUSAVG | None | float64 | Yes |
| HB_HOUSTON | None | float64 | Yes |
| HB_HUBAVG | None | float64 | Yes |
| HB_NORTH | None | float64 | Yes |
| HB_PAN | None | float64 | Yes |
| HB_SOUTH | None | float64 | Yes |
| HB_WEST | None | float64 | Yes |
| LZ_AEN | None | float64 | Yes |
| LZ_CPS | None | float64 | Yes |
| LZ_HOUSTON | None | float64 | Yes |
| LZ_LCRA | None | float64 | Yes |
| LZ_NORTH | None | float64 | Yes |
| LZ_RAYBN | None | float64 | Yes |
| LZ_SOUTH | None | float64 | Yes |
| LZ_WEST | None | float64 | Yes |
| ECRS | None | float64 | Yes |
| NSPIN | None | float64 | Yes |
| REGDN | None | float64 | Yes |
| REGUP | None | float64 | Yes |
| RRS | None | float64 | Yes |


### DA_AS_combined_2024_12.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined/monthly/DA_AS_combined/DA_AS_combined_2024_12.parquet`

**Size**: 52.8 KB

**Rows**: 288

**Last Modified**: 2025-08-18T16:05:22.297023


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DC_E | None | float64 | Yes |
| DC_L | None | float64 | Yes |
| DC_N | None | float64 | Yes |
| DC_R | None | float64 | Yes |
| DC_S | None | float64 | Yes |
| HB_BUSAVG | None | float64 | Yes |
| HB_HOUSTON | None | float64 | Yes |
| HB_HUBAVG | None | float64 | Yes |
| HB_NORTH | None | float64 | Yes |
| HB_PAN | None | float64 | Yes |
| HB_SOUTH | None | float64 | Yes |
| HB_WEST | None | float64 | Yes |
| LZ_AEN | None | float64 | Yes |
| LZ_CPS | None | float64 | Yes |
| LZ_HOUSTON | None | float64 | Yes |
| LZ_LCRA | None | float64 | Yes |
| LZ_NORTH | None | float64 | Yes |
| LZ_RAYBN | None | float64 | Yes |
| LZ_SOUTH | None | float64 | Yes |
| LZ_WEST | None | float64 | Yes |
| ECRS | None | float64 | Yes |
| NSPIN | None | float64 | Yes |
| REGDN | None | float64 | Yes |
| REGUP | None | float64 | Yes |
| RRS | None | float64 | Yes |


## Flattened Price Data

### AS_prices_2010.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/flattened/AS_prices_2010.parquet`

**Size**: 21.8 KB

**Rows**: 744

**Last Modified**: 2025-08-18T17:46:21.128821


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| DeliveryDate | Date | object | Yes |
| DeliveryDateStr | String | object | Yes |
| datetime_ts | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| NSPIN | None | float64 | Yes |
| REGDN | None | float64 | Yes |
| REGUP | None | float64 | Yes |
| RRS | None | float64 | Yes |


### AS_prices_2011.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/flattened/AS_prices_2011.parquet`

**Size**: 215.3 KB

**Rows**: 8,759

**Last Modified**: 2025-08-18T17:46:21.177824


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| DeliveryDate | Date | object | Yes |
| DeliveryDateStr | String | object | Yes |
| datetime_ts | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| NSPIN | None | float64 | Yes |
| REGDN | None | float64 | Yes |
| REGUP | None | float64 | Yes |
| RRS | None | float64 | Yes |


### AS_prices_2012.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/flattened/AS_prices_2012.parquet`

**Size**: 201.0 KB

**Rows**: 8,783

**Last Modified**: 2025-08-18T17:46:21.227826


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| DeliveryDate | Date | object | Yes |
| DeliveryDateStr | String | object | Yes |
| datetime_ts | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| NSPIN | None | float64 | Yes |
| REGDN | None | float64 | Yes |
| REGUP | None | float64 | Yes |
| RRS | None | float64 | Yes |


### AS_prices_2013.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/flattened/AS_prices_2013.parquet`

**Size**: 201.7 KB

**Rows**: 8,759

**Last Modified**: 2025-08-18T17:46:21.277828


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| DeliveryDate | Date | object | Yes |
| DeliveryDateStr | String | object | Yes |
| datetime_ts | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| NSPIN | None | float64 | Yes |
| REGDN | None | float64 | Yes |
| REGUP | None | float64 | Yes |
| RRS | None | float64 | Yes |


### AS_prices_2014.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/flattened/AS_prices_2014.parquet`

**Size**: 214.4 KB

**Rows**: 8,759

**Last Modified**: 2025-08-18T17:46:21.326831


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| DeliveryDate | Date | object | Yes |
| DeliveryDateStr | String | object | Yes |
| datetime_ts | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| NSPIN | None | float64 | Yes |
| REGDN | None | float64 | Yes |
| REGUP | None | float64 | Yes |
| RRS | None | float64 | Yes |


### AS_prices_2023.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/flattened/AS_prices_2023.parquet`

**Size**: 227.6 KB

**Rows**: 8,759

**Last Modified**: 2025-08-18T17:46:21.772854


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| DeliveryDate | Date | object | Yes |
| DeliveryDateStr | String | object | Yes |
| datetime_ts | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| ECRS | None | float64 | Yes |
| NSPIN | None | float64 | Yes |
| REGDN | None | float64 | Yes |
| REGUP | None | float64 | Yes |
| RRS | None | float64 | Yes |


### AS_prices_2024.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/flattened/AS_prices_2024.parquet`

**Size**: 214.9 KB

**Rows**: 8,783

**Last Modified**: 2025-08-18T17:46:21.825856


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| DeliveryDate | Date | object | Yes |
| DeliveryDateStr | String | object | Yes |
| datetime_ts | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| ECRS | None | float64 | Yes |
| NSPIN | None | float64 | Yes |
| REGDN | None | float64 | Yes |
| REGUP | None | float64 | Yes |
| RRS | None | float64 | Yes |


### DA_prices_2023.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/flattened/DA_prices_2023.parquet`

**Size**: 783.3 KB

**Rows**: 8,759

**Last Modified**: 2025-08-18T17:46:11.031308


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| DeliveryDate | Date | object | Yes |
| DeliveryDateStr | String | object | Yes |
| datetime_ts | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DC_E | None | float64 | Yes |
| DC_L | None | float64 | Yes |
| DC_N | None | float64 | Yes |
| DC_R | None | float64 | Yes |
| DC_S | None | float64 | Yes |
| HB_BUSAVG | None | float64 | Yes |
| HB_HOUSTON | None | float64 | Yes |
| HB_HUBAVG | None | float64 | Yes |
| HB_NORTH | None | float64 | Yes |
| HB_PAN | None | float64 | Yes |
| HB_SOUTH | None | float64 | Yes |
| HB_WEST | None | float64 | Yes |
| LZ_AEN | None | float64 | Yes |
| LZ_CPS | None | float64 | Yes |
| LZ_HOUSTON | None | float64 | Yes |
| LZ_LCRA | None | float64 | Yes |
| LZ_NORTH | None | float64 | Yes |
| LZ_RAYBN | None | float64 | Yes |
| LZ_SOUTH | None | float64 | Yes |
| LZ_WEST | None | float64 | Yes |


### DA_prices_2024.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/flattened/DA_prices_2024.parquet`

**Size**: 742.5 KB

**Rows**: 8,783

**Last Modified**: 2025-08-18T17:46:17.094616


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| DeliveryDate | Date | object | Yes |
| DeliveryDateStr | String | object | Yes |
| datetime_ts | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DC_E | None | float64 | Yes |
| DC_L | None | float64 | Yes |
| DC_N | None | float64 | Yes |
| DC_R | None | float64 | Yes |
| DC_S | None | float64 | Yes |
| HB_BUSAVG | None | float64 | Yes |
| HB_HOUSTON | None | float64 | Yes |
| HB_HUBAVG | None | float64 | Yes |
| HB_NORTH | None | float64 | Yes |
| HB_PAN | None | float64 | Yes |
| HB_SOUTH | None | float64 | Yes |
| HB_WEST | None | float64 | Yes |
| LZ_AEN | None | float64 | Yes |
| LZ_CPS | None | float64 | Yes |
| LZ_HOUSTON | None | float64 | Yes |
| LZ_LCRA | None | float64 | Yes |
| LZ_NORTH | None | float64 | Yes |
| LZ_RAYBN | None | float64 | Yes |
| LZ_SOUTH | None | float64 | Yes |
| LZ_WEST | None | float64 | Yes |


### RT_prices_15min_2023.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/flattened/RT_prices_15min_2023.parquet`

**Size**: 2.3 MB

**Rows**: 33,577

**Last Modified**: 2025-08-18T17:46:41.901874


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| DeliveryDate | Date | object | Yes |
| DeliveryDateStr | String | object | Yes |
| datetime_ts | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DeliveryInterval | None | int64 | Yes |
| DC_E | None | float64 | Yes |
| DC_L | None | float64 | Yes |
| DC_N | None | float64 | Yes |
| DC_R | None | float64 | Yes |
| DC_S | None | float64 | Yes |
| HB_BUSAVG | None | float64 | Yes |
| HB_HOUSTON | None | float64 | Yes |
| HB_HUBAVG | None | float64 | Yes |
| HB_NORTH | None | float64 | Yes |
| HB_PAN | None | float64 | Yes |
| HB_SOUTH | None | float64 | Yes |
| HB_WEST | None | float64 | Yes |
| LZ_AEN | None | float64 | Yes |
| LZ_CPS | None | float64 | Yes |
| LZ_HOUSTON | None | float64 | Yes |
| LZ_LCRA | None | float64 | Yes |
| LZ_NORTH | None | float64 | Yes |
| LZ_RAYBN | None | float64 | Yes |
| LZ_SOUTH | None | float64 | Yes |
| LZ_WEST | None | float64 | Yes |


### RT_prices_15min_2024.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/flattened/RT_prices_15min_2024.parquet`

**Size**: 2.2 MB

**Rows**: 33,671

**Last Modified**: 2025-08-18T17:46:44.120986


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| DeliveryDate | Date | object | Yes |
| DeliveryDateStr | String | object | Yes |
| datetime_ts | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DeliveryInterval | None | int64 | Yes |
| DC_E | None | float64 | Yes |
| DC_L | None | float64 | Yes |
| DC_N | None | float64 | Yes |
| DC_R | None | float64 | Yes |
| DC_S | None | float64 | Yes |
| HB_BUSAVG | None | float64 | Yes |
| HB_HOUSTON | None | float64 | Yes |
| HB_HUBAVG | None | float64 | Yes |
| HB_NORTH | None | float64 | Yes |
| HB_PAN | None | float64 | Yes |
| HB_SOUTH | None | float64 | Yes |
| HB_WEST | None | float64 | Yes |
| LZ_AEN | None | float64 | Yes |
| LZ_CPS | None | float64 | Yes |
| LZ_HOUSTON | None | float64 | Yes |
| LZ_LCRA | None | float64 | Yes |
| LZ_NORTH | None | float64 | Yes |
| LZ_RAYBN | None | float64 | Yes |
| LZ_SOUTH | None | float64 | Yes |
| LZ_WEST | None | float64 | Yes |


### RT_prices_2023.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/flattened/RT_prices_2023.parquet`

**Size**: 1016.2 KB

**Rows**: 8,395

**Last Modified**: 2025-08-17T23:31:28.390006


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DC_E | None | float64 | Yes |
| DC_L | None | float64 | Yes |
| DC_N | None | float64 | Yes |
| DC_R | None | float64 | Yes |
| DC_S | None | float64 | Yes |
| HB_BUSAVG | None | float64 | Yes |
| HB_HOUSTON | None | float64 | Yes |
| HB_HUBAVG | None | float64 | Yes |
| HB_NORTH | None | float64 | Yes |
| HB_PAN | None | float64 | Yes |
| HB_SOUTH | None | float64 | Yes |
| HB_WEST | None | float64 | Yes |
| LZ_AEN | None | float64 | Yes |
| LZ_CPS | None | float64 | Yes |
| LZ_HOUSTON | None | float64 | Yes |
| LZ_LCRA | None | float64 | Yes |
| LZ_NORTH | None | float64 | Yes |
| LZ_RAYBN | None | float64 | Yes |
| LZ_SOUTH | None | float64 | Yes |
| LZ_WEST | None | float64 | Yes |


### RT_prices_2024.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/flattened/RT_prices_2024.parquet`

**Size**: 991.5 KB

**Rows**: 8,418

**Last Modified**: 2025-08-17T23:31:30.373100


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DC_E | None | float64 | Yes |
| DC_L | None | float64 | Yes |
| DC_N | None | float64 | Yes |
| DC_R | None | float64 | Yes |
| DC_S | None | float64 | Yes |
| HB_BUSAVG | None | float64 | Yes |
| HB_HOUSTON | None | float64 | Yes |
| HB_HUBAVG | None | float64 | Yes |
| HB_NORTH | None | float64 | Yes |
| HB_PAN | None | float64 | Yes |
| HB_SOUTH | None | float64 | Yes |
| HB_WEST | None | float64 | Yes |
| LZ_AEN | None | float64 | Yes |
| LZ_CPS | None | float64 | Yes |
| LZ_HOUSTON | None | float64 | Yes |
| LZ_LCRA | None | float64 | Yes |
| LZ_NORTH | None | float64 | Yes |
| LZ_RAYBN | None | float64 | Yes |
| LZ_SOUTH | None | float64 | Yes |
| LZ_WEST | None | float64 | Yes |


### RT_prices_5min_2023.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/flattened/RT_prices_5min_2023.parquet`

**Size**: 2.1 MB

**Rows**: 33,577

**Last Modified**: 2025-08-17T23:54:27.201789


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DeliveryInterval | None | int64 | Yes |
| DC_E | None | float64 | Yes |
| DC_L | None | float64 | Yes |
| DC_N | None | float64 | Yes |
| DC_R | None | float64 | Yes |
| DC_S | None | float64 | Yes |
| HB_BUSAVG | None | float64 | Yes |
| HB_HOUSTON | None | float64 | Yes |
| HB_HUBAVG | None | float64 | Yes |
| HB_NORTH | None | float64 | Yes |
| HB_PAN | None | float64 | Yes |
| HB_SOUTH | None | float64 | Yes |
| HB_WEST | None | float64 | Yes |
| LZ_AEN | None | float64 | Yes |
| LZ_CPS | None | float64 | Yes |
| LZ_HOUSTON | None | float64 | Yes |
| LZ_LCRA | None | float64 | Yes |
| LZ_NORTH | None | float64 | Yes |
| LZ_RAYBN | None | float64 | Yes |
| LZ_SOUTH | None | float64 | Yes |
| LZ_WEST | None | float64 | Yes |


### RT_prices_5min_2024.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/flattened/RT_prices_5min_2024.parquet`

**Size**: 2.0 MB

**Rows**: 33,671

**Last Modified**: 2025-08-17T23:54:29.219886


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DeliveryInterval | None | int64 | Yes |
| DC_E | None | float64 | Yes |
| DC_L | None | float64 | Yes |
| DC_N | None | float64 | Yes |
| DC_R | None | float64 | Yes |
| DC_S | None | float64 | Yes |
| HB_BUSAVG | None | float64 | Yes |
| HB_HOUSTON | None | float64 | Yes |
| HB_HUBAVG | None | float64 | Yes |
| HB_NORTH | None | float64 | Yes |
| HB_PAN | None | float64 | Yes |
| HB_SOUTH | None | float64 | Yes |
| HB_WEST | None | float64 | Yes |
| LZ_AEN | None | float64 | Yes |
| LZ_CPS | None | float64 | Yes |
| LZ_HOUSTON | None | float64 | Yes |
| LZ_LCRA | None | float64 | Yes |
| LZ_NORTH | None | float64 | Yes |
| LZ_RAYBN | None | float64 | Yes |
| LZ_SOUTH | None | float64 | Yes |
| LZ_WEST | None | float64 | Yes |


### RT_prices_hourly_2023.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/flattened/RT_prices_hourly_2023.parquet`

**Size**: 988.1 KB

**Rows**: 8,395

**Last Modified**: 2025-08-18T16:05:21.254974


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DC_E | None | float64 | Yes |
| DC_L | None | float64 | Yes |
| DC_N | None | float64 | Yes |
| DC_R | None | float64 | Yes |
| DC_S | None | float64 | Yes |
| HB_BUSAVG | None | float64 | Yes |
| HB_HOUSTON | None | float64 | Yes |
| HB_HUBAVG | None | float64 | Yes |
| HB_NORTH | None | float64 | Yes |
| HB_PAN | None | float64 | Yes |
| HB_SOUTH | None | float64 | Yes |
| HB_WEST | None | float64 | Yes |
| LZ_AEN | None | float64 | Yes |
| LZ_CPS | None | float64 | Yes |
| LZ_HOUSTON | None | float64 | Yes |
| LZ_LCRA | None | float64 | Yes |
| LZ_NORTH | None | float64 | Yes |
| LZ_RAYBN | None | float64 | Yes |
| LZ_SOUTH | None | float64 | Yes |
| LZ_WEST | None | float64 | Yes |


### RT_prices_hourly_2024.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/flattened/RT_prices_hourly_2024.parquet`

**Size**: 962.6 KB

**Rows**: 8,418

**Last Modified**: 2025-08-18T16:05:21.272975


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DC_E | None | float64 | Yes |
| DC_L | None | float64 | Yes |
| DC_N | None | float64 | Yes |
| DC_R | None | float64 | Yes |
| DC_S | None | float64 | Yes |
| HB_BUSAVG | None | float64 | Yes |
| HB_HOUSTON | None | float64 | Yes |
| HB_HUBAVG | None | float64 | Yes |
| HB_NORTH | None | float64 | Yes |
| HB_PAN | None | float64 | Yes |
| HB_SOUTH | None | float64 | Yes |
| HB_WEST | None | float64 | Yes |
| LZ_AEN | None | float64 | Yes |
| LZ_CPS | None | float64 | Yes |
| LZ_HOUSTON | None | float64 | Yes |
| LZ_LCRA | None | float64 | Yes |
| LZ_NORTH | None | float64 | Yes |
| LZ_RAYBN | None | float64 | Yes |
| LZ_SOUTH | None | float64 | Yes |
| LZ_WEST | None | float64 | Yes |


## Rollup - AS_prices

### 2010.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/AS_prices/2010.parquet`

**Size**: 12.4 KB

**Rows**: 2,976

**Last Modified**: 2025-08-18T16:46:50.791606


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| DeliveryDate | String | object | Yes |
| HourEnding | String | object | Yes |
| AncillaryType | String | object | Yes |
| MCPC | None | float64 | Yes |
| DSTFlag | String | object | Yes |
| hour | String | object | Yes |
| datetime | None | float64 | Yes |


### 2011.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/AS_prices/2011.parquet`

**Size**: 119.5 KB

**Rows**: 35,040

**Last Modified**: 2025-08-18T16:46:50.830608


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| DeliveryDate | String | object | Yes |
| HourEnding | String | object | Yes |
| AncillaryType | String | object | Yes |
| MCPC | None | float64 | Yes |
| DSTFlag | String | object | Yes |
| hour | String | object | Yes |
| datetime | None | float64 | Yes |


### 2012.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/AS_prices/2012.parquet`

**Size**: 113.4 KB

**Rows**: 35,136

**Last Modified**: 2025-08-18T16:46:50.864609


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| DeliveryDate | String | object | Yes |
| HourEnding | String | object | Yes |
| AncillaryType | String | object | Yes |
| MCPC | None | float64 | Yes |
| DSTFlag | String | object | Yes |
| hour | String | object | Yes |
| datetime | None | float64 | Yes |


### 2013.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/AS_prices/2013.parquet`

**Size**: 115.0 KB

**Rows**: 35,040

**Last Modified**: 2025-08-18T16:46:50.904611


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| DeliveryDate | String | object | Yes |
| HourEnding | String | object | Yes |
| AncillaryType | String | object | Yes |
| MCPC | None | float64 | Yes |
| DSTFlag | String | object | Yes |
| hour | String | object | Yes |
| datetime | None | float64 | Yes |


### 2014.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/AS_prices/2014.parquet`

**Size**: 123.0 KB

**Rows**: 35,040

**Last Modified**: 2025-08-18T16:46:50.939613


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| DeliveryDate | String | object | Yes |
| HourEnding | String | object | Yes |
| AncillaryType | String | object | Yes |
| MCPC | None | float64 | Yes |
| DSTFlag | String | object | Yes |
| hour | String | object | Yes |
| datetime | None | float64 | Yes |


### 2023.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/AS_prices/2023.parquet`

**Size**: 138.4 KB

**Rows**: 39,985

**Last Modified**: 2025-08-18T16:46:51.258628


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| DeliveryDate | String | object | Yes |
| HourEnding | String | object | Yes |
| AncillaryType | String | object | Yes |
| MCPC | None | float64 | Yes |
| DSTFlag | String | object | Yes |
| hour | String | object | Yes |
| datetime | None | float64 | Yes |


### 2024.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/AS_prices/2024.parquet`

**Size**: 138.5 KB

**Rows**: 43,920

**Last Modified**: 2025-08-18T16:46:51.294630


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| DeliveryDate | String | object | Yes |
| HourEnding | String | object | Yes |
| AncillaryType | String | object | Yes |
| MCPC | None | float64 | Yes |
| DSTFlag | String | object | Yes |
| hour | String | object | Yes |
| datetime | None | float64 | Yes |


## Rollup - COP_Snapshots

### 2011.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/COP_Snapshots/2011.parquet`

**Size**: 2.7 MB

**Rows**: 1,246,550

**Last Modified**: 2025-08-17T22:04:22.382799


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| DeliveryDate | Date | object | Yes |
| HourEnding | String | object | Yes |
| ResourceName | String | object | Yes |
| Status | String | object | Yes |
| HSL | None | float64 | Yes |
| LSL | None | float64 | Yes |
| MinSOC | None | float64 | Yes |
| MaxSOC | None | float64 | Yes |
| PlannedSOC | None | float64 | Yes |
| hour | None | float64 | Yes |
| datetime | None | float64 | Yes |


### 2012.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/COP_Snapshots/2012.parquet`

**Size**: 16.6 MB

**Rows**: 7,371,856

**Last Modified**: 2025-08-17T22:04:27.019021


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| DeliveryDate | Date | object | Yes |
| HourEnding | String | object | Yes |
| ResourceName | String | object | Yes |
| Status | String | object | Yes |
| HSL | None | float64 | Yes |
| LSL | None | float64 | Yes |
| MinSOC | None | float64 | Yes |
| MaxSOC | None | float64 | Yes |
| PlannedSOC | None | float64 | Yes |
| hour | None | float64 | Yes |
| datetime | None | float64 | Yes |


### 2013.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/COP_Snapshots/2013.parquet`

**Size**: 17.3 MB

**Rows**: 7,488,937

**Last Modified**: 2025-08-17T22:04:31.790249


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| DeliveryDate | Date | object | Yes |
| HourEnding | String | object | Yes |
| ResourceName | String | object | Yes |
| Status | String | object | Yes |
| HSL | None | float64 | Yes |
| LSL | None | float64 | Yes |
| MinSOC | None | float64 | Yes |
| MaxSOC | None | float64 | Yes |
| PlannedSOC | None | float64 | Yes |
| hour | None | float64 | Yes |
| datetime | None | float64 | Yes |


### 2014.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/COP_Snapshots/2014.parquet`

**Size**: 16.7 MB

**Rows**: 7,091,053

**Last Modified**: 2025-08-17T22:04:36.478473


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| DeliveryDate | Date | object | Yes |
| HourEnding | String | object | Yes |
| ResourceName | String | object | Yes |
| Status | String | object | Yes |
| HSL | None | float64 | Yes |
| LSL | None | float64 | Yes |
| MinSOC | None | float64 | Yes |
| MaxSOC | None | float64 | Yes |
| PlannedSOC | None | float64 | Yes |
| hour | None | float64 | Yes |
| datetime | None | float64 | Yes |


### 2015.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/COP_Snapshots/2015.parquet`

**Size**: 19.4 MB

**Rows**: 8,067,566

**Last Modified**: 2025-08-17T22:04:41.554716


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| DeliveryDate | Date | object | Yes |
| HourEnding | String | object | Yes |
| ResourceName | String | object | Yes |
| Status | String | object | Yes |
| HSL | None | float64 | Yes |
| LSL | None | float64 | Yes |
| MinSOC | None | float64 | Yes |
| MaxSOC | None | float64 | Yes |
| PlannedSOC | None | float64 | Yes |
| hour | None | float64 | Yes |
| datetime | None | float64 | Yes |


### 2023.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/COP_Snapshots/2023.parquet`

**Size**: 43.0 MB

**Rows**: 16,301,737

**Last Modified**: 2025-08-17T22:05:41.160565


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| DeliveryDate | Date | object | Yes |
| HourEnding | String | object | Yes |
| ResourceName | String | object | Yes |
| Status | String | object | Yes |
| HSL | None | float64 | Yes |
| LSL | None | float64 | Yes |
| MinSOC | None | float64 | Yes |
| MaxSOC | None | float64 | Yes |
| PlannedSOC | None | float64 | Yes |
| hour | None | float64 | Yes |
| datetime | None | float64 | Yes |


### 2024.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/COP_Snapshots/2024.parquet`

**Size**: 48.0 MB

**Rows**: 17,690,390

**Last Modified**: 2025-08-17T22:05:53.671163


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| DeliveryDate | Date | object | Yes |
| HourEnding | String | object | Yes |
| ResourceName | String | object | Yes |
| Status | String | object | Yes |
| HSL | None | float64 | Yes |
| LSL | None | float64 | Yes |
| MinSOC | None | float64 | Yes |
| MaxSOC | None | float64 | Yes |
| PlannedSOC | None | float64 | Yes |
| hour | None | float64 | Yes |
| datetime | None | float64 | Yes |


## Rollup - DAM_Gen_Resources

### 2011.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/DAM_Gen_Resources/2011.parquet`

**Size**: 968.2 KB

**Rows**: 107,136

**Last Modified**: 2025-08-18T16:46:52.093668


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| DeliveryDate | Date | object | Yes |
| HourEnding | String | object | Yes |
| ResourceName | String | object | Yes |
| ResourceType | String | object | Yes |
| SettlementPointName | String | object | Yes |
| LSL | None | float64 | Yes |
| HSL | None | float64 | Yes |
| AwardedQuantity | None | float64 | Yes |
| EnergySettlementPointPrice | None | float64 | Yes |
| RegUpAwarded | None | float64 | Yes |
| RegDownAwarded | None | float64 | Yes |
| RRSAwarded | None | float64 | Yes |
| RRSPFRAwarded | None | float64 | Yes |
| RRSFFRAwarded | None | float64 | Yes |
| RRSUFRAwarded | None | float64 | Yes |
| ECRSAwarded | None | float64 | Yes |
| ECRSSDAwarded | None | float64 | Yes |
| NonSpinAwarded | None | float64 | Yes |
| QSE submitted Curve-MW1 | None | float64 | Yes |
| QSE submitted Curve-Price1 | None | float64 | Yes |
| QSE submitted Curve-MW2 | None | float64 | Yes |
| QSE submitted Curve-Price2 | None | float64 | Yes |
| QSE submitted Curve-MW3 | None | float64 | Yes |
| QSE submitted Curve-Price3 | None | float64 | Yes |
| QSE submitted Curve-MW4 | None | float64 | Yes |
| QSE submitted Curve-Price4 | None | float64 | Yes |
| QSE submitted Curve-MW5 | None | float64 | Yes |
| QSE submitted Curve-Price5 | None | float64 | Yes |
| QSE submitted Curve-MW6 | None | float64 | Yes |
| QSE submitted Curve-Price6 | None | float64 | Yes |
| QSE submitted Curve-MW7 | None | float64 | Yes |
| QSE submitted Curve-Price7 | None | float64 | Yes |
| QSE submitted Curve-MW8 | None | float64 | Yes |
| QSE submitted Curve-Price8 | None | float64 | Yes |
| QSE submitted Curve-MW9 | None | float64 | Yes |
| QSE submitted Curve-Price9 | None | float64 | Yes |
| QSE submitted Curve-MW10 | None | float64 | Yes |
| QSE submitted Curve-Price10 | None | float64 | Yes |
| hour | None | int32 | Yes |
| datetime | None | float64 | Yes |


### 2012.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/DAM_Gen_Resources/2012.parquet`

**Size**: 48.8 MB

**Rows**: 5,579,280

**Last Modified**: 2025-08-18T16:47:03.591217


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| DeliveryDate | Date | object | Yes |
| HourEnding | String | object | Yes |
| ResourceName | String | object | Yes |
| ResourceType | String | object | Yes |
| SettlementPointName | String | object | Yes |
| LSL | None | float64 | Yes |
| HSL | None | float64 | Yes |
| AwardedQuantity | None | float64 | Yes |
| EnergySettlementPointPrice | None | float64 | Yes |
| RegUpAwarded | None | float64 | Yes |
| RegDownAwarded | None | float64 | Yes |
| RRSAwarded | None | float64 | Yes |
| RRSPFRAwarded | None | float64 | Yes |
| RRSFFRAwarded | None | float64 | Yes |
| RRSUFRAwarded | None | float64 | Yes |
| ECRSAwarded | None | float64 | Yes |
| ECRSSDAwarded | None | float64 | Yes |
| NonSpinAwarded | None | float64 | Yes |
| QSE submitted Curve-MW1 | None | float64 | Yes |
| QSE submitted Curve-Price1 | None | float64 | Yes |
| QSE submitted Curve-MW2 | None | float64 | Yes |
| QSE submitted Curve-Price2 | None | float64 | Yes |
| QSE submitted Curve-MW3 | None | float64 | Yes |
| QSE submitted Curve-Price3 | None | float64 | Yes |
| QSE submitted Curve-MW4 | None | float64 | Yes |
| QSE submitted Curve-Price4 | None | float64 | Yes |
| QSE submitted Curve-MW5 | None | float64 | Yes |
| QSE submitted Curve-Price5 | None | float64 | Yes |
| QSE submitted Curve-MW6 | None | float64 | Yes |
| QSE submitted Curve-Price6 | None | float64 | Yes |
| QSE submitted Curve-MW7 | None | float64 | Yes |
| QSE submitted Curve-Price7 | None | float64 | Yes |
| QSE submitted Curve-MW8 | None | float64 | Yes |
| QSE submitted Curve-Price8 | None | float64 | Yes |
| QSE submitted Curve-MW9 | None | float64 | Yes |
| QSE submitted Curve-Price9 | None | float64 | Yes |
| QSE submitted Curve-MW10 | None | float64 | Yes |
| QSE submitted Curve-Price10 | None | float64 | Yes |
| hour | None | int32 | Yes |
| datetime | None | float64 | Yes |


### 2013.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/DAM_Gen_Resources/2013.parquet`

**Size**: 46.5 MB

**Rows**: 5,195,560

**Last Modified**: 2025-08-18T16:47:14.208724


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| DeliveryDate | Date | object | Yes |
| HourEnding | String | object | Yes |
| ResourceName | String | object | Yes |
| ResourceType | String | object | Yes |
| SettlementPointName | String | object | Yes |
| LSL | None | float64 | Yes |
| HSL | None | float64 | Yes |
| AwardedQuantity | None | float64 | Yes |
| EnergySettlementPointPrice | None | float64 | Yes |
| RegUpAwarded | None | float64 | Yes |
| RegDownAwarded | None | float64 | Yes |
| RRSAwarded | None | float64 | Yes |
| RRSPFRAwarded | None | float64 | Yes |
| RRSFFRAwarded | None | float64 | Yes |
| RRSUFRAwarded | None | float64 | Yes |
| ECRSAwarded | None | float64 | Yes |
| ECRSSDAwarded | None | float64 | Yes |
| NonSpinAwarded | None | float64 | Yes |
| QSE submitted Curve-MW1 | None | float64 | Yes |
| QSE submitted Curve-Price1 | None | float64 | Yes |
| QSE submitted Curve-MW2 | None | float64 | Yes |
| QSE submitted Curve-Price2 | None | float64 | Yes |
| QSE submitted Curve-MW3 | None | float64 | Yes |
| QSE submitted Curve-Price3 | None | float64 | Yes |
| QSE submitted Curve-MW4 | None | float64 | Yes |
| QSE submitted Curve-Price4 | None | float64 | Yes |
| QSE submitted Curve-MW5 | None | float64 | Yes |
| QSE submitted Curve-Price5 | None | float64 | Yes |
| QSE submitted Curve-MW6 | None | float64 | Yes |
| QSE submitted Curve-Price6 | None | float64 | Yes |
| QSE submitted Curve-MW7 | None | float64 | Yes |
| QSE submitted Curve-Price7 | None | float64 | Yes |
| QSE submitted Curve-MW8 | None | float64 | Yes |
| QSE submitted Curve-Price8 | None | float64 | Yes |
| QSE submitted Curve-MW9 | None | float64 | Yes |
| QSE submitted Curve-Price9 | None | float64 | Yes |
| QSE submitted Curve-MW10 | None | float64 | Yes |
| QSE submitted Curve-Price10 | None | float64 | Yes |
| hour | None | int32 | Yes |
| datetime | None | float64 | Yes |


### 2014.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/DAM_Gen_Resources/2014.parquet`

**Size**: 50.6 MB

**Rows**: 5,684,818

**Last Modified**: 2025-08-18T16:47:25.579267


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| DeliveryDate | Date | object | Yes |
| HourEnding | String | object | Yes |
| ResourceName | String | object | Yes |
| ResourceType | String | object | Yes |
| SettlementPointName | String | object | Yes |
| LSL | None | float64 | Yes |
| HSL | None | float64 | Yes |
| AwardedQuantity | None | float64 | Yes |
| EnergySettlementPointPrice | None | float64 | Yes |
| RegUpAwarded | None | float64 | Yes |
| RegDownAwarded | None | float64 | Yes |
| RRSAwarded | None | float64 | Yes |
| RRSPFRAwarded | None | float64 | Yes |
| RRSFFRAwarded | None | float64 | Yes |
| RRSUFRAwarded | None | float64 | Yes |
| ECRSAwarded | None | float64 | Yes |
| ECRSSDAwarded | None | float64 | Yes |
| NonSpinAwarded | None | float64 | Yes |
| QSE submitted Curve-MW1 | None | float64 | Yes |
| QSE submitted Curve-Price1 | None | float64 | Yes |
| QSE submitted Curve-MW2 | None | float64 | Yes |
| QSE submitted Curve-Price2 | None | float64 | Yes |
| QSE submitted Curve-MW3 | None | float64 | Yes |
| QSE submitted Curve-Price3 | None | float64 | Yes |
| QSE submitted Curve-MW4 | None | float64 | Yes |
| QSE submitted Curve-Price4 | None | float64 | Yes |
| QSE submitted Curve-MW5 | None | float64 | Yes |
| QSE submitted Curve-Price5 | None | float64 | Yes |
| QSE submitted Curve-MW6 | None | float64 | Yes |
| QSE submitted Curve-Price6 | None | float64 | Yes |
| QSE submitted Curve-MW7 | None | float64 | Yes |
| QSE submitted Curve-Price7 | None | float64 | Yes |
| QSE submitted Curve-MW8 | None | float64 | Yes |
| QSE submitted Curve-Price8 | None | float64 | Yes |
| QSE submitted Curve-MW9 | None | float64 | Yes |
| QSE submitted Curve-Price9 | None | float64 | Yes |
| QSE submitted Curve-MW10 | None | float64 | Yes |
| QSE submitted Curve-Price10 | None | float64 | Yes |
| hour | None | int32 | Yes |
| datetime | None | float64 | Yes |


### 2015.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/DAM_Gen_Resources/2015.parquet`

**Size**: 53.0 MB

**Rows**: 5,966,878

**Last Modified**: 2025-08-18T16:47:37.447834


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| DeliveryDate | Date | object | Yes |
| HourEnding | String | object | Yes |
| ResourceName | String | object | Yes |
| ResourceType | String | object | Yes |
| SettlementPointName | String | object | Yes |
| LSL | None | float64 | Yes |
| HSL | None | float64 | Yes |
| AwardedQuantity | None | float64 | Yes |
| EnergySettlementPointPrice | None | float64 | Yes |
| RegUpAwarded | None | float64 | Yes |
| RegDownAwarded | None | float64 | Yes |
| RRSAwarded | None | float64 | Yes |
| RRSPFRAwarded | None | float64 | Yes |
| RRSFFRAwarded | None | float64 | Yes |
| RRSUFRAwarded | None | float64 | Yes |
| ECRSAwarded | None | float64 | Yes |
| ECRSSDAwarded | None | float64 | Yes |
| NonSpinAwarded | None | float64 | Yes |
| QSE submitted Curve-MW1 | None | float64 | Yes |
| QSE submitted Curve-Price1 | None | float64 | Yes |
| QSE submitted Curve-MW2 | None | float64 | Yes |
| QSE submitted Curve-Price2 | None | float64 | Yes |
| QSE submitted Curve-MW3 | None | float64 | Yes |
| QSE submitted Curve-Price3 | None | float64 | Yes |
| QSE submitted Curve-MW4 | None | float64 | Yes |
| QSE submitted Curve-Price4 | None | float64 | Yes |
| QSE submitted Curve-MW5 | None | float64 | Yes |
| QSE submitted Curve-Price5 | None | float64 | Yes |
| QSE submitted Curve-MW6 | None | float64 | Yes |
| QSE submitted Curve-Price6 | None | float64 | Yes |
| QSE submitted Curve-MW7 | None | float64 | Yes |
| QSE submitted Curve-Price7 | None | float64 | Yes |
| QSE submitted Curve-MW8 | None | float64 | Yes |
| QSE submitted Curve-Price8 | None | float64 | Yes |
| QSE submitted Curve-MW9 | None | float64 | Yes |
| QSE submitted Curve-Price9 | None | float64 | Yes |
| QSE submitted Curve-MW10 | None | float64 | Yes |
| QSE submitted Curve-Price10 | None | float64 | Yes |
| hour | None | int32 | Yes |
| datetime | None | float64 | Yes |


### 2023.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/DAM_Gen_Resources/2023.parquet`

**Size**: 93.8 MB

**Rows**: 9,906,322

**Last Modified**: 2025-08-18T12:09:33.206486


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| DeliveryDate | Date | object | Yes |
| HourEnding | String | object | Yes |
| ResourceName | String | object | Yes |
| ResourceType | String | object | Yes |
| SettlementPointName | String | object | Yes |
| LSL | None | float64 | Yes |
| HSL | None | float64 | Yes |
| AwardedQuantity | None | float64 | Yes |
| EnergySettlementPointPrice | None | float64 | Yes |
| RegUpAwarded | None | float64 | Yes |
| RegDownAwarded | None | float64 | Yes |
| RRSAwarded | None | float64 | Yes |
| RRSPFRAwarded | None | float64 | Yes |
| RRSFFRAwarded | None | float64 | Yes |
| RRSUFRAwarded | None | float64 | Yes |
| ECRSAwarded | None | float64 | Yes |
| ECRSSDAwarded | None | float64 | Yes |
| NonSpinAwarded | None | float64 | Yes |
| QSE submitted Curve-MW1 | None | float64 | Yes |
| QSE submitted Curve-Price1 | None | float64 | Yes |
| QSE submitted Curve-MW2 | None | float64 | Yes |
| QSE submitted Curve-Price2 | None | float64 | Yes |
| QSE submitted Curve-MW3 | None | float64 | Yes |
| QSE submitted Curve-Price3 | None | float64 | Yes |
| QSE submitted Curve-MW4 | None | float64 | Yes |
| QSE submitted Curve-Price4 | None | float64 | Yes |
| QSE submitted Curve-MW5 | None | float64 | Yes |
| QSE submitted Curve-Price5 | None | float64 | Yes |
| QSE submitted Curve-MW6 | None | float64 | Yes |
| QSE submitted Curve-Price6 | None | float64 | Yes |
| QSE submitted Curve-MW7 | None | float64 | Yes |
| QSE submitted Curve-Price7 | None | float64 | Yes |
| QSE submitted Curve-MW8 | None | float64 | Yes |
| QSE submitted Curve-Price8 | None | float64 | Yes |
| QSE submitted Curve-MW9 | None | float64 | Yes |
| QSE submitted Curve-Price9 | None | float64 | Yes |
| QSE submitted Curve-MW10 | None | float64 | Yes |
| QSE submitted Curve-Price10 | None | float64 | Yes |
| hour | None | int32 | Yes |
| datetime | None | float64 | Yes |


### 2024.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/DAM_Gen_Resources/2024.parquet`

**Size**: 112.2 MB

**Rows**: 11,847,736

**Last Modified**: 2025-08-18T12:09:57.176650


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| DeliveryDate | Date | object | Yes |
| HourEnding | String | object | Yes |
| ResourceName | String | object | Yes |
| ResourceType | String | object | Yes |
| SettlementPointName | String | object | Yes |
| LSL | None | float64 | Yes |
| HSL | None | float64 | Yes |
| AwardedQuantity | None | float64 | Yes |
| EnergySettlementPointPrice | None | float64 | Yes |
| RegUpAwarded | None | float64 | Yes |
| RegDownAwarded | None | float64 | Yes |
| RRSAwarded | None | float64 | Yes |
| RRSPFRAwarded | None | float64 | Yes |
| RRSFFRAwarded | None | float64 | Yes |
| RRSUFRAwarded | None | float64 | Yes |
| ECRSAwarded | None | float64 | Yes |
| ECRSSDAwarded | None | float64 | Yes |
| NonSpinAwarded | None | float64 | Yes |
| QSE submitted Curve-MW1 | None | float64 | Yes |
| QSE submitted Curve-Price1 | None | float64 | Yes |
| QSE submitted Curve-MW2 | None | float64 | Yes |
| QSE submitted Curve-Price2 | None | float64 | Yes |
| QSE submitted Curve-MW3 | None | float64 | Yes |
| QSE submitted Curve-Price3 | None | float64 | Yes |
| QSE submitted Curve-MW4 | None | float64 | Yes |
| QSE submitted Curve-Price4 | None | float64 | Yes |
| QSE submitted Curve-MW5 | None | float64 | Yes |
| QSE submitted Curve-Price5 | None | float64 | Yes |
| QSE submitted Curve-MW6 | None | float64 | Yes |
| QSE submitted Curve-Price6 | None | float64 | Yes |
| QSE submitted Curve-MW7 | None | float64 | Yes |
| QSE submitted Curve-Price7 | None | float64 | Yes |
| QSE submitted Curve-MW8 | None | float64 | Yes |
| QSE submitted Curve-Price8 | None | float64 | Yes |
| QSE submitted Curve-MW9 | None | float64 | Yes |
| QSE submitted Curve-Price9 | None | float64 | Yes |
| QSE submitted Curve-MW10 | None | float64 | Yes |
| QSE submitted Curve-Price10 | None | float64 | Yes |
| hour | None | int32 | Yes |
| datetime | None | float64 | Yes |


## Rollup - DAM_Load_Resources

### 2011.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/DAM_Load_Resources/2011.parquet`

**Size**: 80.5 KB

**Rows**: 30,720

**Last Modified**: 2025-08-18T12:56:08.739702


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| Delivery Date | String | object | Yes |
| DeliveryDate | Date | object | Yes |
| Hour Ending | String | object | Yes |
| Load Resource Name | String | object | Yes |
| Low Power Consumption for Load Resource | None | float64 | Yes |
| Max Power Consumption for Load Resource | None | float64 | Yes |
| NonSpin Awarded | None | float64 | Yes |
| NonSpin MCPC | None | float64 | Yes |
| RRS Awarded | None | float64 | Yes |
| RRS MCPC | None | float64 | Yes |
| RegDown Awarded | None | float64 | Yes |
| RegDown MCPC | None | float64 | Yes |
| RegUp Awarded | None | float64 | Yes |
| RegUp MCPC | None | float64 | Yes |
| datetime | None | float64 | Yes |
| hour | None | int32 | Yes |


### 2012.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/DAM_Load_Resources/2012.parquet`

**Size**: 4.1 MB

**Rows**: 1,732,161

**Last Modified**: 2025-08-18T12:56:09.821754


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| Delivery Date | String | object | Yes |
| DeliveryDate | Date | object | Yes |
| Hour Ending | String | object | Yes |
| Load Resource Name | String | object | Yes |
| Low Power Consumption for Load Resource | None | float64 | Yes |
| Max Power Consumption for Load Resource | None | float64 | Yes |
| NonSpin Awarded | None | float64 | Yes |
| NonSpin MCPC | None | float64 | Yes |
| RRS Awarded | None | float64 | Yes |
| RRS MCPC | None | float64 | Yes |
| RegDown Awarded | None | float64 | Yes |
| RegDown MCPC | None | float64 | Yes |
| RegUp Awarded | None | float64 | Yes |
| RegUp MCPC | None | float64 | Yes |
| datetime | None | float64 | Yes |
| hour | None | int32 | Yes |


### 2013.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/DAM_Load_Resources/2013.parquet`

**Size**: 4.1 MB

**Rows**: 1,727,616

**Last Modified**: 2025-08-18T12:56:10.925808


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| Delivery Date | String | object | Yes |
| DeliveryDate | Date | object | Yes |
| Hour Ending | String | object | Yes |
| Load Resource Name | String | object | Yes |
| Low Power Consumption for Load Resource | None | float64 | Yes |
| Max Power Consumption for Load Resource | None | float64 | Yes |
| NonSpin Awarded | None | float64 | Yes |
| NonSpin MCPC | None | float64 | Yes |
| RRS Awarded | None | float64 | Yes |
| RRS MCPC | None | float64 | Yes |
| RegDown Awarded | None | float64 | Yes |
| RegDown MCPC | None | float64 | Yes |
| RegUp Awarded | None | float64 | Yes |
| RegUp MCPC | None | float64 | Yes |
| datetime | None | float64 | Yes |
| hour | None | int32 | Yes |


### 2014.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/DAM_Load_Resources/2014.parquet`

**Size**: 4.7 MB

**Rows**: 1,984,626

**Last Modified**: 2025-08-18T12:56:12.175868


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| Delivery Date | String | object | Yes |
| DeliveryDate | Date | object | Yes |
| Hour Ending | String | object | Yes |
| Load Resource Name | String | object | Yes |
| Low Power Consumption for Load Resource | None | float64 | Yes |
| Max Power Consumption for Load Resource | None | float64 | Yes |
| NonSpin Awarded | None | float64 | Yes |
| NonSpin MCPC | None | float64 | Yes |
| RRS Awarded | None | float64 | Yes |
| RRS MCPC | None | float64 | Yes |
| RegDown Awarded | None | float64 | Yes |
| RegDown MCPC | None | float64 | Yes |
| RegUp Awarded | None | float64 | Yes |
| RegUp MCPC | None | float64 | Yes |
| datetime | None | float64 | Yes |
| hour | None | int32 | Yes |


### 2015.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/DAM_Load_Resources/2015.parquet`

**Size**: 5.1 MB

**Rows**: 2,037,616

**Last Modified**: 2025-08-18T12:56:13.505932


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| Delivery Date | String | object | Yes |
| DeliveryDate | Date | object | Yes |
| Hour Ending | String | object | Yes |
| Load Resource Name | String | object | Yes |
| Low Power Consumption for Load Resource | None | float64 | Yes |
| Max Power Consumption for Load Resource | None | float64 | Yes |
| NonSpin Awarded | None | float64 | Yes |
| NonSpin MCPC | None | float64 | Yes |
| RRS Awarded | None | float64 | Yes |
| RRS MCPC | None | float64 | Yes |
| RegDown Awarded | None | float64 | Yes |
| RegDown MCPC | None | float64 | Yes |
| RegUp Awarded | None | float64 | Yes |
| RegUp MCPC | None | float64 | Yes |
| datetime | None | float64 | Yes |
| hour | None | int32 | Yes |


### 2023.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/DAM_Load_Resources/2023.parquet`

**Size**: 17.6 MB

**Rows**: 5,277,770

**Last Modified**: 2025-08-18T12:56:33.889917


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| Delivery Date | String | object | Yes |
| DeliveryDate | Date | object | Yes |
| ECRS MCPC | None | float64 | Yes |
| ECRSMD Awarded | None | float64 | Yes |
| ECRSSD Awarded | None | float64 | Yes |
| Hour Ending | String | object | Yes |
| Load Resource Name | String | object | Yes |
| Low Power Consumption for Load Resource | None | float64 | Yes |
| Max Power Consumption for Load Resource | None | float64 | Yes |
| NonSpin Awarded | None | float64 | Yes |
| NonSpin MCPC | None | float64 | Yes |
| RRS MCPC | None | float64 | Yes |
| RRSFFR Awarded | None | float64 | Yes |
| RRSPFR Awarded | None | float64 | Yes |
| RRSUFR Awarded | None | float64 | Yes |
| RegDown Awarded | None | float64 | Yes |
| RegDown MCPC | None | float64 | Yes |
| RegUp Awarded | None | float64 | Yes |
| RegUp MCPC | None | float64 | Yes |
| datetime | None | float64 | Yes |
| hour | None | int32 | Yes |


### 2024.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/DAM_Load_Resources/2024.parquet`

**Size**: 19.7 MB

**Rows**: 5,811,917

**Last Modified**: 2025-08-18T12:56:37.772105


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| Delivery Date | String | object | Yes |
| DeliveryDate | Date | object | Yes |
| ECRS MCPC | None | float64 | Yes |
| ECRSMD Awarded | None | float64 | Yes |
| ECRSSD Awarded | None | float64 | Yes |
| Hour Ending | String | object | Yes |
| Load Resource Name | String | object | Yes |
| Low Power Consumption for Load Resource | None | float64 | Yes |
| Max Power Consumption for Load Resource | None | float64 | Yes |
| NonSpin Awarded | None | float64 | Yes |
| NonSpin MCPC | None | float64 | Yes |
| RRS MCPC | None | float64 | Yes |
| RRSFFR Awarded | None | float64 | Yes |
| RRSPFR Awarded | None | float64 | Yes |
| RRSUFR Awarded | None | float64 | Yes |
| RegDown Awarded | None | float64 | Yes |
| RegDown MCPC | None | float64 | Yes |
| RegUp Awarded | None | float64 | Yes |
| RegUp MCPC | None | float64 | Yes |
| datetime | None | float64 | Yes |
| hour | None | int32 | Yes |


## Rollup - DA_prices

### 2010.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/DA_prices/2010.parquet`

**Size**: 1.4 MB

**Rows**: 431,928

**Last Modified**: 2025-08-18T17:16:20.133984


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| DeliveryDate | Date | object | Yes |
| HourEnding | String | object | Yes |
| SettlementPoint | String | object | Yes |
| SettlementPointPrice | None | float64 | Yes |
| DSTFlag | String | object | Yes |
| hour | String | object | Yes |
| DeliveryDateStr | String | object | Yes |
| datetime_ms | None | float64 | Yes |


### 2011.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/DA_prices/2011.parquet`

**Size**: 15.7 MB

**Rows**: 4,880,420

**Last Modified**: 2025-08-18T17:16:20.700011


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| DeliveryDate | Date | object | Yes |
| HourEnding | String | object | Yes |
| SettlementPoint | String | object | Yes |
| SettlementPointPrice | None | float64 | Yes |
| DSTFlag | String | object | Yes |
| hour | String | object | Yes |
| DeliveryDateStr | String | object | Yes |
| datetime_ms | None | float64 | Yes |


### 2012.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/DA_prices/2012.parquet`

**Size**: 11.8 MB

**Rows**: 4,791,679

**Last Modified**: 2025-08-18T17:16:21.184035


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| DeliveryDate | Date | object | Yes |
| HourEnding | String | object | Yes |
| SettlementPoint | String | object | Yes |
| SettlementPointPrice | None | float64 | Yes |
| DSTFlag | String | object | Yes |
| hour | String | object | Yes |
| DeliveryDateStr | String | object | Yes |
| datetime_ms | None | float64 | Yes |


### 2013.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/DA_prices/2013.parquet`

**Size**: 12.0 MB

**Rows**: 4,865,271

**Last Modified**: 2025-08-18T17:16:21.676059


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| DeliveryDate | Date | object | Yes |
| HourEnding | String | object | Yes |
| SettlementPoint | String | object | Yes |
| SettlementPointPrice | None | float64 | Yes |
| DSTFlag | String | object | Yes |
| hour | String | object | Yes |
| DeliveryDateStr | String | object | Yes |
| datetime_ms | None | float64 | Yes |


### 2014.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/DA_prices/2014.parquet`

**Size**: 12.1 MB

**Rows**: 5,038,937

**Last Modified**: 2025-08-18T17:16:22.216085


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| DeliveryDate | Date | object | Yes |
| HourEnding | String | object | Yes |
| SettlementPoint | String | object | Yes |
| SettlementPointPrice | None | float64 | Yes |
| DSTFlag | String | object | Yes |
| hour | String | object | Yes |
| DeliveryDateStr | String | object | Yes |
| datetime_ms | None | float64 | Yes |


### 2023.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/DA_prices/2023.parquet`

**Size**: 14.8 MB

**Rows**: 7,428,915

**Last Modified**: 2025-08-18T17:34:51.658512


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| DeliveryDate | String | object | Yes |
| HourEnding | String | object | Yes |
| SettlementPoint | String | object | Yes |
| SettlementPointPrice | None | float64 | Yes |
| DSTFlag | String | object | Yes |
| hour | None | int64 | Yes |
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |


### 2024.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/DA_prices/2024.parquet`

**Size**: 21.5 MB

**Rows**: 8,033,160

**Last Modified**: 2025-08-18T17:13:04.081472


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| DeliveryDate | Date | object | Yes |
| HourEnding | String | object | Yes |
| SettlementPoint | String | object | Yes |
| SettlementPointPrice | None | float64 | Yes |
| DSTFlag | String | object | Yes |
| hour | String | object | Yes |
| datetime | None | float64 | Yes |


## Rollup - RT_prices

### 2010.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/RT_prices/2010.parquet`

**Size**: 10.5 MB

**Rows**: 1,728,292

**Last Modified**: 2025-08-17T22:20:11.605342


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| DeliveryDate | String | object | Yes |
| DeliveryHour | None | int64 | Yes |
| DeliveryInterval | None | int64 | Yes |
| SettlementPointName | String | object | Yes |
| SettlementPointType | String | object | Yes |
| SettlementPointPrice | None | float64 | Yes |
| DSTFlag | String | object | Yes |
| datetime | None | int64 | Yes |


### 2011.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/RT_prices/2011.parquet`

**Size**: 121.1 MB

**Rows**: 19,521,721

**Last Modified**: 2025-08-17T22:20:22.243859


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| DeliveryDate | String | object | Yes |
| DeliveryHour | None | int64 | Yes |
| DeliveryInterval | None | int64 | Yes |
| SettlementPointName | String | object | Yes |
| SettlementPointType | String | object | Yes |
| SettlementPointPrice | None | float64 | Yes |
| DSTFlag | String | object | Yes |
| datetime | None | int64 | Yes |


### 2012.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/RT_prices/2012.parquet`

**Size**: 35.6 MB

**Rows**: 19,556,052

**Last Modified**: 2025-08-17T22:27:04.598330


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| DeliveryDate | String | object | Yes |
| DeliveryHour | None | int64 | Yes |
| DeliveryInterval | None | int64 | Yes |
| SettlementPointName | String | object | Yes |
| SettlementPointType | String | object | Yes |
| SettlementPointPrice | None | float64 | Yes |
| DSTFlag | String | object | Yes |
| datetime | None | int64 | Yes |


### 2013.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/RT_prices/2013.parquet`

**Size**: 35.2 MB

**Rows**: 19,916,588

**Last Modified**: 2025-08-17T22:37:13.542597


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| DeliveryDate | String | object | Yes |
| DeliveryHour | None | int64 | Yes |
| DeliveryInterval | None | int64 | Yes |
| SettlementPointName | String | object | Yes |
| SettlementPointType | String | object | Yes |
| SettlementPointPrice | None | float64 | Yes |
| DSTFlag | String | object | Yes |
| datetime | None | int64 | Yes |


### 2014.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/RT_prices/2014.parquet`

**Size**: 120.6 MB

**Rows**: 20,611,248

**Last Modified**: 2025-08-17T22:21:06.964032


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| DeliveryDate | String | object | Yes |
| DeliveryHour | None | int64 | Yes |
| DeliveryInterval | None | int64 | Yes |
| SettlementPointName | String | object | Yes |
| SettlementPointType | String | object | Yes |
| SettlementPointPrice | None | float64 | Yes |
| DSTFlag | String | object | Yes |
| datetime | None | int64 | Yes |


### 2023.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/RT_prices/2023.parquet`

**Size**: 223.1 MB

**Rows**: 30,152,396

**Last Modified**: 2025-08-17T22:24:21.566460


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| DeliveryDate | String | object | Yes |
| DeliveryHour | None | int64 | Yes |
| DeliveryInterval | None | int64 | Yes |
| SettlementPointName | String | object | Yes |
| SettlementPointType | String | object | Yes |
| SettlementPointPrice | None | float64 | Yes |
| DSTFlag | String | object | Yes |
| datetime | None | int64 | Yes |


### 2024.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/RT_prices/2024.parquet`

**Size**: 253.9 MB

**Rows**: 32,553,950

**Last Modified**: 2025-08-17T22:24:41.496423


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| DeliveryDate | String | object | Yes |
| DeliveryHour | None | int64 | Yes |
| DeliveryInterval | None | int64 | Yes |
| SettlementPointName | String | object | Yes |
| SettlementPointType | String | object | Yes |
| SettlementPointPrice | None | float64 | Yes |
| DSTFlag | String | object | Yes |
| datetime | None | int64 | Yes |


## Rollup - SCED_Gen_Resources

### 2011.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/SCED_Gen_Resources/2011.parquet`

**Size**: 944.4 KB

**Rows**: 84,387

**Last Modified**: 2025-08-17T22:15:34.077886


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| AS_NSRS | None | float64 | Yes |
| AS_RRS | None | float64 | Yes |
| BasePoint | None | float64 | Yes |
| HSL | None | float64 | Yes |
| LSL | None | float64 | Yes |
| ResourceName | String | object | Yes |
| ResourceType | String | object | Yes |
| SCEDTimeStamp | String | object | Yes |
| datetime | String | object | Yes |


### 2012.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/SCED_Gen_Resources/2012.parquet`

**Size**: 175.3 MB

**Rows**: 15,445,674

**Last Modified**: 2025-08-17T19:42:59.546983


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| AS_NSRS | None | float64 | Yes |
| AS_RRS | None | float64 | Yes |
| BasePoint | None | float64 | Yes |
| HSL | None | float64 | Yes |
| LSL | None | float64 | Yes |
| ResourceName | String | object | Yes |
| ResourceType | String | object | Yes |
| SCEDTimeStamp | String | object | Yes |
| datetime | String | object | Yes |


### 2013.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/SCED_Gen_Resources/2013.parquet`

**Size**: 168.4 MB

**Rows**: 15,263,384

**Last Modified**: 2025-08-17T19:44:07.112160


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| AS_NSRS | None | float64 | Yes |
| AS_RRS | None | float64 | Yes |
| BasePoint | None | float64 | Yes |
| HSL | None | float64 | Yes |
| LSL | None | float64 | Yes |
| ResourceName | String | object | Yes |
| ResourceType | String | object | Yes |
| SCEDTimeStamp | String | object | Yes |
| datetime | String | object | Yes |


### 2014.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/SCED_Gen_Resources/2014.parquet`

**Size**: 65.9 MB

**Rows**: 14,582,238

**Last Modified**: 2025-08-17T19:46:05.096717


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| AS_NSRS | None | float64 | Yes |
| AS_RRS | None | float64 | Yes |
| BasePoint | None | float64 | Yes |
| HSL | None | float64 | Yes |
| LSL | None | float64 | Yes |
| ResourceName | String | object | Yes |
| ResourceType | String | object | Yes |
| SCEDTimeStamp | String | object | Yes |
| datetime | String | object | Yes |


### 2015.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/SCED_Gen_Resources/2015.parquet`

**Size**: 75.6 MB

**Rows**: 16,531,599

**Last Modified**: 2025-08-17T19:48:18.043991


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| AS_NSRS | None | float64 | Yes |
| AS_RRS | None | float64 | Yes |
| BasePoint | None | float64 | Yes |
| HSL | None | float64 | Yes |
| LSL | None | float64 | Yes |
| ResourceName | String | object | Yes |
| ResourceType | String | object | Yes |
| SCEDTimeStamp | String | object | Yes |
| datetime | String | object | Yes |


### 2023.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/SCED_Gen_Resources/2023.parquet`

**Size**: 125.6 MB

**Rows**: 23,188,158

**Last Modified**: 2025-08-17T20:07:33.427652


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| AS_ECRS | None | float64 | Yes |
| AS_NSRS | None | float64 | Yes |
| AS_RRS | None | float64 | Yes |
| AS_RRSFFR | None | float64 | Yes |
| BasePoint | None | float64 | Yes |
| HSL | None | float64 | Yes |
| LSL | None | float64 | Yes |
| ResourceName | String | object | Yes |
| ResourceType | String | object | Yes |
| SCEDTimeStamp | String | object | Yes |
| datetime | String | object | Yes |


### 2024.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/SCED_Gen_Resources/2024.parquet`

**Size**: 177.0 MB

**Rows**: 31,598,314

**Last Modified**: 2025-08-17T20:10:39.084439


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| AS_ECRS | None | float64 | Yes |
| AS_NSRS | None | float64 | Yes |
| AS_RRS | None | float64 | Yes |
| AS_RRSFFR | None | float64 | Yes |
| BasePoint | None | float64 | Yes |
| HSL | None | float64 | Yes |
| LSL | None | float64 | Yes |
| ResourceName | String | object | Yes |
| ResourceType | String | object | Yes |
| SCEDTimeStamp | String | object | Yes |
| datetime | String | object | Yes |


## Rollup - SCED_Load_Resources

### 2019.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/SCED_Load_Resources/2019.parquet`

**Size**: 11.7 MB

**Rows**: 3,162,528

**Last Modified**: 2025-08-18T12:59:00.088979


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| SCEDTimeStamp | String | object | Yes |
| MaxPowerConsumption | None | float64 | Yes |
| LDL | None | float64 | Yes |
| HDL | None | float64 | Yes |
| BasePoint | None | float64 | Yes |
| HASL | None | float64 | Yes |
| LASL | None | float64 | Yes |
| SCED Bid to Buy Curve-MW1 | None | float64 | Yes |
| SCED Bid to Buy Curve-Price1 | None | float64 | Yes |
| SCED Bid to Buy Curve-MW2 | None | float64 | Yes |
| SCED Bid to Buy Curve-Price2 | None | float64 | Yes |
| SCED Bid to Buy Curve-MW3 | None | float64 | Yes |
| SCED Bid to Buy Curve-Price3 | None | float64 | Yes |
| SCED Bid to Buy Curve-MW4 | None | float64 | Yes |
| SCED Bid to Buy Curve-Price4 | None | float64 | Yes |
| SCED Bid to Buy Curve-MW5 | None | float64 | Yes |
| SCED Bid to Buy Curve-Price5 | None | float64 | Yes |
| SCED Bid to Buy Curve-MW6 | None | float64 | Yes |
| SCED Bid to Buy Curve-Price6 | None | float64 | Yes |
| SCED Bid to Buy Curve-MW7 | None | float64 | Yes |
| SCED Bid to Buy Curve-Price7 | None | float64 | Yes |
| SCED Bid to Buy Curve-MW8 | None | float64 | Yes |
| SCED Bid to Buy Curve-Price8 | None | float64 | Yes |
| SCED Bid to Buy Curve-MW9 | None | float64 | Yes |
| SCED Bid to Buy Curve-Price9 | None | float64 | Yes |
| SCED Bid to Buy Curve-MW10 | None | float64 | Yes |
| SCED Bid to Buy Curve-Price10 | None | float64 | Yes |
| datetime | String | object | Yes |


### 2020.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/SCED_Load_Resources/2020.parquet`

**Size**: 51.1 MB

**Rows**: 13,806,740

**Last Modified**: 2025-08-18T12:59:19.956939


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| SCEDTimeStamp | String | object | Yes |
| MaxPowerConsumption | None | float64 | Yes |
| LDL | None | float64 | Yes |
| HDL | None | float64 | Yes |
| BasePoint | None | float64 | Yes |
| HASL | None | float64 | Yes |
| LASL | None | float64 | Yes |
| SCED Bid to Buy Curve-MW1 | None | float64 | Yes |
| SCED Bid to Buy Curve-Price1 | None | float64 | Yes |
| SCED Bid to Buy Curve-MW2 | None | float64 | Yes |
| SCED Bid to Buy Curve-Price2 | None | float64 | Yes |
| SCED Bid to Buy Curve-MW3 | None | float64 | Yes |
| SCED Bid to Buy Curve-Price3 | None | float64 | Yes |
| SCED Bid to Buy Curve-MW4 | None | float64 | Yes |
| SCED Bid to Buy Curve-Price4 | None | float64 | Yes |
| SCED Bid to Buy Curve-MW5 | None | float64 | Yes |
| SCED Bid to Buy Curve-Price5 | None | float64 | Yes |
| SCED Bid to Buy Curve-MW6 | None | float64 | Yes |
| SCED Bid to Buy Curve-Price6 | None | float64 | Yes |
| SCED Bid to Buy Curve-MW7 | None | float64 | Yes |
| SCED Bid to Buy Curve-Price7 | None | float64 | Yes |
| SCED Bid to Buy Curve-MW8 | None | float64 | Yes |
| SCED Bid to Buy Curve-Price8 | None | float64 | Yes |
| SCED Bid to Buy Curve-MW9 | None | float64 | Yes |
| SCED Bid to Buy Curve-Price9 | None | float64 | Yes |
| SCED Bid to Buy Curve-MW10 | None | float64 | Yes |
| SCED Bid to Buy Curve-Price10 | None | float64 | Yes |
| datetime | String | object | Yes |


### 2021.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/SCED_Load_Resources/2021.parquet`

**Size**: 90.4 MB

**Rows**: 23,465,177

**Last Modified**: 2025-08-18T12:59:48.221304


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| SCEDTimeStamp | String | object | Yes |
| MaxPowerConsumption | None | float64 | Yes |
| LDL | None | float64 | Yes |
| HDL | None | float64 | Yes |
| BasePoint | None | float64 | Yes |
| HASL | None | float64 | Yes |
| LASL | None | float64 | Yes |
| SCED Bid to Buy Curve-MW1 | None | float64 | Yes |
| SCED Bid to Buy Curve-Price1 | None | float64 | Yes |
| SCED Bid to Buy Curve-MW2 | None | float64 | Yes |
| SCED Bid to Buy Curve-Price2 | None | float64 | Yes |
| SCED Bid to Buy Curve-MW3 | None | float64 | Yes |
| SCED Bid to Buy Curve-Price3 | None | float64 | Yes |
| SCED Bid to Buy Curve-MW4 | None | float64 | Yes |
| SCED Bid to Buy Curve-Price4 | None | float64 | Yes |
| SCED Bid to Buy Curve-MW5 | None | float64 | Yes |
| SCED Bid to Buy Curve-Price5 | None | float64 | Yes |
| SCED Bid to Buy Curve-MW6 | None | float64 | Yes |
| SCED Bid to Buy Curve-Price6 | None | float64 | Yes |
| SCED Bid to Buy Curve-MW7 | None | float64 | Yes |
| SCED Bid to Buy Curve-Price7 | None | float64 | Yes |
| SCED Bid to Buy Curve-MW8 | None | float64 | Yes |
| SCED Bid to Buy Curve-Price8 | None | float64 | Yes |
| SCED Bid to Buy Curve-MW9 | None | float64 | Yes |
| SCED Bid to Buy Curve-Price9 | None | float64 | Yes |
| SCED Bid to Buy Curve-MW10 | None | float64 | Yes |
| SCED Bid to Buy Curve-Price10 | None | float64 | Yes |
| datetime | String | object | Yes |


### 2022.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/SCED_Load_Resources/2022.parquet`

**Size**: 98.3 MB

**Rows**: 23,177,304

**Last Modified**: 2025-08-18T13:00:12.137460


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| SCEDTimeStamp | String | object | Yes |
| MaxPowerConsumption | None | float64 | Yes |
| LDL | None | float64 | Yes |
| HDL | None | float64 | Yes |
| BasePoint | None | float64 | Yes |
| HASL | None | float64 | Yes |
| LASL | None | float64 | Yes |
| SCED Bid to Buy Curve-MW1 | None | float64 | Yes |
| SCED Bid to Buy Curve-Price1 | None | float64 | Yes |
| SCED Bid to Buy Curve-MW2 | None | float64 | Yes |
| SCED Bid to Buy Curve-Price2 | None | float64 | Yes |
| SCED Bid to Buy Curve-MW3 | None | float64 | Yes |
| SCED Bid to Buy Curve-Price3 | None | float64 | Yes |
| SCED Bid to Buy Curve-MW4 | None | float64 | Yes |
| SCED Bid to Buy Curve-Price4 | None | float64 | Yes |
| SCED Bid to Buy Curve-MW5 | None | float64 | Yes |
| SCED Bid to Buy Curve-Price5 | None | float64 | Yes |
| SCED Bid to Buy Curve-MW6 | None | float64 | Yes |
| SCED Bid to Buy Curve-Price6 | None | float64 | Yes |
| SCED Bid to Buy Curve-MW7 | None | float64 | Yes |
| SCED Bid to Buy Curve-Price7 | None | float64 | Yes |
| SCED Bid to Buy Curve-MW8 | None | float64 | Yes |
| SCED Bid to Buy Curve-Price8 | None | float64 | Yes |
| SCED Bid to Buy Curve-MW9 | None | float64 | Yes |
| SCED Bid to Buy Curve-Price9 | None | float64 | Yes |
| SCED Bid to Buy Curve-MW10 | None | float64 | Yes |
| SCED Bid to Buy Curve-Price10 | None | float64 | Yes |
| datetime | String | object | Yes |


### 2023.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/SCED_Load_Resources/2023.parquet`

**Size**: 73.2 MB

**Rows**: 15,588,219

**Last Modified**: 2025-08-18T13:00:28.925271


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| SCEDTimeStamp | String | object | Yes |
| MaxPowerConsumption | None | float64 | Yes |
| LDL | None | float64 | Yes |
| HDL | None | float64 | Yes |
| BasePoint | None | float64 | Yes |
| HASL | None | float64 | Yes |
| LASL | None | float64 | Yes |
| SCED Bid to Buy Curve-MW1 | None | float64 | Yes |
| SCED Bid to Buy Curve-Price1 | None | float64 | Yes |
| SCED Bid to Buy Curve-MW2 | None | float64 | Yes |
| SCED Bid to Buy Curve-Price2 | None | float64 | Yes |
| SCED Bid to Buy Curve-MW3 | None | float64 | Yes |
| SCED Bid to Buy Curve-Price3 | None | float64 | Yes |
| SCED Bid to Buy Curve-MW4 | None | float64 | Yes |
| SCED Bid to Buy Curve-Price4 | None | float64 | Yes |
| SCED Bid to Buy Curve-MW5 | None | float64 | Yes |
| SCED Bid to Buy Curve-Price5 | None | float64 | Yes |
| SCED Bid to Buy Curve-MW6 | None | float64 | Yes |
| SCED Bid to Buy Curve-Price6 | None | float64 | Yes |
| SCED Bid to Buy Curve-MW7 | None | float64 | Yes |
| SCED Bid to Buy Curve-Price7 | None | float64 | Yes |
| SCED Bid to Buy Curve-MW8 | None | float64 | Yes |
| SCED Bid to Buy Curve-Price8 | None | float64 | Yes |
| SCED Bid to Buy Curve-MW9 | None | float64 | Yes |
| SCED Bid to Buy Curve-Price9 | None | float64 | Yes |
| SCED Bid to Buy Curve-MW10 | None | float64 | Yes |
| SCED Bid to Buy Curve-Price10 | None | float64 | Yes |
| datetime | String | object | Yes |


### 2024.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/SCED_Load_Resources/2024.parquet`

**Size**: 122.1 MB

**Rows**: 19,089,805

**Last Modified**: 2025-08-18T13:00:51.053339


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| SCEDTimeStamp | String | object | Yes |
| MaxPowerConsumption | None | float64 | Yes |
| LDL | None | float64 | Yes |
| HDL | None | float64 | Yes |
| BasePoint | None | float64 | Yes |
| HASL | None | float64 | Yes |
| LASL | None | float64 | Yes |
| SCED Bid to Buy Curve-MW1 | None | float64 | Yes |
| SCED Bid to Buy Curve-Price1 | None | float64 | Yes |
| SCED Bid to Buy Curve-MW2 | None | float64 | Yes |
| SCED Bid to Buy Curve-Price2 | None | float64 | Yes |
| SCED Bid to Buy Curve-MW3 | None | float64 | Yes |
| SCED Bid to Buy Curve-Price3 | None | float64 | Yes |
| SCED Bid to Buy Curve-MW4 | None | float64 | Yes |
| SCED Bid to Buy Curve-Price4 | None | float64 | Yes |
| SCED Bid to Buy Curve-MW5 | None | float64 | Yes |
| SCED Bid to Buy Curve-Price5 | None | float64 | Yes |
| SCED Bid to Buy Curve-MW6 | None | float64 | Yes |
| SCED Bid to Buy Curve-Price6 | None | float64 | Yes |
| SCED Bid to Buy Curve-MW7 | None | float64 | Yes |
| SCED Bid to Buy Curve-Price7 | None | float64 | Yes |
| SCED Bid to Buy Curve-MW8 | None | float64 | Yes |
| SCED Bid to Buy Curve-Price8 | None | float64 | Yes |
| SCED Bid to Buy Curve-MW9 | None | float64 | Yes |
| SCED Bid to Buy Curve-Price9 | None | float64 | Yes |
| SCED Bid to Buy Curve-MW10 | None | float64 | Yes |
| SCED Bid to Buy Curve-Price10 | None | float64 | Yes |
| datetime | String | object | Yes |


## Rollup - combined_test

### DA_AS_combined_2024.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/combined_test/DA_AS_combined_2024.parquet`

**Size**: 777.2 KB

**Rows**: 8,783

**Last Modified**: 2025-08-18T17:30:28.524870


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DC_E | None | float64 | Yes |
| DC_L | None | float64 | Yes |
| DC_N | None | float64 | Yes |
| DC_R | None | float64 | Yes |
| DC_S | None | float64 | Yes |
| HB_BUSAVG | None | float64 | Yes |
| HB_HOUSTON | None | float64 | Yes |
| HB_HUBAVG | None | float64 | Yes |
| HB_NORTH | None | float64 | Yes |
| HB_PAN | None | float64 | Yes |
| HB_SOUTH | None | float64 | Yes |
| HB_WEST | None | float64 | Yes |
| LZ_AEN | None | float64 | Yes |
| LZ_CPS | None | float64 | Yes |
| LZ_HOUSTON | None | float64 | Yes |
| LZ_LCRA | None | float64 | Yes |
| LZ_NORTH | None | float64 | Yes |
| LZ_RAYBN | None | float64 | Yes |
| LZ_SOUTH | None | float64 | Yes |
| LZ_WEST | None | float64 | Yes |
| ECRS | None | float64 | Yes |
| NSPIN | None | float64 | Yes |
| REGDN | None | float64 | Yes |
| REGUP | None | float64 | Yes |
| RRS | None | float64 | Yes |


## Rollup - flattened_test

### AS_prices_2024.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/flattened_test/AS_prices_2024.parquet`

**Size**: 162.0 KB

**Rows**: 8,783

**Last Modified**: 2025-08-18T17:30:15.683253


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| ECRS | None | float64 | Yes |
| NSPIN | None | float64 | Yes |
| REGDN | None | float64 | Yes |
| REGUP | None | float64 | Yes |
| RRS | None | float64 | Yes |


### DA_prices_2024.parquet

**Path**: `/home/enrico/data/ERCOT_data/rollup_files/flattened_test/DA_prices_2024.parquet`

**Size**: 693.6 KB

**Rows**: 8,783

**Last Modified**: 2025-08-18T17:30:15.647251


#### Schema

| Column | Type | Pandas Type | Nullable |
|--------|------|-------------|----------|
| datetime | Timestamp(isAdjustedToUTC=false, timeUnit=nanoseconds, is_from_converted_type=false, force_set_converted_type=false) | datetime64[ns] | Yes |
| DC_E | None | float64 | Yes |
| DC_L | None | float64 | Yes |
| DC_N | None | float64 | Yes |
| DC_R | None | float64 | Yes |
| DC_S | None | float64 | Yes |
| HB_BUSAVG | None | float64 | Yes |
| HB_HOUSTON | None | float64 | Yes |
| HB_HUBAVG | None | float64 | Yes |
| HB_NORTH | None | float64 | Yes |
| HB_PAN | None | float64 | Yes |
| HB_SOUTH | None | float64 | Yes |
| HB_WEST | None | float64 | Yes |
| LZ_AEN | None | float64 | Yes |
| LZ_CPS | None | float64 | Yes |
| LZ_HOUSTON | None | float64 | Yes |
| LZ_LCRA | None | float64 | Yes |
| LZ_NORTH | None | float64 | Yes |
| LZ_RAYBN | None | float64 | Yes |
| LZ_SOUTH | None | float64 | Yes |
| LZ_WEST | None | float64 | Yes |

