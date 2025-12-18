# Data Dictionary Summary

This document provides mapping logic and data definitions for the KMB/LWB bus datasets and Transport Department traffic data.

## 1. KMB/LWB Bus Data (etabus.gov.hk)

### 1.1 Core Relationships
- **Route Segment**: A combination of `route` (e.g., 68X), `bound` (I/O), and `service_type` (1=main).
- **Stop Sequence**: Obtained from the `Route-Stop API`. Field `seq` (1-indexed) defines the order of stops.
- **ETA Mapping**: Both `ETA` and `Route-ETA` APIs include a `seq` field. This **MUST** be matched to the `seq` from `Route-Stop` to correctly identify the stop's position in the journey.

### 1.2 Key Fields for Calibration
| Entity | Field | Type | Description |
| :--- | :--- | :--- | :--- |
| **Stop** | `lat`, `long` | Float | WGS84 coordinates of the bus stop. |
| **ETA** | `eta` | ISO 8601 | Estimated arrival time. `null` if no data. |
| **ETA** | `eta_seq` | Integer | Sequence of upcoming buses (1=next, 2=following, etc.). |
| **ETA** | `data_timestamp` | ISO 8601 | Time when the ETA was calculated by the server. |
| **ETA** | `rmk_en` | String | Remarks (e.g., "Scheduled Bus"). |

### 1.3 Arrival/Departure Time Derivation
Since the API provides *predictions*, "Actual Arrival" must be inferred:
1. Track a specific vehicle (if possible, though the API doesn't provide persistent vehicle IDs, `eta_seq=1` is the best proxy per run).
2. Record the `data_timestamp` of the last snapshot where `eta` was in the future.
3. Record the `data_timestamp` when the `eta` disappears or is marked as arrived (if the API supports it, otherwise capture the gap).

---

## 2. Traffic Data (data.gov.hk)

### 2.1 Raw Detector Data
- **Location**: `detector_id` maps to CSV `Latitude`/`Longitude`.
- **Metrics**: Speed (km/h), Volume, Occupancy (%) per lane.
- **Frequency**: 30-second intervals.

### 2.2 Processed Segment Data (IRN)
- **Mapping**: `segment_id` corresponds to `ROUTE_ID` in the IRN `CENTERLINE` shapefile.
- **Metrics**: `speed` (average km/h).
- **Frequency**: 1-minute or 2-minute updates.

---

## 3. Bus Line Shapes (GeoJSON)
- **Source**: `JSON_BUS.json`.
- **Attributes**: `routeId`, `routeSeq` (outbound/inbound), `stopSeq`.
- **Geometry**: `LineString` for the bus path.
- **Usage**: Calculation of cumulative mileage between stops.

---

## 4. "Gotchas" and Warnings
- **Daily Update**: Static data (Routes, Stops) is updated at **05:00 HKT daily**. Downloaders should refresh daily.
- **Case Sensitivity**: URL parameters and values are case-sensitive.
- **Service Types**: A single route (e.g., 960) may have multiple `service_type` values (special morning trips, etc.). Always default to `service_type=1` for baseline.
- **ETA Precision**: ETAs are provided in 1-minute or variable precision. The `data_timestamp` is crucial for aligning snapshots.
