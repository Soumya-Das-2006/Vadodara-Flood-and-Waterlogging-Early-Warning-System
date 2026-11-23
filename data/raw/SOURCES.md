# Data sources — status in THIS build

The build sandbox had no internet access to data sites (curl returned no response for
archive-api.open-meteo.com, flood-api.open-meteo.com, api.open-meteo.com, indiawris.gov.in,
kaggle.com, zenodo.org, data.gov.in). So **no gridded or gauge data could be downloaded here.**
Nothing below was invented.

| # | Source | Status | Label |
|---|---|---|---|
| 1 | Open-Meteo Historical (ERA5) https://archive-api.open-meteo.com/v1/archive | NOT downloaded (blocked). Run `make data-real` | REAL when downloaded |
| 2 | Open-Meteo Flood (GloFAS) https://flood-api.open-meteo.com/v1/flood | NOT downloaded (blocked). `make data-real` | REAL (model-based) |
| 3 | Open-Meteo Forecast https://api.open-meteo.com/v1/forecast | Used live by the backend if reachable | REAL |
| 4 | IMD / GPM IMERG / CHIRPS | NOT attempted (need login or manual download) | - |
| 5 | India-WRIS / CWC gauges | NOT downloaded (blocked, portal needs manual export) | - |
| 6 | Kaggle / Zenodo / data.gov.in | NOT downloaded (blocked, Kaggle needs login) | - |
| 7 | Historical flood events | `flood_events.csv` (3 rows, from news, URL per row) | REAL (news report) |

Licence: Open-Meteo data is CC BY 4.0 (attribution required). Flood event rows are facts quoted
from the linked news pages; follow each site's terms.

Street-level and gauge levels at Akota / Alkapuri / Kamnath do not exist as open data, so they are
always a **SYNTHETIC-TARGET** (see README).
