# Error analysis (TEST set, SYNTHETIC-TARGET)

## +3 h level MAE by rain in the last 6 h (cm)

| rain6            |   count |   mean |   max |
|:-----------------|--------:|-------:|------:|
| dry (<1 mm/6h)   |   24447 |   0.69 | 40.37 |
| light (1-10)     |     292 |   2.96 | 51.97 |
| moderate (10-30) |     247 |   3.6  | 45.91 |
| heavy (>30)      |     168 |  12.92 | 80.78 |


## +3 h level MAE by current level (cm)

| lr             |   count |   mean |   max |
|:---------------|--------:|-------:|------:|
| <25% of danger |   24094 |   0.7  | 51.97 |
| 25-50%         |     512 |   2.07 | 45.17 |
| 50-100%        |     287 |   2.77 | 45.91 |
| >=100%         |     261 |   8.62 | 80.78 |


## Effect of the rain forecast input

| fc                 |   count |   mean |
|:-------------------|--------:|-------:|
| forecast available |   18725 |   0.82 |
| no forecast        |    6429 |   0.87 |


## Flood events in TEST: detection and lead time

| station   | event_hour       |   peak_rain_mmh_6h_before |   level_3h_before_cm |   max_prob_3h_before |   lead_min | detected   |
|:----------|:-----------------|--------------------------:|---------------------:|---------------------:|-----------:|:-----------|
| akota     | 2025-06-29 08:00 |                      20.2 |                 24.2 |                0.742 |        120 | True       |
| akota     | 2025-07-02 18:00 |                       9.4 |                 30.7 |                0.867 |        180 | True       |
| akota     | 2025-07-26 08:00 |                      18.8 |                 31   |                1     |        180 | True       |
| akota     | 2025-08-10 06:00 |                      12.9 |                 39.4 |                0.874 |        180 | True       |
| alkapuri  | 2025-06-29 06:00 |                      11.9 |                  6.8 |                1     |        180 | True       |
| alkapuri  | 2025-07-02 17:00 |                       5.4 |                 13   |                1     |        180 | True       |
| alkapuri  | 2025-07-26 06:00 |                      14.3 |                  8.7 |                1     |        120 | True       |
| alkapuri  | 2025-07-30 04:00 |                       8.7 |                  9   |                0.742 |        180 | True       |
| kamnath   | 2025-07-02 23:00 |                      20.1 |                 69.6 |                1     |        180 | True       |
| kamnath   | 2025-07-26 13:00 |                      25.5 |                 68.7 |                0.994 |        180 | True       |


Missed events: 0 of 10.

Caveat: a single test year holds only a handful of events, so every number has wide uncertainty. Hourly resolution also limits how early a very fast rise can be seen.
