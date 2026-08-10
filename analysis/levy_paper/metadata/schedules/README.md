# Local Match Schedule Inputs

The raw AWS cache builder requires season schedule workbooks to identify
competitive match dates, scheduled kick-off times, teams, and stadiums. The
original Norges Fotballforbund (NFF) exports are intentionally not distributed
in this repository.

Authorised collaborators should obtain the relevant Toppserien schedule export
from NFF and place the files at:

```text
analysis/levy_paper/metadata/schedules/kamper_2020.xlsx
analysis/levy_paper/metadata/schedules/kamper_2021.xlsx
```

Official season page used for the 2020 schedule:

https://www.fotball.no/fotballdata/turnering/hjem/?fiksId=169786

For later seasons, use the corresponding official NFF Toppserien season page
and its `Last ned til Excel` export. Obtain and use NFF data in accordance with
NFF's applicable terms.

## Required workbook schema

The loader accepts the Norwegian NFF column labels and normalises them
internally. Each workbook must contain one row per fixture and these fields:

| NFF column | Meaning | Example |
| --- | --- | --- |
| `Dato` | Match date | `2020-07-03` |
| `Tid` | Scheduled local kick-off time | `19:45` |
| `Hjemmelag` | Home team | `Rosenborg` |
| `Bortelag` | Away team | `LSK Kvinner` |
| `Bane` | Scheduled stadium/pitch | `Koteng Arena` |
| `Resultat` | Final score, when available | `1 - 1` |
| `Kampnummer` | NFF fixture identifier | `99220001003` |

Additional NFF columns may remain present. The publication reviewer and local
processed-cache modes do not require these workbooks. They are required only
when rebuilding processed caches from raw AWS/S3 tracking data.
