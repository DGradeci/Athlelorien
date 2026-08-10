# Data And Asset Licensing

This repository combines original software, author-created research outputs,
and third-party data. The root MIT licence does not apply to every file.

## Software

Original source code and software documentation are licensed under the MIT
License in `LICENSE`.

## Author-created figures and aggregate source tables

The paper figures and aggregate, non-player-identifying source tables under
`analysis/levy_paper/figures/` are licensed by the paper authors under the
Creative Commons Attribution 4.0 International licence (CC BY 4.0):

https://creativecommons.org/licenses/by/4.0/

This permission covers the authors' analytical outputs only. It does not grant
rights to the underlying Soccermon tracking records, NFF football data, or
OpenStreetMap database content.

## Soccermon tracking data

Raw Soccermon tracking data, player-level trajectories, device records, and
large processed caches are not distributed in this repository and are not
covered by either MIT or CC BY 4.0. Access to the underlying AWS/S3 data is
restricted to authorised collaborators under the applicable data agreement.

The tracked figure-source package contains aggregate analytical outputs used
to inspect the paper figures. It does not contain player names, device
identifiers, raw GPS coordinates, or individual player trajectories.

## Match schedules

The analysis uses season schedule exports from Norges Fotballforbund (NFF).
Original NFF spreadsheet exports are not redistributed in this repository and
are not covered by its licences. Authorised collaborators must obtain or
supply them separately and place them at the documented local paths. See
`analysis/levy_paper/metadata/schedules/README.md`.

## Pitch geometry

The pitch registry under `analysis/levy_paper/metadata/pitches/` contains
OpenStreetMap data retrieved through the Overpass API. OpenStreetMap data are
licensed under the Open Data Commons Open Database License (ODbL) 1.0.

Attribution: (c) OpenStreetMap contributors

https://www.openstreetmap.org/copyright

See `analysis/levy_paper/metadata/pitches/README.md` for file-specific details.

## Scope

Where third-party terms conflict with this summary, the third-party terms
control. No rights are granted to material that the repository authors do not
own or have authority to license.
