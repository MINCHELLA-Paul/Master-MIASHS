# Bike Sharing data provenance

Source: UCI Machine Learning Repository, Bike Sharing, by Hadi Fanaee-T.
Dataset DOI: https://doi.org/10.24432/C5W894
Dataset page: https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset
Official archive: https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip
License on the current UCI page: Creative Commons Attribution 4.0 International (CC BY 4.0).
License: https://creativecommons.org/licenses/by/4.0/

Associated paper: Hadi Fanaee-T and João Gama (2013), Event labeling combining
ensemble detectors and background knowledge. Progress in Artificial Intelligence.
https://doi.org/10.1007/s13748-013-0040-3

The archive and day.csv are unmodified. Readme.txt is the original archive member.
The notebook uses only day.csv (731 days), not hour.csv.
The complete archive includes hour.csv but the notebook does not extract it.

SHA-256:
- bike_sharing.zip: b70182d0d0508e9abbb79306ce5c0cec34869000f8220175ac83d11dbe845401
- day.csv: a6bcf826782d3c0fbfdcbeead17cd0884185a0dafe8ff10cd48a874ee7ba18be
- Readme.txt: b92c8628622948bf0828c43a1b315d1517c3d7fd63654730d0dea379b8e88175

Transformations are reproducible in the notebook: strictly shifted demand and weather
features, calendar cycles, fixed target/feature scaling, and chronological partitioning.
No modification is made to the bundled original CSV.
