# Equine sutured-organ measurements: accession and importer

## Scope and provenance

Pascoletti et al., [dataset DOI 10.17632/zpf2jrtgsx.1](https://data.mendeley.com/datasets/zpf2jrtgsx/1), version 1, CC BY 4.0; the [accompanying data article](https://doi.org/10.1016/j.dib.2020.105644) describes eight equine laryngoplasty specimens, each subjected to 3000 displacement-controlled cycles between 30 and 50 N, then a 1 mm/s tensile ramp to rupture. The published data are sampled nominally at 100 Hz. Measurements are from a load cell and an optical encoder in a coupled suture–organ fixture, following cyclic preconditioning. The ramp measurements alone do not identify strand properties, tissue properties, contact friction, or human-skin suturing.

All eight ramp files were downloaded on ssh numi from the dataset's official public-file endpoints and checked against the SHA-256 hashes and byte lengths in its version-1 public file listing. The larger cyclic data were not downloaded. Only Test1Ramp has been parsed or inspected; Test2–8 are retained as unseen files, not claimed as a preregistered held-out test yet. The initial Python urllib fetch received HTTP 403; a subsequent curl request with a browser user-agent succeeded, followed by hash verification. This access behavior is not a physics result.

| File | Bytes | Published and verified SHA-256 |
| --- | ---: | --- |
| Test1Ramp.txt | 148080 | b3d735204570fb880339a7470abdb46eea58e0e2dc08dbd53d9301ee9e3cb115 |
| Test2Ramp.txt | 454822 | d22c547ac14e5feac2006a1055886f2824b0718c81085c80475d3c2d00786437 |
| Test3Ramp.txt | 134162 | 8fbdfc0fa48f2045ebcfad67caf84f9875d9435de4169fd9396b4af246ed4185 |
| Test4Ramp.txt | 258588 | 13692839dacc02dedb122086327cc19a678be68d7fbc009dae118b0d6788a752 |
| Test5Ramp.txt | 122856 | 2372dd878b84fb72304abab6551a3ac0b6aec1ddcbab705b998c7edda30261ac |
| Test6Ramp.txt | 181428 | d448d114731dd102dedcaf9d807483fa36d84b969395f091ae1f9e6b4adbd0e5 |
| Test7Ramp.txt | 277666 | 291747d84a0cfbb9235df08b00a8ec613200f8542ac69d59c1b9f6ea98cfc668 |
| Test8Ramp.txt | 127018 | c4ab61a72b5d24d1d9e880db1698f1b02209fb693ff8cf179def416348304565 |

## Executable conversion and checks

[The importer](measurements/import_equine_suture_ramp.py) verifies the raw file hash before parsing those same bytes, requires the published five-column UTF-16 instrument schema and finite values, rejects non-increasing total time and refuses to overwrite its output. It converts both position channels from mm to m and preserves times, force signs and sample order. It does not resample, baseline-correct, fit, infer failure, or assume uniform timestamps.

The three tests in `measurements/test_import_equine_suture_ramp.py` passed on ssh numi on 2026-11-08, covering exact unit/sign mapping, output identity and overwrite refusal, invalid hashes, invalid timestamps, nonfinite values and schema changes. A real Test1 conversion produced 1,474 rows from 0 to 14.809 s; its CSV SHA-256 was `007ed003d938a4d13d1acef2ed0f68a09f2f1d1cf13a6a24246917e02e418fd5`. The development data include negative force readings; no sign clipping is applied. These checks establish faithful data handling only.

Reproduce using the official file and its published hash:

```sh
python3 numi/measurements/import_equine_suture_ramp.py Test1Ramp.txt \
  --sha256 b3d735204570fb880339a7470abdb46eea58e0e2dc08dbd53d9301ee9e3cb115 \
  --output Test1Ramp.si.csv
python3 -m unittest discover -s numi/measurements -p 'test_import_equine_suture_ramp.py' -v
```

Before fitting or evaluating a model, define the geometry and preconditioned initial state, a physically interpretable fixture model, calibration and held-out specimen assignments, force/displacement and timing metrics, uncertainty, and failure criteria. The current measurements do not establish that the necessary geometry, constitutive inputs, or complete tissue-coupled simulation are available. Do not treat this accession as a calibrated material or a successful surgical validation.
