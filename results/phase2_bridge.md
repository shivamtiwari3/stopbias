# Phase 2 bridge validation, offline against known ground truth
180 recoveries: 9 carrier legs x 5 true latencies x 2 detectors x 2 viewpoints. The real bridge code is in the path -- real packetisation, real loss, real concealment, real transcode.

`err_ms` is signed error against what each viewpoint should return: `L` at the model tap, `L + D_bridge` at the SIP tap.

## detector = `energy`

| leg | model err (ms) | sip err (ms) | D_bridge measured | concealed | late | reord |
|---|--:|--:|--:|--:|--:|--:|
| clean | +15.0 (+15 to +20) | +20.0 (+15 to +20) | 60.0 of 60 | 0.0% | 0 | 0 |
| pstn_band | +15.0 (+15 to +20) | +20.0 (+15 to +20) | 60.0 of 60 | 0.0% | 0 | 0 |
| noise_20db | +15.0 (+15 to +20) | +20.0 (+15 to +20) | 60.0 of 60 | 0.0% | 0 | 0 |
| loss_1pct | +15.0 (+15 to +20) | +20.0 (+15 to +20) | 60.0 of 60 | 1.2% | 0 | 0 |
| loss_5pct_bursty | +15.0 (+15 to +20) | +20.0 (+15 to +20) | 60.0 of 60 | 5.4% | 0 | 0 |
| jitter_15ms | +15.0 (+15 to +20) | +20.0 (+15 to +20) | 60.0 of 60 | 0.0% | 0 | 0 |
| jitter_80ms | +15.0 (+15 to +20) | +20.0 (+15 to +20) | 60.0 of 60 | 42.1% | 142 | 116 |
| drift_200ppm | +14.4 (+14 to +19) | +19.4 (+14 to +19) | 60.0 of 60 | 0.0% | 0 | 0 |
| pstn_full | +14.8 (+15 to +20) | +19.8 (+15 to +20) | 60.0 of 60 | 3.9% | 0 | 0 |

## detector = `silero`

| leg | model err (ms) | sip err (ms) | D_bridge measured | concealed | late | reord |
|---|--:|--:|--:|--:|--:|--:|
| clean | +8.0 (+4 to +28) | +16.0 (+4 to +20) | 72.0 of 60 | 0.0% | 0 | 0 |
| pstn_band | +8.0 (+4 to +28) | +16.0 (+4 to +20) | 72.0 of 60 | 0.0% | 0 | 0 |
| noise_20db | +8.0 (+4 to +28) | +16.0 (+4 to +20) | 72.0 of 60 | 0.0% | 0 | 0 |
| loss_1pct | +8.0 (+4 to +28) | +16.0 (+4 to +20) | 72.0 of 60 | 1.2% | 0 | 0 |
| loss_5pct_bursty | +8.0 (+4 to +28) | +16.0 (+4 to +20) | 72.0 of 60 | 5.4% | 0 | 0 |
| jitter_15ms | +8.0 (+4 to +28) | +16.0 (+4 to +20) | 72.0 of 60 | 0.0% | 0 | 0 |
| jitter_80ms | +8.0 (+4 to +28) | +16.0 (+4 to +20) | 72.0 of 60 | 42.1% | 142 | 116 |
| drift_200ppm | +7.4 (+3 to +27) | +15.4 (+3 to +19) | 72.0 of 60 | 0.0% | 0 | 0 |
| pstn_full | +7.8 (+4 to +28) | +15.7 (+4 to +20) | 72.0 of 60 | 3.9% | 0 | 0 |

Non-`ok` recoveries: 0 of 180.
