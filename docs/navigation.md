# Camera, IMU and GPS fusion (Phase 11)

Version 1 measured how much navigation errors hurt ranging by adding assumed errors to the simulator's exact camera poses (D-621). A drone does not have exact poses: it has an IMU, a GPS receiver and a camera. This phase builds the filter that turns those into poses with an honest uncertainty, checks the result on Mid-Air, and feeds the poses and their uncertainty into terrain-from-motion ranging. Decision record: D-741 in `docs/decisions.md`.

## The filter

A multi-state constraint Kalman filter (MSCKF; Mourikis and Roumeliotis, 2007), in `src/aerialrange/navigation/`:

- **IMU propagation** at 100 Hz, with the gyroscope and accelerometer biases estimated online.
- **Camera updates** from features tracked across frames (version 1's Shi-Tomasi corners and pyramidal Lucas-Kanade tracking, checked forward and backward, plus a RANSAC check). The filter keeps the last 11 camera poses. Each feature track is triangulated, and the part of its error that does not depend on where the feature is goes into one update.
- **GPS updates** of position and velocity at 1 Hz, with a model of the receiver's correlated errors.
- **Consistency by design:** first-estimate Jacobians (Li and Mourikis, 2013), and the filter's NEES (normalised estimation error squared) is checked against its expected range.

Settings follow OpenVINS's published configuration where Mid-Air gives no reason to differ. The feature noise was chosen on the development flights by consistency, with the rule written down first (D-741, item 7).

## What Mid-Air needed

Four properties of the data shaped the filter. Each was measured on the development flights, and each has a script and a saved result:

| Finding | Evidence | What the filter does |
|---|---|---|
| The gyroscope reports world-frame rates, although the file says body frame | Integrated as body rates, 1 s of noise-free motion gives 0.8 to 5.9° median error per flight; as world rates, under 0.001° (`imu_frames.json`) | The gyroscope's frame is a setting: body for real hardware, world for Mid-Air |
| IMU noise is different on every flight and not recorded | Estimated from the IMU signal alone, it lands within 0.98 to 1.05 of the true level (`imu_noise_check.json`) | Measures each flight's IMU noise from its own signal, as it flies |
| GPS errors are correlated: a constant offset per flight (about 12 m in height), a part that fades over about 3 s, and white noise | `gnss_fit.json`; scaling by dilution of precision made it worse | Offset and fading part are filter states; white noise is measurement noise |
| GPS velocity is exactly the true velocity | Largest difference 7 mm/s | Adds noise at a published receiver's accuracy (u-blox NEO-M8, 0.074 m/s per axis), seeded and labelled synthetic |

The flights also start in mid-air at 5 to 20 m/s, with no compass. The filter therefore starts with an alignment that fits attitude and both IMU biases together, using GPS velocity changes and the camera's direction of travel. Without the camera, yaw often cannot be found in steady flight: one flight started 61° off, and the filter never recovered.

## Results on the development flights

The table compares the full filter with the same filter without the camera, on 15 development flights (medians over flights). "Error over 0.04 s / 0.4 s" is the error in how the drone moved between two frames that far apart: the quantity terrain from motion depends on.

| | Camera, IMU and GPS | IMU and GPS only |
|---|---|---|
| Translation error over 0.04 s | 5.6 mm | 34 mm |
| Translation error over 0.4 s | 7.3 cm | 38 cm |
| Rotation error over 0.4 s | 0.21° | 0.36° |
| Tilt error | 0.37° | 0.82° |
| Yaw error (worst flight) | 0.36° (0.8°) | 4.5° (52°) |
| Horizontal position error (root mean square) | 0.84 m | 0.86 m |
| NEES, attitude / position / velocity (expected 3, 95 % band 1.9 to 4.4) | 2.8 / 3.0 / 3.3 | 7.7 / 3.0 / 2.6 |

Height is off by about 12 m in both columns. That error is the GPS's own offset, which nothing on board can observe; the filter's uncertainty includes it.

## Effect on ranging

The same terrain-from-motion samples as version 1 (590,130 land samples on the 15 development flights, while the filter runs), with five kinds of camera pose. Range error is the median |relative error| with a 95 % interval over flights. Coverage is how often the truth falls inside the method's own interval, with no calibration.

| Poses | Range error | Coverage at 68 / 90 / 95 % |
|---|---|---|
| Exact (simulator) | 2.5 % (2.1 to 3.0) | 60 / 78 / 83 % |
| **Filter, with its own pose uncertainty** | **3.8 % (3.1 to 4.7)** | **89 / 92 / 93 %** |
| Filter, without a pose uncertainty | 3.6 % | 50 / 68 / 74 % |
| Version 1's assumed "good" navigation | 6.3 % | 36 / 51 / 56 % |
| Version 1's assumed "typical" navigation | 12.0 % | 32 / 46 / 52 % |
| Version 1's assumed "poor" navigation | 36.9 % | 40 / 53 / 58 % |

- The filter's poses cost 1.3 points against exact ones, and do better than the best navigation grade version 1 assumed.
- Its pose uncertainty, carried into the range interval, brings coverage close to nominal at 90 and 95 % (the 68 % interval is too wide, because the errors are heavy-tailed).
- Ranging on a flat ground plane needs the camera's absolute height, and fails with real GPS heights (201 % error, because GPS height is about 12 m off). Terrain from motion needs only how the camera moved between frames, so it is unaffected.

## Against OpenVINS

OpenVINS (v2.7), a widely used open-source visual-inertial filter, was run as a reference on the same 15 flights, in its own container (it is GPL-licensed, so none of its code is in this package). It uses the camera and IMU only. Our filter was also run with GPS used only to start, so the two can be compared like for like. Medians over flights:

| | Translation error over 0.04 s | Tilt error | Start in flight failed |
|---|---|---|---|
| OpenVINS (camera and IMU) | 3.7 cm | 1.2° | 6 of 15 flights |
| Ours, GPS at the start only | 3.3 cm | 0.57° | none |
| Ours, camera, IMU and GPS | 0.56 cm | 0.38° | none |

- On the 9 flights where OpenVINS starts well, it is somewhat more accurate frame to frame than our filter without GPS (2.7 against 2.9 cm; better on 7 of 9), and ours has the smaller tilt error (0.52° against 0.76°).
- Without GPS our filter is overconfident, because its settings were chosen with GPS. It is not offered as a GPS-free navigator.
- The six-fold gain comes from fusing GPS tightly, and from a start that does not fail when the drone is already moving.

## Limits

- **Simulated sensors.** Mid-Air's IMU has no scale or axis-misalignment errors, and its camera is ideal. The GPS velocity noise was added by us.
- **One climate.** Kite, sunny; 15 development flights so far. The test flights run once, after the settings are frozen.
- **Laptop speed only.** Runtime has been measured on a laptop only. Raspberry Pi figures will be estimates until measured (build2 R12).
