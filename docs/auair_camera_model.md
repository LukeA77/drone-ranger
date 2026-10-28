# AU-AIR camera model

Every AU-AIR result depends on the assumptions below. Values come from `results/auair/camera_scale.json` (`scripts/auair_scale.py`), fitted on the calibration recordings only (D-506) and applied unchanged to the held-out clips. Decisions D-503 to D-509 in `docs/decisions.md`.

## 1. Specification (prior)

Parrot Bebop 2: retail listings describe a 180° fisheye lens from which a Full HD (1920 × 1080) frame is cropped, with electronic (digital) stabilisation. No focal length is published (searched 2026-10-21; retail listings, for example B&H Photo product 1206694-REG). The prior therefore only says "cropped from a fisheye, digitally stabilised"; it gives no number.

## 2. Focal length (`{{FOCAL_PX_AUAIR}}`)

**f = 1,259 px (95 % interval 1,155 to 1,308 px, bootstrap over clips), horizontal field of view about 75°.**

Method (D-508): on straight-down calibration clips, the image centre moves by f × d / h when the drone moves d metres at true height h. The image motion is a similarity transform fitted to tracked ground corners between two frames 1 or 2 s apart, measured directly (not chained). d is the GPS displacement, kept only where it agrees within ±25 % with the displacement integrated from the recorded velocities (this removes GPS jumps while hovering). h is the altitude corrected for the altimeter offset (section 3). 52 windows from clips 5, 6, 7, 8 and 30.

Check: ground-truth car boxes give f × L (long side × corrected height); with f this implies a car length of 4.61 m (4.37 to 5.06). Axis-aligned boxes overstate the length of cars seen at an angle, so this leans long; the average length of cars produced in Europe was 4.36 m in 2022 (Inovev market analysis, accessed 2026-10-21; a production mean, not a fleet distribution). The check is consistent; it is not a second measurement of f.

## 3. Altimeter offset

True height = recorded altitude + h0. h0 is estimated from the image scale change during climbs (scale = h_i / h_j).
- Recording 20190829091111: h0 drifts linearly, −0.79 m at the start and −0.43 m per minute (climbs in clips 0, 5, 6 and 9; residual spread 1.8 m).
- Recording 20190905142119: one climbing clip only (30), so a constant h0 = +5.6 m, with a residual spread of 7.4 m. **Weak.**

The offset is not constant within a flight, so any range that uses the altimeter carries an error of a few metres at 10 to 30 m (10 to 30 %).

## 4. Distortion (`{{DIST_COEFFS_AUAIR}}`)

No visible curvature of straight lines in any clip (the manual labels, D-504), but the tilt test shows distortion matters for geometry fitted over the whole frame: restricting to the central 60 % moved the straight-down estimate of clip 0 from 85° to 88.5° (D-514). No coefficients are fitted; analyses that depend on precise geometry use the central region. A numerical test against image radius is still to do.

## 5. Orientation (`{{CAMERA_TILT_DEG_BY_CLIP}}`)

AU-AIR records body angles only, not the camera tilt (D-503).
- **Mount and model (D-511):** image up is the drone's forward, no mirror; **digitally stabilised** (the six best of 72 hypotheses on the calibration clips are all stabilised).
- **Per-frame tilt from motion (D-514):** the ground-plane homography between frames 1 to 2 s apart, decomposed, taking the solution whose rotation matches the recorded yaw change; fitted on the central 60 % of the image to limit lens distortion. Calibration checks: clip 0 88.5°, clip 7 84° (independently about 85° from car size against image row), mixed clip 31 23.5° (label 15-45°). Uncertainty about ±5°. Not valid where buildings or trees dominate the central view.

## 6. What remains uncertain

- The focal interval (about ±5 %) and the altimeter offset (spread 1.8 m on one flight, 7.4 m on the other) both scale every range.
- GPS and the recorded velocities both come from the drone's own estimator, so their agreement is not fully independent.
- One drone, one camera, one site; low altitude (3 to 31 m).
