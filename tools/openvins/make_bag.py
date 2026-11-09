"""Convert one Mid-Air flight to a ROS 1 bag and an OpenVINS configuration (Phase 11, D-741).

Usage (in the separate environment cache/venvs/rosbags, which has the `rosbags` package,
Apache-2.0; not a dependency of src/):
    python tools/openvins/make_bag.py <flight> <out dir>

Writes <out dir>/<flight>/flight.bag (/cam0/image_raw mono8 at 25 Hz, /imu0 at 100 Hz) and
<out dir>/<flight>/config/ (OpenVINS v2.7 EuRoC configuration with Mid-Air's values).

Differences from what OpenVINS expects, and how they are handled (all stated with the result):
- Mid-Air's gyroscope is in the world frame (D-741); OpenVINS needs body rates, so the bag
  carries R_wb^T (gyro) with R_wb the ground-truth attitude. This gives OpenVINS a gyroscope
  rotated with the true attitude (its noise and bias then turn with the vehicle, which its
  constant-bias model does not expect); our filter models the world-frame gyroscope directly.
- Mid-Air flights start moving, so OpenVINS's dynamic initialisation is switched on.
- Calibration is exact in a simulator, so online calibration is off (as in our filter).
- IMU noise: the flight's white noise estimated from the signal (`navigation.imu`, the same as
  our filter) as a density sigma sqrt(dt); bias random walk at the generator's bound / sqrt(dt).
Everything else is OpenVINS's EuRoC configuration.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from aerialrange.data.midair import MidAirTrajectory, midair_camera  # noqa: E402
from aerialrange.geometry.frames import quat_to_R  # noqa: E402
from aerialrange.navigation.imu import ImuNoise  # noqa: E402
from midair_common import midair_root  # noqa: E402

ESTIMATOR = """%YAML:1.0
verbosity: "WARNING"
use_fej: true
integration: "rk4"
use_stereo: false
max_cameras: 1
calib_cam_extrinsics: false
calib_cam_intrinsics: false
calib_cam_timeoffset: false
calib_imu_intrinsics: false
calib_imu_g_sensitivity: false
max_clones: 11
max_slam: 50
max_slam_in_update: 25
max_msckf_in_update: 40
dt_slam_delay: 1
gravity_mag: 9.81
feat_rep_msckf: "GLOBAL_3D"
feat_rep_slam: "ANCHORED_MSCKF_INVERSE_DEPTH"
feat_rep_aruco: "ANCHORED_MSCKF_INVERSE_DEPTH"
try_zupt: false
zupt_chi2_multipler: 0
zupt_max_velocity: 0.1
zupt_noise_multiplier: 10
zupt_max_disparity: 0.5
zupt_only_at_beginning: false
init_window_time: 2.0
init_imu_thresh: 1.5
init_max_disparity: 10.0
init_max_features: 50
init_dyn_use: true
init_dyn_mle_opt_calib: false
init_dyn_mle_max_iter: 50
init_dyn_mle_max_time: 0.05
init_dyn_mle_max_threads: 6
init_dyn_num_pose: 6
init_dyn_min_deg: 10.0
init_dyn_inflation_ori: 10
init_dyn_inflation_vel: 100
init_dyn_inflation_bg: 10
init_dyn_inflation_ba: 100
init_dyn_min_rec_cond: 1e-12
init_dyn_bias_g: [ 0.0, 0.0, 0.0 ]
init_dyn_bias_a: [ 0.0, 0.0, 0.0 ]
record_timing_information: false
record_timing_filepath: "/tmp/traj_timing.txt"
save_total_state: false
filepath_est: "/tmp/ov_estimate.txt"
filepath_std: "/tmp/ov_estimate_std.txt"
filepath_gt: "/tmp/ov_groundtruth.txt"
use_klt: true
num_pts: 200
fast_threshold: 20
grid_x: 5
grid_y: 5
min_px_dist: 10
knn_ratio: 0.70
track_frequency: 25.0
downsample_cameras: false
num_opencv_threads: 2
histogram_method: "HISTOGRAM"
use_aruco: false
num_aruco: 1024
downsize_aruco: true
up_msckf_sigma_px: 1
up_msckf_chi2_multipler: 1
up_slam_sigma_px: 1
up_slam_chi2_multipler: 1
up_aruco_sigma_px: 1
up_aruco_chi2_multipler: 1
use_mask: false
relative_config_imu: "kalibr_imu_chain.yaml"
relative_config_imucam: "kalibr_imucam_chain.yaml"
"""

IMU = """%YAML:1.0
imu0:
  T_i_b:
    - [1.0, 0.0, 0.0, 0.0]
    - [0.0, 1.0, 0.0, 0.0]
    - [0.0, 0.0, 1.0, 0.0]
    - [0.0, 0.0, 0.0, 1.0]
  accelerometer_noise_density: {acc_nd:.6e}
  accelerometer_random_walk: {acc_rw:.6e}
  gyroscope_noise_density: {gyr_nd:.6e}
  gyroscope_random_walk: {gyr_rw:.6e}
  rostopic: /imu0
  time_offset: 0.0
  update_rate: 100.0
  model: "kalibr"
  Tw: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
  R_IMUtoGYRO: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
  Ta: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
  R_IMUtoACC: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
  Tg: [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
"""

IMUCAM = """%YAML:1.0
cam0:
  T_imu_cam:
{T}
  cam_overlaps: []
  camera_model: pinhole
  distortion_coeffs: [0.0, 0.0, 0.0, 0.0]
  distortion_model: radtan
  intrinsics: [{fx}, {fy}, {cx}, {cy}]
  resolution: [{w}, {h}]
  rostopic: /cam0/image_raw
"""


def main() -> int:
    from rosbags.rosbag1 import Writer
    from rosbags.typesys import Stores, get_typestore

    name, out = sys.argv[1], Path(sys.argv[2])
    root = midair_root()
    dest = out / name
    (dest / "config").mkdir(parents=True, exist_ok=True)
    traj = MidAirTrajectory(root, name)
    with h5py.File(root / "sensor_records.hdf5", "r") as f:
        g = f[name]
        gyr_w = g["imu/gyroscope"][()]
        acc = g["imu/accelerometer"][()]
        R = np.array([quat_to_R(q, order="wxyz") for q in g["groundtruth/attitude"][()]])
    gyr_b = np.einsum("nji,nj->ni", R, gyr_w)  # R_wb^T w_world (see the module docstring)
    noise = ImuNoise.from_signal(gyr_w, acc)
    dt = 0.01
    (dest / "config" / "estimator_config.yaml").write_text(ESTIMATOR)
    (dest / "config" / "kalibr_imu_chain.yaml").write_text(
        IMU.format(
            acc_nd=float(noise.acc.max() * np.sqrt(dt)),
            acc_rw=float(noise.acc_bias_rw.max() / np.sqrt(dt)),
            gyr_nd=float(noise.gyr.max() * np.sqrt(dt)),
            gyr_rw=float(noise.gyr_bias_rw.max() / np.sqrt(dt)),
        )
    )
    first = cv2.imread(str(traj.frame_path("color_left", 0)), cv2.IMREAD_GRAYSCALE)
    h, w = first.shape
    cam = midair_camera(w, h)
    T = np.eye(4)
    T[:3, :3] = cam.R_bc
    T[:3, 3] = cam.t_bc
    rows = "\n".join(f"    - [{', '.join(f'{x:.9f}' for x in r)}]" for r in T)
    (dest / "config" / "kalibr_imucam_chain.yaml").write_text(
        IMUCAM.format(T=rows, fx=cam.fx, fy=cam.fy, cx=cam.cx, cy=cam.cy, w=w, h=h)
    )
    ts = get_typestore(Stores.ROS1_NOETIC)
    Header = ts.types["std_msgs/msg/Header"]
    Time = ts.types["builtin_interfaces/msg/Time"]
    Imu = ts.types["sensor_msgs/msg/Imu"]
    Image = ts.types["sensor_msgs/msg/Image"]
    Vec = ts.types["geometry_msgs/msg/Vector3"]
    Quat = ts.types["geometry_msgs/msg/Quaternion"]

    def stamp(t: float) -> Time:
        sec = int(t)
        return Time(sec=sec, nanosec=int(round((t - sec) * 1e9)))

    bag = dest / "flight.bag"
    bag.unlink(missing_ok=True)
    n_frames = min(traj.n_frames, len(acc) // 4)
    events = [(r * dt, "imu", r) for r in range(len(acc))]
    events += [(k / 25.0, "cam", k) for k in range(n_frames)]
    events.sort(key=lambda e: (e[0], e[1] == "cam"))
    with Writer(bag) as wr:
        ci = wr.add_connection("/imu0", Imu.__msgtype__, typestore=ts)
        cc = wr.add_connection("/cam0/image_raw", Image.__msgtype__, typestore=ts)
        z9 = np.zeros(9)
        for seq, (t, kind, i) in enumerate(events):
            hdr = Header(seq=seq, stamp=stamp(t + 1.0), frame_id=kind)  # +1 s: no zero stamps
            tn = int(round((t + 1.0) * 1e9))
            if kind == "imu":
                m = Imu(
                    header=hdr,
                    orientation=Quat(x=0.0, y=0.0, z=0.0, w=1.0),
                    orientation_covariance=z9,
                    angular_velocity=Vec(x=gyr_b[i, 0], y=gyr_b[i, 1], z=gyr_b[i, 2]),
                    angular_velocity_covariance=z9,
                    linear_acceleration=Vec(x=acc[i, 0], y=acc[i, 1], z=acc[i, 2]),
                    linear_acceleration_covariance=z9,
                )
                wr.write(ci, tn, ts.serialize_ros1(m, Imu.__msgtype__))
            else:
                img = cv2.imread(str(traj.frame_path("color_left", i)), cv2.IMREAD_GRAYSCALE)
                m = Image(
                    header=hdr,
                    height=h,
                    width=w,
                    encoding="mono8",
                    is_bigendian=0,
                    step=w,
                    data=np.ascontiguousarray(img).ravel(),
                )
                wr.write(cc, tn, ts.serialize_ros1(m, Image.__msgtype__))
    traj.close()
    print(f"wrote {bag} ({bag.stat().st_size / 2**30:.2f} GB), {n_frames} frames")
    return 0


if __name__ == "__main__":
    sys.exit(main())
