# One tracked object in one frame (Drone Ranger, Phase 12). Positions are in the array's
# header frame (ENU world, REP 103); people are a class and a track number only.
uint32 track_id
string object_class              # "person" or "vehicle"
vision_msgs/BoundingBox2D bbox   # in the image, pixels

bool valid                       # false: no ground point (for example above the horizon)
string reason                    # why not valid ("ok" when valid)
string reliability               # "reliable", "uncertain" or "invalid"

geometry_msgs/Point position     # ground point, m (z: the flat ground's height)
float64[4] position_covariance   # 2 x 2 (east, north), row-major, m^2, raw (before calibration)

float64 range_m                  # slant range from the camera, m
float64 range_sigma_m
float64[3] range_half_width_m    # calibrated 68 / 90 / 95 % interval half-widths, NaN if none

bool velocity_valid              # true only where the display rule shows the speed
geometry_msgs/Vector3 velocity   # ground velocity (east, north, 0), m/s
float64[4] velocity_covariance   # 2 x 2 (east, north), row-major, (m/s)^2
