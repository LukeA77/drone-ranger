# Tracked objects of one camera frame (Drone Ranger, Phase 12).
std_msgs/Header header           # stamp of the image; frame_id of the ENU world frame
uint32 frame_index               # 1-based count of frames processed
TrackedObject[] objects
