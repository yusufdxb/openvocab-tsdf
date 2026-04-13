# One grounded 3D target produced by the openvocab-tsdf pipeline.
# Frame: the map frame name is carried in the parent message / service response header.

geometry_msgs/Point center
geometry_msgs/Point bbox_min
geometry_msgs/Point bbox_max
float32 score              # cosine similarity, [-1, 1]; higher is better
int32 voxel_count          # number of voxels supporting this target
