import numpy as np
import h5py
import rospy
import ros_numpy
from sensor_msgs.msg import PointCloud2
import torch

# altered ros_numpy.point_cloud2 function
# originally found at http://docs.ros.org/en/kinetic/api/ros_numpy/html/point__cloud2_8py_source.html
# changed get_xyz_points to retrieve reflectivity
def get_reflectivity(cloud_array, remove_nans=True, dtype=np.float):
    '''Pulls out reflectivity columns from the cloud recordarray, and returns
    a 1xN matrix of same length N as x,y,z array.
    '''
    # remove crap points
    if remove_nans:
        mask = np.isfinite(cloud_array['x']) & np.isfinite(cloud_array['y']) & np.isfinite(cloud_array['z'])
        cloud_array = cloud_array[mask]
    
    # pull out x, y, and z values
    points = np.zeros(cloud_array.shape + (3,), dtype=dtype)
    points[...,0] = cloud_array['reflectivity']

    return points


def callback(data) :
    # returns a 3xN matrix for x,y,z coordinates
    xyz_array = ros_numpy.point_cloud2.get_xyz_points(data)
    # returns a 1xN matrix for reflectivity
    #reflectivity_1 = get_reflectivity(data)

    # NEED TO CONFIRM AXIS DIRECTION ON THIS
    #distance_1 = np.sqrt(np.sum(xyz_array**2, axis=1))


    ##PYTORCH BLOCK
    #convert to PyTorch tensor
    #distance = torch.as_tensor(distance_1.astype(np.float32, copy=False)).contiguous()
    #reflectivity = torch.as_tensor(reflectivity_1.astype(np.float32, copy=False)).contiguous()

    #to dataloader and model
    ##END PYTORCH BLOCK

    #simple manipulation
    # double all point coordinates
    xyz_array[...,0:3] = xyz_array[...,0:3]*2

    #convert back to pointcloud2 and return
    output = data.copy()
    output['x'] = xyz_array[...,0]
    output['y'] = xyz_array[...,1]
    output['z'] = xyz_array[...,2]

    rospy.Publisher('PCD_points' , PointCloud2, output)




def main() :

    #init PCD model

    #init ros node pub/sub and spin up
    rospy.init_node("PC_Denoiser")
    rospy.Subscriber("/lidar/parent/points_raw", PointCloud2, callback)
    rospy.spin()

if __name__ == '__main__' :
    main()

