from cmath import nan
import numpy as np
import h5py
import rospy
import ros_numpy
from sensor_msgs.msg import PointCloud2
import torch

from weathernet.datasets import DENSE
from weathernet.model import WeatherNet

def denoise(model, distance_1, reflectivity_1) :
    #convert to PyTorch tensor
    distance = torch.as_tensor(distance_1.astype(np.float32, copy=False)).contiguous()
    reflectivity = torch.as_tensor(reflectivity_1.astype(np.float32, copy=False)).contiguous()

    #Get predictions
    pred = model(distance.cuda(), reflectivity.cuda())
    #print(pred.shape)

    #TODO: remove the unlabeled prediction label

    #choose argmax prediction for each point
    pred = torch.argmax(pred, dim=1, keepdim=True)

    distance_out = torch.squeeze(distance).cpu().numpy()
    reflectivity_out = torch.squeeze(distance).cpu().numpy()
    labels = torch.squeeze(pred).cpu().numpy()

    label_dict= {0:0, 1:100, 2:101, 3:102}
    labels = np.vectorize(label_dict.get)(labels)

    return labels

def callback(data) :

    global model

    pub = rospy.Publisher('PCD_points' , PointCloud2, queue_size=1)
    pc = ros_numpy.point_cloud2.pointcloud2_to_array(data)
    
    points=np.zeros((pc.shape[0], pc.shape[1] ,3))
    
    points[:,:,0]=pc['x']
    points[:,:,1]=pc['y']
    points[:,:,2]=pc['z']
    reflect=np.zeros((pc.shape[0], pc.shape[1]))
    reflect=pc['reflectivity']

    # NEED TO CONFIRM AXIS DIRECTION ON THIS
    distance_1 = np.sqrt(np.sum(points[:,:,:3]**2, axis=1))


    ##PYTORCH BLOCK
    #call model and get the labeled predictions
    labels = denoise(model, distance_1, reflect)
    
    #index invalid points
    labelsind = labels[not labels == 1]
    #remove invalid points
    points[labelsind, 0] = nan
    points[labelsind, 1] = nan
    points[labelsind, 2] = nan
    ##END PYTORCH BLOCK

    #simple manipulation
    # double all point coordinates
    #points[:,:,:3] = points[:,:,:3] / 2

    #convert back to pointcloud2 and return
    pcc = np.copy(pc)
    pcc['x'] = points[:,:,0]
    pcc['y'] = points[:,:,1]
    pcc['z'] = points[:,:,2]

    output = ros_numpy.point_cloud2.array_to_pointcloud2(pcc, frame_id=data.header.frame_id)

    pub.publish(output)





def main() :

    #init PCD model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    num_classes = DENSE.num_classes()
    model = WeatherNet(num_classes)
    model = model.to(device)

    #Init and Load in model
    model.load_state_dict(torch.load('checkpoints/model_epoch7_mIoU=75.5.pth'))
    model.eval()

    #init ros node pub/sub and spin up
    rospy.init_node("PC_Denoiser")
    rospy.Subscriber("/lidar/parent/points_raw", PointCloud2, callback, queue_size=1)
    rospy.spin()

if __name__ == '__main__' :
    main()

