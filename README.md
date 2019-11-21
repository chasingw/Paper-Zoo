# Paper
- 深度学习等相关论文 笔记
- object detection 相关

## paper list

### 2D Object Detection

#### two-stage

- RCNN Fast-RCNN Faster-RCNN
  - RCNN-2013-Rich feature hierarchies for accurate object detection and semantic segmentation
    Tech report (v5)
    - [paper](https://arxiv.org/abs/1311.2524)
    - [Playing around with RCNN - karpathy](https://cs.stanford.edu/people/karpathy/rcnn/)
  - [Fast-RCNN-2015]
    - [paper](https://arxiv.org/abs/1504.08083)
    - [code-rbgirshick](https://github.com/rbgirshick/fast-rcnn)
  - [Faster-RCNN-2015]
    - [paper](https://arxiv.org/abs/1506.01497)
    - [code-rbgirshick](https://github.com/rbgirshick/py-faster-rcnn)
  - Mask R-CNN 2017
    - [paper](https://arxiv.org/abs/1703.06870)

#### one-stage

- YOLO
  - paper 
    - [pjreddie Publications](https://pjreddie.com/publications/)
  - [author pjreddie homepage](https://pjreddie.com/darknet/yolo/)
- SSD: Single Shot MultiBox Detector 2015
  - [paper](https://arxiv.org/abs/1512.02325)

FPN Feature Pyramid Networks for Object Detection 2016 

- [paper](https://arxiv.org/abs/1612.03144)

Focal Loss for Dense Object Detection 

- [paper](https://arxiv.org/abs/1708.02002)

### Computer Vision 

- FPN [Fully Convolutional Networks for Semantic Segmentation](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf) 2014

- ResNet Deep Residual Learning for Image Recognition 2015

  - [paper](https://arxiv.org/abs/1512.03385)

  

### 3D Object Detection 

[KITTI 3D Object Detection Evaluation](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)

#### Vote3Deep: Fast Object Detection in 3D Point Clouds Using Efficient 1609.06666

- [paper](https://arxiv.org/abs/1609.06666)

#### MV3D 2016 Multi-View 3D Object Detection Network for Autonomous Driving

- [paper](https://github.com/bostondiditeam/MV3D)
- [code unofficial](https://github.com/bostondiditeam/MV3D)

#### Pointnet pointnet++ f-pointnet

- pointnet 2017 PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation
  - [Homepage (paper code and so on)](http://stanford.edu/~rqi/pointnet/)
- pointnet2 2017 PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space
  - [Homepage](http://stanford.edu/~rqi/pointnet2/)
- F-Pointnet - Frustum PointNets for 3D Object Detection from RGB-D Data
  - [paper](https://arxiv.org/abs/1711.08488)
- [Author Homepage](http://stanford.edu/~rqi/)

#### VoxelNet 2017

- [paper](https://arxiv.org/abs/1711.06396)
- [code]

#### Complex-YOLO

- [paper]()
- [code unofficial](https://github.com/AI-liu/Complex-YOLO)

#### PointRCNN

- [PointRCNN](paper/PointRCNN: 3D Object Proposal Generation and Detection from Point Cloud-1812.04244.pdf)
- [code unofficial inplementation]()

#### RoarNet

- [official website](https://sites.google.com/berkeley.edu/roarnet)

- [paper](https://arxiv.org/abs/1811.03818)
- [code](https://github.com/Kiwoo/RoarNet)

#### SECOND: Sparsely EmbeddedConvolutional Detection

- [SECOND: Sparsely EmbeddedConvolutional Detection](paper/SECOND: Sparsely EmbeddedConvolutional Detection.pdf)



### 点云补全

- [PCN: Point Completion Network](paper/1808.00671-PCN: Point Completion Network.pdf)

### Deep Learning

#### Dropout 2014

- [paper](http://jmlr.org/papers/volume15/srivastava14a.old/srivastava14a.pdf)

#### SparseConvNet

- paper
  - [Submanifold Sparse Convolutional Networks-1706.01307](paper/Submanifold Sparse ConvolutionalNetworks-1706.01307.pdf)
  - [3D Semantic Segmentation withSubmanifold Sparse Convolutional Networks-1711.10275](paper/3D Semantic Segmentation withSubmanifold Sparse Convolutional Networks-1711.10275.pdf)
- [code](https://github.com/facebookresearch/SparseConvNet)

#### [可变形卷积 Deformable Convolutional Networks]()

- paper

### 路径规划

- [OpenPlanner Open Source Integrated Planner for Autonomous Navigation in Highly Dynamic Environments](paper/Open Source Integrated Planner for Autonomous Navigation in Highly Dynamic Environments.pdf)

#### 局部路径规划
- [Optimal_Trajectory_Generation_for_Dynamic_Street_Scenarios_in_a_Frenet_Frame.pdf](paper/Optimal_Trajectory_Generation_for_Dynamic_Street_Scenarios_in_a_Frenet_Frame.pdf) 
- [Baidu  Apollo  EM  Motion  Planner-1807.08048.pdf](paper/Baidu  Apollo  EM  Motion  Planner-1807.08048.pdf)
- [A Survey of Motion Planning and Control
Techniques for Self-driving Urban Vehicles](paper/1604.07446-A Survey of Motion Planning and ControlTechniques for Self-driving Urban Vehicles.pdf)

### book

- [http://www.probabilistic-robotics.org](http://www.probabilistic-robotics.org/) ProbabilisticRobotics 来自google x实验室创始人Sebastian Thrun的经典著作，详细介绍了基于概率的机器人感知，定位与规划控制方法
- [Multiple View Geometry in Computer Vision \(Second Edition)] 深入的介绍了机器视觉中的视角变换 

- the global positioning system and inertial navigation 深入浅出介绍了GPS工作原理，IMU工作原理和Kalman滤波，并提供了多种数据融合解决方案，能够让人从头读到尾的好书。
### SLAM

- [VIO 回顾：从滤波和优化的视角](https://mp.weixin.qq.com/s/zpZERtWPKljWNAiASBLJxA)

- Visual Odometry Visual_Odometry_VO_Part_I_Scaramuzza.pdf

### 优化

### 滤波