This is a little demo of deep-learning based odomentry write by myself.
The demo has tested on the Kitti Dasaset.

# Multi_view_Cylinder_Net

An end-to-end deep-learning-based lidar odometry.

In order to ensure the speed of odometry, cylinder projection of  lidar points is adopted. 

### Pipeline

![image-20210121232112102](C:\Users\phy12321\AppData\Roaming\Typora\typora-user-images\image-20210121232112102.png)

### NetWork

![image-20210121232134936](C:\Users\phy12321\AppData\Roaming\Typora\typora-user-images\image-20210121232134936.png)

##### Illustration of multi-view projection and segmentation

![image-20210121232308753](C:\Users\phy12321\AppData\Roaming\Typora\typora-user-images\image-20210121232308753.png)

## Run

#### Dependencies

```bash
python>3.6
pytorch>=1.5
cupy
tqdm
shutil
matplotlib
glob
argparse
```

#### Train

Firstly, change the config information in `args.py`.

Then run, 

`python train.py`

##### Test

Change the config information in `args.py` and run,

`python test.py`




## Results on Kitti

* 01

  ![image-20210121232500701](C:\Users\phy12321\AppData\Roaming\Typora\typora-user-images\image-20210121232500701.png)

* 03

  ![image-20210121232530428](C:\Users\phy12321\AppData\Roaming\Typora\typora-user-images\image-20210121232530428.png)

* 04

  ![image-20210121232546114](C:\Users\phy12321\AppData\Roaming\Typora\typora-user-images\image-20210121232546114.png)

* 07

  ![image-20210121232620545](C:\Users\phy12321\AppData\Roaming\Typora\typora-user-images\image-20210121232620545.png)

* 08:

  ![image-20210121232635286](C:\Users\phy12321\AppData\Roaming\Typora\typora-user-images\image-20210121232635286.png)

* 09:

  ![image-20210121232648109](C:\Users\phy12321\AppData\Roaming\Typora\typora-user-images\image-20210121232648109.png)

* 10:

  ![image-20210121232708299](C:\Users\phy12321\AppData\Roaming\Typora\typora-user-images\image-20210121232708299.png)

Note: sequence 00-08 for training, 09 and 10 for testing.

 