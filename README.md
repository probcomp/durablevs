# DURableVS: Data-efficient Unsupervised Recalibrating Visual Servoing via online learning in a structured generative model

Nishad Gothoskar, Miguel Lázaro-Gredilla, Yasemin Bekiroglu, Abhishek Agarwal, Joshua B. Tenenbaum, Vikash K. Mansinghka, Dileep George

This is the official implementation of our ICRA 2022 paper "DURableVS: Data-efficient Unsupervised Recalibrating Visual Servoing via online learning in a structured generative model".

DurableVS is our a method for unsupervised learning of visual servoing which does not require any prior calibration and is extremely data-efficient. Our key insight is that visual servoing does not depend on identifying the veridical kinematic and camera parameters, but instead only on an accurate generative model of image feature observations from the joint positions of the robot. With our model architecture and learning algorithm, we can consistently learn accurate models from less than 50 training samples (which amounts to less than 1 min of unsupervised data collection) and then use these to enable accurate visual servoing. We can also continue learning online, which enables the robotic system to recover from calibration errors and to detect and quickly adapt to possibly unexpected changes in the robot-camera system (e.g. bumped camera, new objects).


## Installation

To install Python dependencies and the `durablevs` package:

```setup
pip install --upgrade pip
pip install -r requirements.txt
python setup.py develop
```

## Demo

The jupyter notebook in `notebooks/demo.ipynb` runs data collection, learning, and servoing given desired image feature locations for the 6DoF PyBullet robot.


## Citation

```
@article{gothoskar2022durablevs,
  title={DURableVS: Data-efficient Unsupervised Recalibrating Visual Servoing via online learning in a structured generative model},
  author={Gothoskar, Nishad and L{\'a}zaro-Gredilla, Miguel and Bekiroglu, Yasemin and Agarwal, Abhishek and Tenenbaum, Joshua B and Mansinghka, Vikash K and George, Dileep},
  journal={arXiv preprint arXiv:2202.03697},
  year={2022}
}
```

## Contact

If you have any questions, please email NishadG Gothoskar at nishad@mit.edu.
