# naco-project
Particle Swarm Optimization for Atelectasis Detection in X-ray Images.

- Sample dataset (5% of the full dataset): [Link to sample dataset on Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/sample)
- Full dataset (not uploaded because its 46GB...): [Link to full dataset on Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/data)

**--- DATASET A ---**

Only 'atelectasis' (and other diseases) - 'no findings' on PA view.

- [Link to only-PA (raw) dataset on Kaggle.](https://www.kaggle.com/datasets/lisanneweidmann/only-pa)
- [Link to only-PA (PSO 5 iterations applied) dataset on Kaggle.](https://www.kaggle.com/datasets/lisanneweidmann/only-pa-pso5)
- [Link to only-PA (PSO 20 iterations applied) dataset on Kaggle.](https://www.kaggle.com/datasets/lisanneweidmann/only-pa-pso20)

**--- DATASET B ---**

As close as possible 50-50 division on both 'atelectasis' (and other diseases) - 'no findings' and PA-AP view. It's not possible to achieve a perfect 50-50 division, but with count_occurrences.py you can see that it's pretty close.

- [Link to PA-AP_atelectasis-normal (raw) dataset on Kaggle](https://www.kaggle.com/datasets/lisanneweidmann/pa-ap-atelectasis-normal)
- [Link to PA-AP_atelectasis-normal (PSO 5 iterations applied) dataset on Kaggle](https://www.kaggle.com/datasets/lisanneweidmann/pa-ap-atelectasis-normal-pso5)

**--- DATASET C ---**

Only images taken from standard position PA, and the images have only the label 'No Findings' or 'Atelectasis' (so no other diseases known to be associated with the patient!). 

- [Link to only-pa-and-atelectasis (raw) dataset on Kaggle.](https://www.kaggle.com/datasets/lisanneweidmann/only-pa-and-atelectasis)
- [Link to only-pa-and-atelectasis (PSO 5 iterations applied) dataset on Kaggle.](https://www.kaggle.com/datasets/lisanneweidmann/only-pa-atelectasis-pso5)
