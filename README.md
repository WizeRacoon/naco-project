# naco-project
Particle Swarm Optimization for Atelectasis Detection in X-ray Images

data -> sample dataset (5% of the full dataset), available via [Link to sample dataset on Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/sample)

archive -> full dataset (not uploaded because its 46GB...), available via [Link to full dataset on Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/data)

**--- DATASET A ---**

only_PA -> only atelectasis-normal on PA view. This is because the PA view data seems to be better (by eyeballing the data).

- [Link to only-pa (raw) dataset on Kaggle.](https://www.kaggle.com/datasets/lisanneweidmann/only-pa)
- [Link to only-PA (PSO 5 iterations applied) dataset on Kaggle]( https://www.kaggle.com/datasets/lisanneweidmann/only-pa-pso5)

```
$ python3 count_occurrences.py 
directory_path='./only_PA/train/NORMAL'
Directory: ./only_PA/train/NORMAL
  PA: 3493
  AP: 0
  Atelectasis: 0
  No Finding: 3493
directory_path='./only_PA/train/ATELECTASIS'
Directory: ./only_PA/train/ATELECTASIS
  PA: 3493
  AP: 0
  Atelectasis: 3493
  No Finding: 0
directory_path='./only_PA/val/NORMAL'
Directory: ./only_PA/val/NORMAL
  PA: 874
  AP: 0
  Atelectasis: 0
  No Finding: 874
directory_path='./only_PA/val/ATELECTASIS'
Directory: ./only_PA/val/ATELECTASIS
  PA: 874
  AP: 0
  Atelectasis: 874
  No Finding: 0
directory_path='./only_PA/test_1/NORMAL'
Directory: ./only_PA/test_1/NORMAL
  PA: 453
  AP: 0
  Atelectasis: 0
  No Finding: 453
directory_path='./only_PA/test_1/ATELECTASIS'
Directory: ./only_PA/test_1/ATELECTASIS
  PA: 453
  AP: 0
  Atelectasis: 453
  No Finding: 0
directory_path='./only_PA/test_2/NORMAL'
Directory: ./only_PA/test_2/NORMAL
  PA: 453
  AP: 0
  Atelectasis: 0
  No Finding: 453
directory_path='./only_PA/test_2/ATELECTASIS'
Directory: ./only_PA/test_2/ATELECTASIS
  PA: 453
  AP: 0
  Atelectasis: 453
  No Finding: 0
directory_path='./only_PA/test_3/NORMAL'
Directory: ./only_PA/test_3/NORMAL
  PA: 453
  AP: 0
  Atelectasis: 0
  No Finding: 453
directory_path='./only_PA/test_3/ATELECTASIS'
Directory: ./only_PA/test_3/ATELECTASIS
  PA: 453
  AP: 0
  Atelectasis: 453
  No Finding: 0
```

**--- DATASET B ---**

PA-AP_atelectasis-normal -> as close as possible 50-50 division on both atelectasis-normal and PA-AP view. It's not possible to achieve a perfect 50-50 division, but with count_occurrences.py you can see that it's pretty close.

- [Link to PA-AP_atelectasis-normal (raw) dataset on Kaggle](https://www.kaggle.com/datasets/lisanneweidmann/pa-ap-atelectasis-normal)
- [Link to PA-AP_atelectasis-normal (PSO 5 iterations applied) dataset on Kaggle](https://www.kaggle.com/datasets/lisanneweidmann/pa-ap-atelectasis-normal-pso5)
```
$ python3 count_occurrences.py 
directory_path='./PA-AP_atelectasis-normal/train/NORMAL'
Directory: ./PA-AP_atelectasis-normal/train/NORMAL
  PA: 1746
  AP: 1746
  Atelectasis: 0
  No Finding: 3492
directory_path='./PA-AP_atelectasis-normal/train/ATELECTASIS'
Directory: ./PA-AP_atelectasis-normal/train/ATELECTASIS
  PA: 1746
  AP: 1746
  Atelectasis: 3492
  No Finding: 0
directory_path='./PA-AP_atelectasis-normal/val/NORMAL'
Directory: ./PA-AP_atelectasis-normal/val/NORMAL
  PA: 437
  AP: 437
  Atelectasis: 0
  No Finding: 874
directory_path='./PA-AP_atelectasis-normal/val/ATELECTASIS'
Directory: ./PA-AP_atelectasis-normal/val/ATELECTASIS
  PA: 437
  AP: 437
  Atelectasis: 874
  No Finding: 0
directory_path='./PA-AP_atelectasis-normal/test_1/NORMAL'
Directory: ./PA-AP_atelectasis-normal/test_1/NORMAL
  PA: 211
  AP: 245
  Atelectasis: 0
  No Finding: 456
directory_path='./PA-AP_atelectasis-normal/test_1/ATELECTASIS'
Directory: ./PA-AP_atelectasis-normal/test_1/ATELECTASIS
  PA: 217
  AP: 233
  Atelectasis: 450
  No Finding: 0
directory_path='./PA-AP_atelectasis-normal/test_2/NORMAL'
Directory: ./PA-AP_atelectasis-normal/test_2/NORMAL
  PA: 232
  AP: 223
  Atelectasis: 0
  No Finding: 455
directory_path='./PA-AP_atelectasis-normal/test_2/ATELECTASIS'
Directory: ./PA-AP_atelectasis-normal/test_2/ATELECTASIS
  PA: 230
  AP: 221
  Atelectasis: 451
  No Finding: 0
directory_path='./PA-AP_atelectasis-normal/test_3/NORMAL'
Directory: ./PA-AP_atelectasis-normal/test_3/NORMAL
  PA: 237
  AP: 212
  Atelectasis: 0
  No Finding: 449
directory_path='./PA-AP_atelectasis-normal/test_3/ATELECTASIS'
Directory: ./PA-AP_atelectasis-normal/test_3/ATELECTASIS
  PA: 231
  AP: 226
  Atelectasis: 457
  No Finding: 0
```

**--- DATASET C ---**

only_PA-and-ATELECTASIS -> Only images taken from standard position PA, and the images have only the label 'No Findings' or 'Atelectasis' (so no other diseases known to be associated with the patient!). 

[Link to only-pa-and-atelectasis dataset on Kaggle.](https://www.kaggle.com/datasets/lisanneweidmann/only-pa-and-atelectasis)
```
$ python3 count_occurrences.py 
directory_path='./only_PA-and-A/train/NORMAL'
Directory: ./only_PA-and-A/train/NORMAL
  PA: 1511
  AP: 0
  Atelectasis: 0
  No Finding: 1511
directory_path='./only_PA-and-A/train/ATELECTASIS'
Directory: ./only_PA-and-A/train/ATELECTASIS
  PA: 1511
  AP: 0
  Atelectasis: 1511
  No Finding: 0
directory_path='./only_PA-and-A/val/NORMAL'
Directory: ./only_PA-and-A/val/NORMAL
  PA: 378
  AP: 0
  Atelectasis: 0
  No Finding: 378
directory_path='./only_PA-and-A/val/ATELECTASIS'
Directory: ./only_PA-and-A/val/ATELECTASIS
  PA: 378
  AP: 0
  Atelectasis: 378
  No Finding: 0
directory_path='./only_PA-and-A/test_1/NORMAL'
Directory: ./only_PA-and-A/test_1/NORMAL
  PA: 107
  AP: 0
  Atelectasis: 0
  No Finding: 107
directory_path='./only_PA-and-A/test_1/ATELECTASIS'
Directory: ./only_PA-and-A/test_1/ATELECTASIS
  PA: 107
  AP: 0
  Atelectasis: 107
  No Finding: 0
directory_path='./only_PA-and-A/test_2/NORMAL'
Directory: ./only_PA-and-A/test_2/NORMAL
  PA: 107
  AP: 0
  Atelectasis: 0
  No Finding: 107
directory_path='./only_PA-and-A/test_2/ATELECTASIS'
Directory: ./only_PA-and-A/test_2/ATELECTASIS
  PA: 107
  AP: 0
  Atelectasis: 107
  No Finding: 0
directory_path='./only_PA-and-A/test_3/NORMAL'
Directory: ./only_PA-and-A/test_3/NORMAL
  PA: 107
  AP: 0
  Atelectasis: 0
  No Finding: 107
directory_path='./only_PA-and-A/test_3/ATELECTASIS'
Directory: ./only_PA-and-A/test_3/ATELECTASIS
  PA: 107
  AP: 0
  Atelectasis: 107
  No Finding: 0
```


