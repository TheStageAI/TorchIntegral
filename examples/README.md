### [Checkpoints of trained integral models are available at link][checkpoints_link].

## Edsr
To train integral model run:
```bash
python edsr.py --integral --batch-size 16
```
To resample (prune) and evaluate the integral model run command below:
```bash
python edsr.py --integral --resample --checkpoint=<INTEGRAL MODEL CHECKPOINT> --evaluate --batch-size 16
```

Use following command to perform pruning + grid-tuning:
```bash
python edsr.py --integral --resample --grid-tuning --batch-size 16
```
Specify super resolution scale factor with argument --scale (default=4).

## Imagenet
Train integral model:
```bash
python imagenet.py --integral <IMAGENET FOLDER PATH>
```

Validation trained integral model with original size:
```bash
 python imagenet.py --integral --evaluate --checkpoint <INTEGRAL MODEL CHECKPOINT> <IMAGENET FOLDER PATH> 
```

Validation of pruned model:
```bash
 python imagenet.py --integral --resample --evaluate --checkpoint <INTEGRAL MODEL CHECKPOINT> <IMAGENET FOLDER PATH> 
```

Add --data-parallel to use DataParallel.
```bash
 python imagenet.py --integral --resample --evaluate --data-parallel --checkpoint <INTEGRAL MODEL CHECKPOINT> <IMAGENET FOLDER PATH> 
```

[checkpoints_link]: https://
