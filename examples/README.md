# Training Integral neural networks.

## Edsr
```bash
python edsr.py --integral --batch-size 16
```
Specify super resolution scale factor with argument --scale (default=4).

## Imagenet
```bash
python imagenet.py --integral <IMAGENET FOLDER PATH>
```
Add --data-parallel to use DataParallel training.

# Evaluation of trained INNs.
To resample (prune) and evaluate the integral model run commands below:
## Edsr
```bash
python edsr.py --integral --resample --checkpoint=<INTEGRAL MODEL CHECKPOINT> --evaluate --batch-size 16
```

## Imagenet
```bash
python imagenet.py --integral --resample --evaluate --checkpoint <INTEGRAL MODEL CHECKPOINT> <IMAGENET FOLDER PATH> 
```


### [Checkpoints of trained integral models are available at link][checkpoints_link].

[checkpoints_link]: https://

# Fast pruning of DNNs.
Here we do not train integral neural networks.
Instead we convert pre-trained DNN to integral and tune integration partition with desired size only.

DNN -> INN -> Integration grid tuning.

```bash
python edsr.py --integral --resample --grid-tuning --batch-size 16
```
