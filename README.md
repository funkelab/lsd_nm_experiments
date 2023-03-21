# LSD NM experiments tutorial

## General notes

* A tutorial for running training/inference of networks used for paper (in
  singularity containers).

* Tested on Ubuntu 18.04 with Quadro P6000 (24gb gpu ram)

* Assuming `singularity` is installed and setup 
(some tutorials [here](https://docs.sylabs.io/guides/3.5/user-guide/quick_start.html) and
[here](https://singularity-tutorial.github.io/01-installation/))

* Assuming conda is installed and setup (helpful [instructions](https://docs.conda.io/en/latest/miniconda.html) if needed)

---

## Getting started

* Clone this repo:

```
git clone https://github.com/funkelab/lsd_nm_experiments.git
```

* Create simple environment for fetching containers:

```
conda create -n test_env python=3.8
conda activate test_env
```

If you just need [boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html) for fetching containers:

```
pip install boto3
```

If you also want to view data with [neuroglancer](https://github.com/google/neuroglancer):

```
pip install boto3 neuroglancer h5py zarr
```

---

## Downloading container(s)

* `download_imgs.py` will download the singularity container used in the paper (`lsd:v0.8.img`)
* We found that sometimes this singularity container throws errors because of deprecated cuda versions that don't play well with updated drivers
* We made another image (`lsd_legacy.img`) that should handle this. If running into `libcublas` or `gcc` errors with the original container, consider using the newer one
* To download (uncomment legacy img if desired):

```
python download_imgs.py
```

---

## Or building legacy container from source:

* Nvidia has deprecated the conda packages of the cudnn / cudatoolkit versions used in the original singularity container (6.0/8.0)
* This sometimes causes problems with updated drivers (even though the point of containerization is to solve this...)
* Can get around by installing these from tars, specifically these:
  ```
  https://anaconda.org/numba/cudatoolkit/8.0/download/osx-64/cudatoolkit-8.0-3.tar.bz2
  https://repo.anaconda.com/pkgs/free/linux-64/cudnn-6.0.21-cuda8.0_0.tar.bz2
  ```
* `setup.sh` will fetch these packages and use them to install directly into the conda environment when creating the Singularity container
* The other packages are just specified in the `lsd_legacy.yml`
* To build legacy image locally:

```
./setup.sh
```

* Note - the original container runs fine with `exec`, the legacy one uses `run`. To be honest, not sure what the problem is here
* So `singularity exec --nv lsd:v0.8.img python -c "import tensorflow"` will work, and
* `singularity run --nv lsd_legacy.img python -c "import tensorflow"` will work, but
* `singularity exec --nv lsd_legacy.img python -c "import tensorflow"` will cause import errors

---

## Fetching data

* Navigate to 01_data directory (`cd 01_data`)
* `fetch_data.py` will download training data from the aws bucket using a json file (`datasets.json`) specifying the training volumes for each dataset
* The script defaults to just downloading the first volume for each dataset (one for each of zebrafinch, fib25, hemi, or three total)
* download data:

```
python fetch_data.py
```

* you could also use the singularity container to download the data, but we already have boto3 in the basic env we created anyway

--- 

## Creating masks

* `create_masks.py` will create a `labels_mask` that we use for training to constrain random locations
* If you installed zarr into the conda environment, you can just run with:

```
python create_masks.py
```

* otherwise, using the singularity container:

```
singularity exec --nv ../lsd:v0.8.img python create_masks.py
```

* or `singularity run...` if using legacy container

--- 

## Viewing the data

* If you installed neuroglancer into your environment, you can view the data with `view_data.py`
* e.g `python -i view_data.py -d funke/fib25/training/tstvol-520-1.zarr`
* If you are viewing remotely, you could also set the bind address with -b (defaults to localhost)
* There are some good little packages & tutorials for using neuroglancer differently. Examples 
[here](https://github.com/funkelab/funlib.show.neuroglancer), [here](https://connectomics.readthedocs.io/en/latest/external/neuroglancer.html)
and [here](https://github.com/google/neuroglancer/tree/master/python/examples)

Example fib25 training data:

![](https://github.com/funkelab/lsd_nm_experiments/blob/master/static/fib25_training_data.png)

---

## Downloading network checkpoints

* `fetch_checkpoint.py` will download a specified network checkpoint for a given dataset to the target folder in `02_train`
* We can start with the baseline affinities for the zebrafinch dataset just using conda boto3 (or use singularity if desired):

```
python fetch_checkpoint.py
```

--- 

## Training a network

* Navigate to the zebrafinch baseline directory (`cd ../02_train/zebrafinch/baseline`)
* We start by creating our network in `mknet.py` (e.g placeholders to match to the trained graphs):

```
singularity exec --nv ../../../lsd:v0.8.img python mknet.py
```

* It should print a bunch of layer names and tensor shapes (that looks like a sideways U-Net) to the command line
* Check the files in the directory (e.g `tree .`), it should now look like:

```
.
├── checkpoint
├── config.json
├── config.meta
├── mknet.py
├── predict.py
├── predict_scan.py
├── train_net_checkpoint_400000.data-00000-of-00001
├── train_net_checkpoint_400000.index
├── train_net_checkpoint_400000.meta
├── train_net.json
├── train_net.meta
├── train.py
└── view_batch.py
```

* Train for 1 iteration:

```
singularity exec --nv ../../../lsd:v0.8.img python train.py 1
```

* You'll see that gunpowder will print `ERROR:tensorflow:Couldn't match files for checkpoint ./train_net_checkpoint_500000`
* This is because it checks the `checkpoint` file which specifies iteration 500000 (since this network was trained for longer than the optimal checkpoint)

* If we view the batch, we'll see that the predictions are all grey, since it really only trained for a single iteration and didn't use the checkpoint (`python -i view_batch.py -f snapshots/batch_1.hdf`):

![](https://github.com/funkelab/lsd_nm_experiments/blob/master/static/baseline_zfinch_batch_1.png)

* To fix, simply edit this `checkpoint` file to point to the downloaded checkpoint iteration instead (e.g 500000 -> 400000)
* Now running the above won't do anything, because of line 27 in `train.py`:

```py
if trained_until >= max_iteration:
     return
```

* So just make sure to train to the checkpoint + n, eg for 1 extra iteration:

```
singularity exec --nv ../../../lsd:v0.8.img python train.py 400001
```

* We can then view the saved batch, e.g:

```
python -i view_batch.py -f snapshots/batch_400001.hdf
```

![](https://github.com/funkelab/lsd_nm_experiments/blob/master/static/baseline_zfinch_batch_400k.png)

---

## Running Inference

* For the lsds experiments, we ran everything from inference through evaluation in a blockwise fashion using [daisy](https://github.com/funkelab/daisy)
* For inference, this meant having a blockwise prediction script that called a gunpowder predict pipeline inside each process
* For example, this [script](https://github.com/funkelab/lsd/blob/master/lsd/tutorial/scripts/01_predict_blockwise.py) would distribute this [script](https://github.com/funkelab/lsd_nm_experiments/blob/master/02_train/zebrafinch/baseline/predict.py) by using a [DaisyRequestBlocks](http://funkey.science/gunpowder/api.html?highlight=daisy#gunpowder.DaisyRequestBlocks) gunpowder node
* If you just want to run inference on a small volume (in memory), you can instead use a [Scan](http://funkey.science/gunpowder/api.html?highlight=scan#gunpowder.Scan) node
* We added adapted all blockwise inference scripts to also use scan nodes (e.g `predict_scan.py`)
* Example run on zfinch training data:

```
singularity exec --nv ../../../lsd:v0.8.img python predict_scan.py
```

* Resulting affinities:

![](https://github.com/funkelab/lsd_nm_experiments/blob/master/static/baseline_zfinch_preds.png) 

---

## Multitask (MTLSD)

* This can be run exactly the same as the baseline above. 
* Inside `01_data/fetch_checkpoint.py`, uncomment `mtlsd` and run. 
* Navigate to `02_train/zebrafinch/mtlsd`
* Create network (`... python mknet.py`)
* Change `checkpoint` file iteration to match downloaded checkpoint (500000 -> 400000)
* Train (`... python train.py 400001`)
* View batch (`... python -i view_batch.py ...`) -> will also show LSDs
* Predict (`... python predict_scan.py`) -> will write out LSDs and Affs

* This will also give us LSDs:

![](https://github.com/funkelab/lsd_nm_experiments/blob/master/static/mtlsd_zfinch_preds_lsds.png)

* Tip - you can view the different components of the LSDs by adjusting the shader in neuroglancer, e.g changing `0,1,2` to `3,4,5` will show the diagonal entries of the covariance component (or the direction the processes move):

```
void main() {
    emitRGB(
        vec3(
            toNormalized(getDataValue(3)),
            toNormalized(getDataValue(4)),
            toNormalized(getDataValue(5)))
        );
}
```

* Or to view a single channel of the 10d lsds, (e.g channel 6):

```
void main() {
    float v = toNormalized(getDataValue(6));
    vec4 rgba = vec4(0,0,0,0);
    if (v != 0.0) {
        rgba = vec4(colormapJet(v), 1.0);
    }
    emitRGBA(rgba);
}
```

--- 

## Autocontext (ACLSD and ACRLSD)

* These networks rely on a pretrained LSD (raw -> LSDs) network, eg:
* ACLSD: `Raw -> LSDs -> Affs`
* ACRLSD: `Raw -> LSDs + cropped Raw -> Affs`
* Because of this, they are more computationally expensive (since training requires extra context to first predict the lsds)
* They ran using 23.5 GB of available 24 GB GPU RAM when testing on quadro p6000. If you have less than that you will likely run into cuda OOM errors
* If you have access to sufficient gpu memory, to start navigate to `01_data` and uncomment `lsd` + run script to get the pretrained lsd checkpoint
* Go to lsd directory (`cd ../02_train/zebrafinch/lsd`) and follow same instructions as baseline and mtlsd nets above
* Once you have the lsd checkpoints (and `test_prediction.zarr` with `pred_lsds` following prediction), start with a basic autocontext network (aclsd).
* Get the aclsd checkpoint, navigate to appropriate directory, change checkpoint file as before and train network.
* Visualizing the resulting batch shows us the larger raw context needed for predicting the lsds to ensure that the output affinities remain the same size:

![](https://github.com/funkelab/lsd_nm_experiments/blob/master/static/aclsd_training_batch.png)

* Run prediction as before - note, will not run if lsds have not been predicted first. This script could be adapted to predict the lsds on-the-fly using just the lsds and affs checkpoints
* The same can be done for the ACRLSD network (note, requires a merge provider during prediction as this network takes both lsds and raw data as input) 

---

## Todos: add consolidated fib25/hemi nets to tutorial
