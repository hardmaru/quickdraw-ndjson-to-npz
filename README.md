# Convert RAW Quickdraw datafiles into Stroke-5 format with RDP at variable epsilon

The notebook [convert_ndjson.ipynb](https://github.com/hardmaru/quickdraw-ndjson-to-npz/blob/master/convert_ndjson.ipynb) will convert `.ndjson` files into the `.npz` files one can use to train sketch-rnn. We set the target length to be 200 steps, and vary epsilon parameters to control the granuarity of the RDP algorithm.

To download an `.ndjson` file, try:

```
wget https://storage.googleapis.com/quickdraw_dataset/full/raw/jail.ndjson
```

To download the `jail` class

And we assume it is saved in a subdir called `/ndjson` for this notebook.
