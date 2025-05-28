# GTX code

## Dataset

* Download dataset by running the load_data notebook and move to the `/data` folder

## TensorFlow version

```
cd code_tf
bash train_tf.sh
```

## Pytorch version

```
cd code_py
bash train_py.sh
```

## Sagemaker

`cd code_{tf/py}` and run the `sagemaker_launcher_{tf/py}.py` file.

# MATLAB without onedrive

1. First verify `gtxDataPath`.
2. Create the `ImageData` folder and inside the folder create `TS` and `SL` two subfolders. 
3. Inside the `TS` subfolder create another folder `{name}` and place the `*.stl` image files inside.
4. Inside the `SL` folder, create another folder `SFDIOptPropLookupTables` and place the lookup tables for optical properties inside.
5. Modify the path `TS.subfolderTS` inside the code and put `{name}` that correspond to the folder name in step 3. 
6. Run `gtxSight` at the beginning before running the code. (Should do this everytime MATLAB relauches or path changes.)
7. Run the code.