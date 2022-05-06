# About EMHapp
A pipeline for automatic detection, visualization and localization of epileptic magnetoencephalography high-frequency oscillations.

GUI was programed using App designer in MATALB.

EMHapp pipeline was  programed using python, and accelerated by GPU with functions of pyTorch. 

# Install EMHapp

1. Download EMHapp code via github.

2. Install MATLAB toolbox:

    Brainstorm3: https://neuroimage.usc.edu/brainstorm/Introduction.
  
    SPM12: https://www.fil.ion.ucl.ac.uk/spm/.
  
    FieldTrip: https://www.fieldtriptoolbox.org/.

3. Install python packages (Nvidia drive version is 10.2): 

   ```shell
   conda env create -f requirements.yml
   ```

# USE EMHapp

1. Open EMHapp GUI (run in MATLAB):

```matlab
EMHapp
```

2. Press "Load Button" to load project:

   1). Press "Add Button" and select anatomy file (MRI or FreeSurfer).

   3). Set NAS, LPA and RPA fiducial points for each subject.

   4). Press "Add Button" and select MEG file for each Subject.

   5). Press "OK Button" and check database.
   
3. Press "Process Button" to set parameters and run pipeline:

   1. 
