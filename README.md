# CAFDIM
PyTorch implementation

1. pip install  torch torchvision torchaudio Cython healpy opencv-python pydicom scikit-image numpy
2. Download the helix2fan source code for updating the torch-radon library：
    git clone https://github.com/faebstn96/helix2fan.git
3. Download torch-radon：
    git clone https://github.com/matteo-ronchetti/torch-radon.git
4. Enter the torch-radon folder：
    git apply path/to/helix2fan/torch-radon_fix/torch-radon_fix.patch
    Install torch-radon：
    python setup.py install

simulation:
1. Simulated Dataset Processing 1-sparse-view.py
2. Simulated Dataset Processing 2-training data.py
3. Train and Validation: train.py
4. Test: test.py

real:

1. reformat the raw helical scan data into fan-beam geometry:
    refer to:https://github.com/faebstn96/helix2fan
2. Real dataset processing 0.py, Real dataset processing 1.py,Real dataset processing 2.py,Real dataset processing 3.py
3. Train and Validation: train.py
4. Test: test.py

Pre-trained weights: It provides training weights for 60 views and one real 288 views.

\datasets\Pre_weights\
Can be loaded and used directly