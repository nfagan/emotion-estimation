1. original eig https://github.com/CNCLgithub/EIG-faces
use "vampnet" environment locally
  TODO: test with provided environment

`eig_gen_activations.py` is copied and modified from `infer_render_using_eig/infer.py`
`render_modified.m` is copied and modified from `infer_render_using_eig/render.m`

*** not used
2a. need dlib for face landmark detection 
http://dlib.net
`mamba create -n dlib_env python=3.9`
`mamba activate dlib_env`
`pip install dlib --verbose`
`python dlib_landmark_detect.py D:\data\changlab\ilker_collab\dlib\shape_predictor_81_face_landmarks.dat D:\data\changlab\ilker_collab\goldenberg_faces\images`
end not used ***

2b. need mtcnn for face landmark detection
https://github.com/ipazc/mtcnn
```
set PYTHONPATH=%PYTHONPATH%;C:\Users\nick\source\changlab\ilker_collab\deps\mtcnn
python mtcnn_gen_landmarks.py D:\data\changlab\ilker_collab\goldenberg_faces\images
```

3. updated "eig" https://github.com/sicxu/Deep3DFaceRecon_pytorch
use mamba to create environment (faster)
see `d3dfr_env.bat`
`python test.py --name=face_recon --epoch=20 --img_folder=D:\data\changlab\ilker_collab\goldenberg_faces\images`

4. sparse coding https://github.com/Chibee/Sparse_Coding

* `eig_gen_activations.py` - need to add to the python path (see `eig_env.bat`)