@REM set PYTHONPATH=%PYTHONPATH%;C:\Users\nick\source\changlab\ilker_collab\deps\Deep3DFaceRecon_pytorch
@REM python ..\..\src\d3dfr_gen_activations.py --name=face_recon --epoch=20 --img_folder=D:\data\changlab\ilker_collab\goldenberg_faces\images
python ..\..\src\d3dfr_gen_activations.py --name=face_recon --epoch=20 --img_folder=D:\data\changlab\ilker_collab\flickr-faces\subset-emotion-balanced\train
