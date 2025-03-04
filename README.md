This is the source code of FashionFINE.

### Requirements:
pip install -r requirements.txt

### Prepare dataset:
We followed dataset preparation same as FashionSAP https://github.com/hssip/FashionSAP

#### FashionGen
Download the raw file and extract it from the path *data_root*.
Change the *data_root* and *split* in *prepare_dataset.py* and run it to get the assistance file.

To train BLIP, original FashionGen images should be located in data_root/images.

#### FashionIQ
Download the raw file and extract it in path data_root.
The directory *captions* and *images* in the raw file are put in *data_root*. Besides the file, we also merge all kinds of train files into the *cap.train.json* file in captions so as to *val*.

FashionIQ images should be located in data_root/tgir_images

### Prepare pre-training
cd checkpoint_pretrain
### Please download a model from https://drive.google.com/file/d/16kxbK7u86jVUfkwM7_4q2lJhCYeutgRv/view?usp=sharing

### Pre-training
cd ../
bash pt.sh

### Fine-tuning cross-modal retrieval
bash retrieval.sh

### Evaluate cross-modal retrieval using the fine-tuned FashionFINE checkpoint
### Please download a model from https://drive.google.com/file/d/1IRAs-UG8cwtogEWPYLFetyG8jJ-7mJuz/view?usp=sharing
bash eval_cmr.sh

### Fine-tuning text-guided image retrieval
bash tgir.sh

### Evaluate text-guided image retrieval using the fine-tuned FashionFINE checkpoint
### Please download a model from https://drive.google.com/file/d/1e5tF-QWM2RZa5W4My7SdiOJF2jl3sydN/view?usp=sharing
bash eval_TGIR.sh



### FashionFINE on BLIP
cd FashionFINE_BLIP

### Get pre-trained BLIP weight
wget https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base.pth

### Fine-tuning FashionFINE on BLIP
bash fashionfine_blip.sh

### Evaluate Retrieval using fine-tuned checkpoint. Please check the file named "retrieval_fashionfine_eval.yaml."
cd ../checkpoint
Please download a model from https://drive.google.com/file/d/1rjQXvixkCYwOgC2QcjrQMRIhFxLR0IA4/view?usp=sharing
cd ../FashionFINE_BLIP
bash eval_fashionfine_blip.sh
