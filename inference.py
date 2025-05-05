import os
from PIL import Image
import cv2 as cv
from options.options import parse
import argparse
from archs.retinexformer import RetinexFormer
from torch.nn.parallel import DistributedDataParallel as DDP

parser = argparse.ArgumentParser(description="Script for prediction")
parser.add_argument('-p', '--config', type=str, default='./options/inference/LOLBlur.yml', help = 'Config file of prediction')
parser.add_argument('-i', '--inp_path', type=str, default='./images/inputs', 
                help="Folder path")
args = parser.parse_args()


path_options = args.config
opt = parse(path_options)
os.environ["CUDA_VISIBLE_DEVICES"]= "0"

# PyTorch library
import torch
import torch.optim
import torch.multiprocessing as mp
from tqdm import tqdm
from torchvision.transforms import Resize

from data.dataset_reader.datapipeline import *
from archs import *
from losses import *
from data import *
from utils.test_utils import *
from ptflops import get_model_complexity_info

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

#define some auxiliary functions
pil_to_tensor = transforms.ToTensor()
tensor_to_pil = transforms.ToPILImage()

def path_to_tensor(path):
    img = Image.open(path).convert('RGB')
    img = pil_to_tensor(img).unsqueeze(0)
    
    return img
def normalize_tensor(tensor):
    
    max_value = torch.max(tensor)
    min_value = torch.min(tensor)
    output = (tensor - min_value)/(max_value)
    return output

def save_tensor(tensor, path):
    
    tensor = tensor.squeeze(0)
    print(tensor.shape, tensor.dtype, torch.max(tensor), torch.min(tensor))
    img = tensor_to_pil(tensor)
    img.save(path)

def pad_tensor(tensor, multiple = 8):
    '''pad the tensor to be multiple of some number'''
    multiple = multiple
    _, _, H, W = tensor.shape
    pad_h = (multiple - H % multiple) % multiple
    pad_w = (multiple - W % multiple) % multiple
    tensor = F.pad(tensor, (0, pad_w, 0, pad_h), value = 0)
    
    return tensor

def load_model(model, path_weights):
    # load checkpoint
    ckpt = torch.load(path_weights, map_location="cpu")
    state_dict = ckpt['params']

    # check keys in model vs. in state_dict
    model_keys      = list(model.state_dict().keys())
    checkpoint_keys = list(state_dict.keys())

    # if checkpoint keys start with "module.", strip it
    if all(key.startswith("module.") for key in checkpoint_keys):
        new_sd = { key[len("module."):]: v
                   for key, v in state_dict.items() }
    # if checkpoint keys lack "module." but model expects it, add it
    elif all(not key.startswith("module.") for key in checkpoint_keys) \
         and all(mk.startswith("module.") for mk in model_keys):
        new_sd = { "module." + key: v
                   for key, v in state_dict.items() }
    else:
        # keys already match
        new_sd = state_dict

    # finally load
    model.load_state_dict(new_sd, strict=True)
    print("✅ Loaded weights")
    return model


#parameters for saving model
PATH_MODEL = opt['save']['path']
resize = opt['Resize']

def predict_folder():
    # create and load the model as before…
    model, _, _ = create_model(opt['network'], rank=0)
    model = load_model(model, opt['save']['path'])
    model = model.to(device)
    model.eval()

    os.makedirs('./images/results', exist_ok=True)
    images = [f for f in os.listdir(args.inp_path)
              if f.lower().endswith(('.png','jpg','jpeg'))]

    for fn in tqdm(images, desc="Inferring"):
        img_path = os.path.join(args.inp_path, fn)
        try:
            tensor = path_to_tensor(img_path).to(device)
            H, W = tensor.shape[-2:]

            # optional downsampling for large images
            if opt['Resize'] and max(H, W) >= 1500:
                tensor = Resize([H//2, W//2])(tensor)

            tensor = pad_tensor(tensor)

            with torch.no_grad():
                out = model(tensor, side_loss=False)

            out = torch.clamp(out, 0., 1.)[:, :, :H, :W]
            save_tensor(out, os.path.join('./images/results', fn))

        except RuntimeError as e:
            # Check if this is an OOM error
            if 'out of memory' in str(e):
                tqdm.write(f"⚠️ CUDA OOM on {fn}, skipping.")
                # free up whatever we can
                torch.cuda.empty_cache()
                continue
            else:
                # re‑raise other runtime errors
                raise

    print("Finished inference!")
    
def main():
    predict_folder()

if __name__ == '__main__':
    main()