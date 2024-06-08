import torch
import cv2
import numpy as np

from image_tokenizer import Image_Tokenizer
from utils import normalize_image, unnormalize_image


# Load image
image_path = "/root/share/cat/train/image/00850486-IMG_20211020_091815.jpg"
image = cv2.imread(image_path)

image = cv2.resize(image, (288, 288))
cv2.imwrite("outputs/origin.jpg", image)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = normalize_image(image)
image = np.transpose(image, (2, 0, 1))

data = np.array(image).astype(np.float32)
data = torch.from_numpy(data).unsqueeze(0)


# Load model
model_path = "weights/0604.pth"
device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
model = Image_Tokenizer()
model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(model_path).items()})
model = model.to(device)


# Generate
model.eval()
data = data.to(device)
with torch.no_grad(): 
    output = model.generate(data)

# Post process
image_save = output[0].permute(1, 2, 0).cpu().detach().numpy()
image_save = unnormalize_image(image_save).astype('uint8')
image_save = cv2.cvtColor(image_save, cv2.COLOR_RGB2BGR)
cv2.imwrite("outputs/test.jpg", image_save)