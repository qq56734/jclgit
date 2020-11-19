import torch
import torchvision
from modeling.deeplab import *
import os
import PIL.Image
import numpy as np
from mymodel import SKv3, SKv2
import utils


base_dir = os.path.abspath('..')

#是否用gpu测试
gpu = False

#epoch
start_epoch = 10

model_types = ['deeplabdrn', 'deeplab', 'deeplabx', 'deeplabxce', 'deeplabmobile', 'SKv3', 'SKv2']
model_type = model_types[0]

#分割类型 edge：板材边缘 seg: 板材表面
out_types = ['edge', 'seg']
out_type = out_types[1]

num_class = 1

if gpu:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

print(device)

#读取模型和权重

if 'SKv3' in model_type:
    model = SKv3(nums_block_list = [3, 6, 23, 3]).to(device)

elif model_type == 'SKv2':
    model = SKv2(nums_block_list = [3, 6, 12, 3]).to(device)

elif 'drn' in model_type:
    model = DeepLab(num_classes=num_class,
                        backbone='drn',
                        output_stride=8,
                        sync_bn=False,
                        freeze_bn=False, ).to(device)

elif 'xce' in model_type:

    model = DeepLab(num_classes=num_class,
                        backbone='xception',
                        output_stride=8,
                        sync_bn=False,
                        freeze_bn=False).to(device)

elif 'deeplabx' in model_type:

    model = network.deeplabv3plus_resnet152(num_classes=1, output_stride=8).to(device)

    utils.set_bn_momentum(model.backbone, momentum=0.01)

elif 'mobile' in model_type:

    model = DeepLab(num_classes=num_class,
                        backbone='mobilenet',
                        output_stride=8,
                        sync_bn=False,
                        freeze_bn=False).to(device)
else:
    model = network.deeplabv3plus_resnet101(num_classes=1, output_stride=8).to(device)

    utils.set_bn_momentum(model.backbone, momentum=0.01)


if gpu:
    model.load_state_dict(torch.load(os.path.join(base_dir,'trained', out_type, model_type, \
                                              '{0}_model.pth'.format(start_epoch))))
else:
    model.load_state_dict(torch.load(os.path.join(base_dir,'trained', out_type, model_type, \
                                              '{0}_model.pth'.format(start_epoch)), map_location='cpu'))

model.eval()

import cv2
import torchvision
from torchvision import transforms
import numpy as np

#读取图片并resize到合适的大小和转为tensor
img_dir = r'C:\Users\fscut\Desktop\3fd470da-4b31-46a8-a300-e41a5a6cff99.jpg'

#缩小比例
rate = 2

img = cv2.imread(img_dir)

print(img.shape)

img_h = int(img.shape[0]/rate)
img_w = int(img.shape[1]/rate)

trans = transforms.Compose([
    transforms.Resize([img_h, img_w]),
    transforms.ToTensor(),
    transforms.Normalize([0.52190286, 0.5069523, 0.48598358], [0.22775313, 0.22191888, 0.22364168])
])
#[0.4487, 0.4316, 0.4066], [0.2481, 0.2413, 0.2359]
#[0.52190286, 0.5069523, 0.48598358], [0.22775313, 0.22191888, 0.22364168]



img = PIL.Image.open(img_dir).convert('RGB')
image_tensor = trans(img)
image_tensor.unsqueeze_(0)
image_tensor = image_tensor.to(device)

#预测结果并保存
with torch.no_grad():
    if model_type == 'SKv3':
        pred = model(image_tensor)[-1]
    else:
        pred = model(image_tensor)
    pred = torch.sigmoid(pred)
    pred = pred.data.cpu().numpy()

pred= np.where(pred < 0.5,0, pred)
pred = np.where(pred > 0.5, 255, pred)
pred = pred.squeeze() # img(681, 1023)

cv2.imwrite('result.png', pred)




