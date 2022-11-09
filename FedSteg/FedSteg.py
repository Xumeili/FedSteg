###############################################################################
# 重要: 请务必把任务(jobs)中需要保存的文件存放在 results 文件夹内
# Important : Please make sure your files are saved to the 'results' folder
# in your jobs
###############################################################################
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models

import matplotlib.pyplot as plt
from PIL import Image
torch.manual_seed(65537)





# 模块，将像素值规范化作为模型的一部分，这样图像的有效范围是清晰的[0.1]
def get_model(name, device="cuda"):
    '''
        VGG, Res50, InceptionV3
    '''
    class Normalize(nn.Module):
        def __init__(self, mean, std) :
            super(Normalize, self).__init__()
            self.register_buffer('mean', torch.Tensor(mean))
            self.register_buffer('std', torch.Tensor(std))

        def forward(self, input):
            # Broadcasting
            mean = self.mean.reshape(1, 3, 1, 1)
            std = self.std.reshape(1, 3, 1, 1)
            return (input - mean) / std

    norm_layer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    model = None
    if name=="VGG":
        model = models.vgg19(pretrained=True)
    elif name=="Res50":
        model = models.resnet50(pretrained=True)
    elif name=="InceptionV3":
        model = models.Inception3(pretrained=True)
    elif name=="alexnet":
        model = models.alexnet(pretrained=True)    
    else:
        print("不支持当前模型")
        return None

    whole_model = nn.Sequential(norm_layer, model).to(device)
    whole_model.eval() #设置为测试模式
    return whole_model
# 模块：计算隐写损失函数值
# 更换损失函数在这里
def cal_Steganography_loss(outputs, secret):
    '''
        使用最后一层输出，正负均有。
    '''
    loss = None
    if secret[0]==0:
        loss = (outputs[0][0]-(0))*(outputs[0][0]-(0))
    else:
        loss = (outputs[0][0]-1)*(outputs[0][0]-1)
    for i in range(1,len(secret)):
        if secret[i]==0:
            loss += (outputs[0][i]-(0))*(outputs[0][i]-(0))
        else:
            loss += (outputs[0][i]-1)*(outputs[0][i]-1)

    return loss
# 模块: 读取图像
def load_img(img_path):

    I = Image.open(img_path).convert("RGB")
    I = I.resize((224,224))
    #plt.imshow(I)
    im_tensor = transforms.ToTensor()(I)
    im_tensor = torch.unsqueeze(im_tensor,0) #增加batch这一维度
    #print("图片尺寸:", im_tensor.shape)
    im_tensor.requires_grad=True
    return im_tensor
def encode_secret(im_tensor, secret:str, save_path):
    # 模块：嵌入


    # 实例化一个网络
    net = get_model("alexnet", device="cpu")
    # with torch.no_grad():
    outputs = net(im_tensor) # 输出1000维
    # print(outputs.shape)
    # print(outputs)
    
    loss = cal_Steganography_loss(outputs, secret)

    cnt = 0
    success = False # 隐写状态
    #while success==False and cnt<2000:
    while success==False and cnt<100:
        cnt += 1
        if cnt%100==0:
            print("迭代次数:", cnt)
            for i in range(12):
                print("当前值为", outputs[0][i])
        loss.backward()
        #print("梯度：",im_tensor.grad.data)
        im_tensor = im_tensor-0.1*torch.sign(im_tensor.grad.data)
        im_tensor = torch.clamp(im_tensor,0,1)
        im_tensor = im_tensor.detach() # 每次Backward后清空图像上的梯度，防止上次的影响
        im_tensor.requires_grad_()
        net.zero_grad()
        outputs = net(im_tensor)      
        #print("网络最后一层输出形状:", outputs.shape)
        loss = cal_Steganography_loss(outputs, secret)
        success = True
        for i in range(len(secret)):
            if secret[i]=='0' and outputs[0][i]>1:
                success = False
                break
            if secret[i]=='1' and outputs[0][i]<1:
                success = False
                break
    if success==True:
        print("嵌入成功")
        print(cnt)
    # 保存隐写后图片
    im_tensor_saved = torch.squeeze(im_tensor.detach())
#     print(im_tensor.shape)
#     plt.imshow(im_tensor.permute(1, 2, 0) )
    plt.imsave(save_path, im_tensor_saved.permute(1, 2, 0) )
    return im_tensor

# for i in range(8):
#     print(outputs[0][i])

def decode_secret(img_path, secret:str):
# 模块: 接收方提取隐写信息

    # 读取图像
    I = Image.open(img_path).convert("RGB")
    plt.imshow(I)
    im_tensor = transforms.ToTensor()(I)
    im_tensor = torch.unsqueeze(im_tensor,0)

    # 实例化一个网络
    net = get_model("alexnet", device="cpu")
    outputs = net(im_tensor)   
   
    success_decode = True
    cnt_wronga = 0
    cnt_wrongb = 0
    for i in range(len(secret)):
        print("提取值为", outputs[0][i])
        if secret[i]=='0' and outputs[0][i]>1:
            cnt_wronga += 1
            success_decode = False
        if secret[i]=='1' and outputs[0][i]<1:
            cnt_wrongb += 1
            success_decode = False
    if success_decode:
        print("解码成功，秘密值为", secret)
    else:
        print("解码失败，失败比特数为", cnt_wronga, cnt_wrongb)       
def run(ori_img_path, saved_img_path, secret): 
    im_tensor = load_img(ori_img_path) 
    encode_secret(im_tensor, secret, saved_img_path)
    decode_secret(saved_img_path, secret)
sc = "00000000"
run("ori_imgs/minst/8.jpg", "res_imgs/minst/8/15.jp", sc)