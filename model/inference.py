import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import time
from torch.utils.data import Dataset, DataLoader
import glob 
import os
import cv2
from PIL import Image
import random
from tqdm import tqdm
import time
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import csv
import shutil
import albumentations as A
from albumentations.pytorch import ToTensorV2
import time


class CustomOneDataset(Dataset):
    def __init__(self, img_dir,transform=None):
        self.path_img = glob.glob(img_dir)
        self.transform = transform
    def __len__(self):
        return len(self.path_img)

    def __getitem__(self, idx):

        image = cv2.imread(self.path_img[idx], 1)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        try:
            if self.transform:
                image = self.transform(image=image)["image"]
        except:
            print(self.path_img[idx])
        return image, self.path_img[idx]

def get_one_data_loader(input_path,batch_size , transform):
    data_transforms = transform
        # A.Compose([
        #     A.Resize(224, 224, interpolation=1, p=1),
        #     A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1),
        #     ToTensorV2()
        # ])

    image_datasets = CustomOneDataset(input_path , data_transforms)
    print('size test:',len(image_datasets))
    dataloaders = torch.utils.data.DataLoader(image_datasets,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    num_workers=4)                          
    return dataloaders,len(image_datasets)


class testModel():
    def __init__(self,
        model,
        name_model=None,
        size_image = (336,224),
        name_label = ['left','normal','reverse','right'],
        gpu = "0"
    ) -> None:

        if name_model == None:
            name_model = model.__class__.__name__

        self.name_label = name_label
        if gpu == None:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
        
        self.name_model = name_model
        self.model = model.to(self.device)
        self.transform = A.Compose(
        [
            A.Resize(size_image[0], size_image[1], interpolation=1, p=1),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1),
            ToTensorV2()
        ])


    def report_results(self,parameters,filename='result_model_3012.csv'):
        fields = list(parameters.keys())
        # data rows of csv file
        row = [list(parameters.values())]
        if not os.path.exists(filename):
            with open(filename, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames = fields)
                writer.writeheader()

        with open(filename, 'a') as f:
            write = csv.writer(f)
            write.writerows(row)
            f.close()

    def check_car(self,output, score_car = 0.85, score_car_closeup = 0.85,
                    score_closeup = 0.8, score_car_recaptured = 0.9, score_recaptured = 0.9,score_unknown = 0.8):
        # if output[1] > 0.999:
        #     return 1
        # if output[2] > 0.999:
        #     return 2
        # if output[3] > 0.999:
        #     return 3
        # if output[0] > 0.999:
        #     return 0      

        # if (output[1] > score_car and output[3] > score_car_closeup) and (output[1] < output[3]):
        #     return 3
        # if(output[1] < score_car and output[3] > score_closeup):
        #     return 3

        # if (output[1] > score_car and output[2] > score_car_recaptured) and (output[1] < output[2]):
        #     return 2
        # if (output[1] < score_car and output[2] > score_recaptured):
        #     return 2

        # if output[1] > score_car:
        #     return 1
        
        # if output[0] > score_unknown:
        #     return 0      
        return torch.argmax(output)
    def test_one_class(self,path_test,path_checkpoint,label = None,
                        batch_size=32,
                        print_output = False,
                        plot = False,
                        path_save_csv = None,
                        number_display = None,
                        path_save_file = None
                        ):
 
        Dataloader,size = get_one_data_loader(path_test,batch_size,self.transform)
        if number_display == None:
            number_display = size
        with torch.no_grad():
            self.model.load_state_dict(torch.load(path_checkpoint,map_location=lambda storage, loc: storage)['model_state_dict'])
            self.model.eval()
            count_predict_false = 0
            for idx,(data,path_img) in enumerate(Dataloader):
                
                data   = data.to(self.device)
                output = self.model(data)
                output_sigmoid = torch.sigmoid(output)
                # output_sigmoid = torch.softmax(output,dim=1)
                for idx_step,predict_one_step in enumerate(output_sigmoid):
                    check = self.check_car(predict_one_step)
                    probability = output_sigmoid[idx_step].cpu().detach().numpy()
                    # probability_uknown = probability[0]
                    # probability_car = probability[1]
                    # probability_recaptured = probability[2]
                    # probability_close_up = probability[3]
                    if check != label:
                        count_predict_false +=1

                        if print_output:
                            print("left: {:.2f} | normal: {:.2f} | reverse: {:.2f} | right: {:.2f}".format(
                                *probability))
                        if plot:
                            print("predict: ",self.name_label[check])
                            # plt.imshow(plt.imread(path_img[idx_step]))
                            plt.imshow(cv2.imread(path_img[idx_step]))
                            plt.show()
                        if path_save_file != None:
                            path_target = path_save_file + "left{:.2f}_normal_{:.2f}_reverse_{:.2f}_right_{:.2f}.".format(
                                *probability) + path_img[idx_step].split('.')[-1]

                            shutil.copy(path_img,path_target)
                        if path_save_csv != None:
                            parameters = {
                            "model": self.name_model,
                            "name img": path_img[idx_step].split('/')[-1],
                            "predict": self.name_label[check],
                            "left": round(probability[0],3),
                            "normal": round(probability[1],3),
                            "reverse": round(probability[2],3),
                            "right": round(probability[3],3),
                            }
                            self.report_results(parameters,filename=path_save_csv)
                                        
                        if count_predict_false > number_display:
                            return      
            del Dataloader
            #print("count predict false: ", count_predict_false)           

if __name__ == "__main__":
    
    model = models.efficientnet_b1(pretrained=True)
    model.classifier[1] = nn.Linear(in_features=1280, out_features=4)
    name_model = 'efficientnet_b1'
    test_model = testModel(model=model, 
                    name_model=name_model,
                    size_image=(224,224),
                    gpu = None)
    
    #test
    path_test = '/home/tinhnv/Cty/classifier_flip/Data_test/test_set3/*.jpg'
    path_checkpoint = '/home/tinhnv/Cty/classifier_flip/output15022023/checkpoint15022023/resnext_best.pt'
    batch_size = 128
    lst = os.listdir("/home/tinhnv/Cty/classifier_flip/Data_test/test_set3/") # your directory path
    number_files = len(lst)
    print("num files:",number_files)
    start = time.time()
    test_model.test_one_class(                                                                                                                          
                        path_test=path_test,
                        path_checkpoint=path_checkpoint,
                        batch_size=batch_size,
                        # label = None,
                        plot = True,                                                                                                                                                                
                        print_output = True,
                        # path_save_csv = "results_model/result_VBI_resnext_0901.csv"
                        number_display = 30,
                        )
    infer_time = (time.time() - start)/number_files
    print("Inference time: ", infer_time)