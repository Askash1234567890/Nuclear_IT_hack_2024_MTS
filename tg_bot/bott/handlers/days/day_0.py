from datetime import datetime
import torch
import os
from torchvision import transforms
from PIL import Image
from aiogram import types
from aiogram.dispatcher import FSMContext
from ultralytics import YOLO
import numpy as np
import cv2
from aiogram.types import KeyboardButton

from loader import dp, bot

class net(torch.nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.batch_norm0 = torch.nn.BatchNorm1d(512)

        self.fc_1= torch.nn.Linear(512, 128)
        self.act_1=torch.nn.ELU()
        self.batch_norm1 = torch.nn.BatchNorm1d(128)

        self.fc_2 =torch.nn.Linear(128, 32)
        self.act_2=torch.nn.ELU()
        self.batch_norm2 = torch.nn.BatchNorm1d(32)

        # self.fc_3 =torch.nn.Linear(32, 8)
        # self.act_3=torch.nn.ELU()
        # self.batch_norm3 = torch.nn.BatchNorm1d(16)

        self.fc_4 = torch.nn.Linear(32, 2)

        self.sm = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.batch_norm0(x)
        x=self.fc_1(x)
        x=self.act_1(x)
        x = self.batch_norm1(x)

        x=self.fc_2(x)
        x=self.act_2(x)
        x = self.batch_norm2(x)

        # x=self.fc_3(x)
        # x=self.act_3(x)
        # x = self.batch_norm3(x)

        x = self.fc_4(x)
        return x

    def inference(self, x):
        x = self.forward(x)
        x = self.sm(x)
        return x


def get_tensor(photo_path):

    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ColorJitter(0.6, 0.6, 0.3, 0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    photo = Image.open('./photo/shov')
    photo = transform(photo)

    return photo

def image_preparation(picture):
    grayscale_image = cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)
    gaussian = cv2.GaussianBlur(grayscale_image, (3, 3), 121)
    image_equalize = cv2.equalizeHist(gaussian)
    return image_equalize


def plot_bboxes(results):
    img = results[0].orig_img
    names = results[0].names
    scores = results[0].boxes.conf.numpy()
    classes = results[0].boxes.cls.numpy()
    boxes = results[0].boxes.xyxy.numpy().astype(np.int32)
    for score, cls, bbox in zip(scores, classes, boxes):
        class_label = names[cls]
        label = f"{class_label} : {score:0.2f}"
        lbl_margin = 10
        img = cv2.rectangle(img, (bbox[0], bbox[1]),
                            (bbox[2], bbox[3]),
                            color=(0, 0, 255),
                            thickness=3)
        label_size = cv2.getTextSize(label,
                                     fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                     fontScale=5, thickness=3)
        lbl_w, lbl_h = label_size[0]
        lbl_w += 2* lbl_margin
        lbl_h += 2*lbl_margin
        cv2.putText(img, label, (bbox[0]+ lbl_margin, bbox[1]-lbl_margin),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=2.0, color=(0, 0, 255),
                    thickness=3)
    return img

@dp.message_handler(commands=['start'])
async def hi(message: types.Message):
    markup = types.ReplyKeyboardRemove()
    await message.answer('Приветствую! Меня зовут МИФИст, я чат-бот предназначенный для поиска дефектов сварных швов'
                         '\n\nОтправь мне фотографию сварного шва, чтобы я мог определить, есть ли у тебя дефект', reply_markup=markup)

@dp.message_handler(content_types='photo')
async def for_name(message: types.Message, state: FSMContext):
    markup = types.ReplyKeyboardRemove()
    await message.photo[-1].download(destination_file='./photo/shov')
    model1 = torch.jit.load('./models/model_scripted.pt')
    model2 = torch.jit.load('./models/my_googlenet.pt')
    model1.eval()
    model2.eval()
    photo = get_tensor('shov')
    y_pred1 = model1.forward(photo.unsqueeze(0)).argmax(dim=1)
    y_pred2 = model2.forward(photo.unsqueeze(0)).logits.argmax(dim=1)
    y_pred = int(((y_pred1 + y_pred2) / 2 >= 0.5)[0])

    if y_pred:
        await message.answer('Есть дефект', reply_markup=markup)
        photo = image_preparation('./photo/shov.jpg')
        model = torch.jit.load('./models/timur.pt')
        model.eval()
        results = model.forward(photo.unsqueeze(0))
        img = plot_bboxes(results)
        #cv2_imshow(img)      #Если запускать с обычной IDE, то писать cv2.imshow(<image>)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        await bot.send_photo(
            chat_id=message.chat.id,
            photo=img, reply_markup=markup
        )
    else:
        await message.answer('Нет дефекта', reply_markup=markup)




