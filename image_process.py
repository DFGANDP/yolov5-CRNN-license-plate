# -*- coding: utf-8 -*-
"""

UZYCIE:

Jedziesz autem, nagrywasz droge przed soba
Program szczytuje rejestracje
* Jesli jest wystarczajaco duza zeby odczytac rejestracje (optymalizacja)
Kazda szczytana przerzuca przez RNN
Dekoduje wyniki
Porownuje z baza danych rejestracji (np nieoznakowanych policynych) 
(o ile znakow treshold ustawic 'w terenie')
Jesli jest podobna o 1 znak albo 2 ALERT (czerwona)
Jesli wiecej np 3 lub 4 Pomaranczowa dioda
Jesli wiecej pomin


1. Najpierw zeby dzialalo na zdjeciach
2. potem filmy i live

@author: Wojtek

"""
import argparse
import torch
from torchvision.io import read_image
import torchvision.transforms as T
# import pandas as pd
import cv2
from crnn_data.model.model_crnn import PLUX
from sklearn import preprocessing
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# C:/Users/Wojtek/Desktop/wojtek/paper_implementation/torch/project/yolo_crnn
class Yolo_detector():
    def __init__(self):
        self.yolo_model = torch.hub.load('.',
                               'custom',
                               path='crnn_data/yolo_model.pt',
                               source='local')

    def forward(self, img_path):
        results = self.yolo_model(img_path)
        #print(type(results))
        a = results.pandas().xyxy[0]
        return a

#my_network = Yolo_detector()
#results = my_network.forward(im)

def crop_image(img_path, df):
    print(df)
    img = cv2.imread(img_path)
    df = df.reset_index()  # make sure indexes pair with number of rows
    for index, row in df.iterrows():
        xmin, ymin, xmax, ymax = (int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax']))
        cropped_image = img[ymin:ymax, xmin:xmax]
        cv2.imwrite(f'crnn_data/{index}.jpg', cropped_image)

class CRNN_detector():
    '''
    Zrobic z tego analize jak model zmienia shape w cnn --> rnn
    '''
    def __init__(self):
        self.model = PLUX(36).to(device) # num of letters in alphabet + num 0-9 + nun letter is added in model architecture
        self.model.load_state_dict(torch.load('crnn_data/model/CRNN_state_dict.pt', map_location=device))
        self.model.eval()
        print(self.model)
        self.encoder = self.get_encoder()

    def get_preds_pytorch(self, image_path):
        '''
        Przepisac z numpy i PIL na torch czyste

        Mozna przyspieszyc nie zapisujac do pliku
        tylko odrazu zrzucac z opencv do rozczytywania

        '''
        # image_channels, image_height, image_width
        img = read_image(image_path).type(torch.float).to(device)
        transform = T.Resize((75,300))
        img = transform(img)
        img /= 255
        img = torch.unsqueeze(img, 0)
        #print(type(image))
        image = img.to(device)
        preds, loss = self.model(image, None)
        return preds

    def get_preds_cv2(self, img):
        '''
        Przepisac z numpy i PIL na torch czyste

        Mozna przyspieszyc nie zapisujac do pliku
        tylko odrazu zrzucac z opencv do rozczytywania

        '''
        # image_channels, image_height, image_width
        img = torch.from_numpy(img).type(torch.float).to(device)
        transform = T.Resize((75,300))
        img = transform(img)
        img /= 255
        img = torch.unsqueeze(img, 0)
        #print(type(image))
        image = img.to(device)
        preds, loss = self.model(image, None)
        return preds

    def remove_duplicates(self, x):
        if len(x) < 2:
            return x
        fin = ""
        for j in x:
            if fin == "":
                fin = j
            else:
                if j == fin[-1]:
                    continue
                else:
                    fin = fin + j
        return fin

    def get_encoder(self):
        '''
        encoder przepisac na wlasny

        odpalic w chmurze zobaczyc co i jak enkoduje

        '''

        targets = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                   'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                   'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                   'U', 'V', 'W', 'X', 'Y', 'Z']
        lbl_enc = preprocessing.LabelEncoder()
        lbl_enc.fit(targets)
        return lbl_enc

    def decode_predictions(self, preds):
        preds = preds.permute(1, 0, 2)
        preds = torch.softmax(preds, 2)
        preds = torch.argmax(preds, 2)
        preds = preds.detach().cpu().numpy()
        cap_preds = []
        for j in range(preds.shape[0]):
            temp = []
            for k in preds[j, :]:
                k = k - 1
                if k == -1:
                    temp.append("§")
                else:
                    p = self.encoder.inverse_transform([k])[0]
                    temp.append(p)
            tp = "".join(temp).replace("§", "")
            cap_preds.append(self.remove_duplicates(tp))
        return cap_preds

#crop_image(im, results)

#crnn_network = CRNN_detector()
#preds = crnn_network.get_preds_pytorch("crnn_data/1.jpg")
#literki = crnn_network.decode_predictions(preds)

def levenshteinDistanceDP(token1, token2):
    distances = np.zeros((len(token1) + 1, len(token2) + 1))

    for t1 in range(len(token1) + 1):
        distances[t1][0] = t1

    for t2 in range(len(token2) + 1):
        distances[0][t2] = t2

    a = 0
    b = 0
    c = 0

    for t1 in range(1, len(token1) + 1):
        for t2 in range(1, len(token2) + 1):
            if (token1[t1-1] == token2[t2-1]):
                distances[t1][t2] = distances[t1 - 1][t2 - 1]
            else:
                a = distances[t1][t2 - 1]
                b = distances[t1 - 1][t2]
                c = distances[t1 - 1][t2 - 1]

                minimum = np.min((a,b,c))

                distances[t1][t2] = minimum + 1

    return distances[len(token1)][len(token2)]

def find_closest_registration(registrations, license_plate):
    '''
    registrations = list of strings

    Dla kazdej rejestracji obliczyc dist i zwrocic:
        1. najblizsza
        2. wszystkie o najmniejszym dystansie
    '''
    license_plate = str(license_plate)
    license_plate = license_plate[2:-2]
    scores = np.zeros(len(registrations))
    for ind, reg in enumerate(registrations):
        reg = str(reg)
        scores[ind] = levenshteinDistanceDP(reg, license_plate)
    # get index of min
    minimum = np.min(scores)
    i, = np.where(scores == minimum)
    i = int(i)
    #print(i)
    #print(scores[i])
    #print(license_plate)
    #print(registrations[i])
    return registrations[i]

#registrations = ['kr1asdf972', 'we12asdf798', 'bts20078k', 'afl1234', 'd7sf786', 'asdf678sd5af']
#license_plate = "bts2k78"
#find_closest_registration(registrations, license_plate)

def run(img_path, registrations):
    '''
    1. Wrzuc zdjecie
    2. koordynaty yolo
    3. bez zapisywania do pliku wez zdjecie i wrzuc
    w CRNN
    4. sprawdz wynik z baza danych
    5. jesli dist =< 3 pomarancz
       jesli dist = 0  green
    '''
    my_network = Yolo_detector()
    crnn_network = CRNN_detector()

    results = my_network.forward(img_path) # Dataframe
    img = cv2.imread(img_path)
    results = results.reset_index()  # make sure indexes pair with number of rows
    out_tab = []
    for index, row in results.iterrows():
        xmin, ymin, xmax, ymax = (int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax']))
        cropped_image = img[ymin:ymax, xmin:xmax]
        cropped_image = cv2.cvtColor(cropped_image,cv2.COLOR_BGR2RGB) # input [img_height, img_width, img_channels]
        cropped_image = np.array(cropped_image)
        cropped_image = np.transpose(cropped_image, (2, 0, 1)) # output [image_channels, image_height, image_width]
        preds = crnn_network.get_preds_cv2(cropped_image)
        literki = crnn_network.decode_predictions(preds)
        out_tab.append(literki)
    print(out_tab)
    for element in out_tab:
        license_found = find_closest_registration(registrations, element)
        print(license_found, element)

def read_registrations(filepath="registrations.txt"):
    '''
    IN: filepath to txt
    OUT: List of string 
    '''
    with open(filepath) as f:
        lines = f.read().splitlines() 
    return lines


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default='car.jpg', help='Zdjecie do przetworzenia')
    parser.add_argument('--regfile', type=str, default="registrations.txt", help='File txt with registrations')
    args = parser.parse_args()
    return args

def main():
    args = parse_opt()
    registrations = read_registrations(args.regfile)
    run(args.image, registrations)

if __name__ == "__main__":
   main()