import os
import shutil

import random

import pandas as pd

from tqdm import tqdm


df = pd.read_csv(os.path.join("AResizedDataset", "Reference.csv"))


if os.path.isdir("ASplitted_Dataset"):
    shutil.rmtree("ASplitted_Dataset")
os.mkdir("ASplitted_Dataset")
os.mkdir(os.path.join("ASplitted_Dataset", "Train"))
os.mkdir(os.path.join("ASplitted_Dataset", "Validation"))


for Class in range(len(df)):
    
    os.mkdir(os.path.join("ASplitted_Dataset", "Train", str(Class)))
    
    os.mkdir(os.path.join("ASplitted_Dataset", "Validation", str(Class)))

#for Class in tqdm(df.iloc[:,0]):

Files = os.listdir(os.path.join("AResizedDataset", "0"))
random.shuffle(Files)
for File in Files[:1053]:
    shutil.copy(os.path.join("AResizedDataset", "0", File), os.path.join("ASplitted_Dataset", "Validation", "0", File))

for File in Files[1053:]:
    shutil.copy(os.path.join("AResizedDataset", "0", File), os.path.join("ASplitted_Dataset", "Train", "0", File))

Files = os.listdir(os.path.join("AResizedDataset", "1"))
random.shuffle(Files)                   
for File in Files[:555]:
    shutil.copy(os.path.join("AResizedDataset", "1", File), os.path.join("ASplitted_Dataset", "Validation", "1", File))
for File in Files[555:]:
    shutil.copy(os.path.join("AResizedDataset", "1", File), os.path.join("ASplitted_Dataset", "Train", "1", File))
Files = os.listdir(os.path.join("AResizedDataset", "2"))
random.shuffle(Files)        
for File in Files[:500]:
    shutil.copy(os.path.join("AResizedDataset", "2", File), os.path.join("ASplitted_Dataset", "Validation", "2", File))
for File in Files[500:]:
    shutil.copy(os.path.join("AResizedDataset", "2", File), os.path.join("ASplitted_Dataset", "Train", "2", File))
Files = os.listdir(os.path.join("AResizedDataset", "3"))
random.shuffle(Files)        
for File in Files[:458]:
    shutil.copy(os.path.join("AResizedDataset", "3", File), os.path.join("ASplitted_Dataset", "Validation", "3", File))
for File in Files[458:]:
    shutil.copy(os.path.join("AResizedDataset", "3", File), os.path.join("ASplitted_Dataset", "Train", "3", File))
Files = os.listdir(os.path.join("AResizedDataset", "4"))
random.shuffle(Files)       
for File in Files[:428]:
    shutil.copy(os.path.join("AResizedDataset", "4", File), os.path.join("ASplitted_Dataset", "Validation", "4", File))
for File in Files[428:]:
    shutil.copy(os.path.join("AResizedDataset", "4", File), os.path.join("ASplitted_Dataset", "Train", "4", File))
Files = os.listdir(os.path.join("AResizedDataset", "5"))
random.shuffle(Files)     
for File in Files[:358]:
    shutil.copy(os.path.join("AResizedDataset", "5", File), os.path.join("ASplitted_Dataset", "Validation", "5", File))
for File in Files[358:]:
    shutil.copy(os.path.join("AResizedDataset", "5", File), os.path.join("ASplitted_Dataset", "Train", "5", File))
Files = os.listdir(os.path.join("AResizedDataset", "6"))
random.shuffle(Files)     
for File in Files[:183]:
    shutil.copy(os.path.join("AResizedDataset", "6", File), os.path.join("ASplitted_Dataset", "Validation", "6", File))
for File in Files[183:]:
    shutil.copy(os.path.join("AResizedDataset", "6", File), os.path.join("ASplitted_Dataset", "Train", "6", File))
Files = os.listdir(os.path.join("AResizedDataset", "7"))
random.shuffle(Files)     
for File in Files[:180]:
    shutil.copy(os.path.join("AResizedDataset", "7", File), os.path.join("ASplitted_Dataset", "Validation", "7", File))
for File in Files[180:]:
    shutil.copy(os.path.join("AResizedDataset", "7", File), os.path.join("ASplitted_Dataset", "Train", "7", File))    
        

shutil.copy(os.path.join("AResizedDataset", "Reference.csv"), os.path.join("ASplitted_Dataset", "Reference.csv"))
