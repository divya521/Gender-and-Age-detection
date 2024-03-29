import os
import shutil
import random
import pandas as pd

from tqdm import tqdm

df = pd.read_csv(os.path.join("ResizedDataset", "Reference.csv"))

if os.path.isdir("Splitted_Dataset"):
    shutil.rmtree("Splitted_Dataset")

os.mkdir("Splitted_Dataset")
os.mkdir(os.path.join("Splitted_Dataset", "Train"))
os.mkdir(os.path.join("Splitted_Dataset", "Validation"))

for Class in range(len(df)):
    os.mkdir(os.path.join("Splitted_Dataset", "Train", str(Class)))
    os.mkdir(os.path.join("Splitted_Dataset", "Validation", str(Class)))

for Class in tqdm(df.iloc[:,0]):
    Files = os.listdir(os.path.join("ResizedDataset", str(Class)))
    random.shuffle(Files)


    for File in Files[:1745]:
        shutil.copy(os.path.join("ResizedDataset", str(Class), File), os.path.join("Splitted_Dataset", "Validation", str(Class), File))
    for File in Files[1745:]:
        shutil.copy(os.path.join("ResizedDataset", str(Class), File), os.path.join("Splitted_Dataset", "Train", str(Class), File))

shutil.copy(os.path.join("ResizedDataset", "Reference.csv"), os.path.join("Splitted_Dataset", "Reference.csv"))
