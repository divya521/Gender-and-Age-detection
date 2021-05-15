import os
import shutil
from facenet_pytorch import MTCNN
mtcnn = MTCNN(device='cpu', keep_all = False)

if os.path.isdir("AResizedDataset"):
    shutil.rmtree("AResizedDataset")
os.mkdir("AResizedDataset")
os.mkdir(os.path.join("AResizedDataset", "0"))
os.mkdir(os.path.join("AResizedDataset", "1"))
os.mkdir(os.path.join("AResizedDataset", "2"))
os.mkdir(os.path.join("AResizedDataset", "3"))
os.mkdir(os.path.join("AResizedDataset", "4"))
os.mkdir(os.path.join("AResizedDataset", "5"))
os.mkdir(os.path.join("AResizedDataset", "6"))
os.mkdir(os.path.join("AResizedDataset", "7"))

size = total_data.index

for i in size:
    Label = total_data.loc[i,"age"]
    path = "faces/" +total_data.user_id.loc[i] +"/coarse_tilt_aligned_face." +str(total_data.face_id.loc[i])+"."+total_data.original_image.loc[i]
     
    try:
        img = cv2.imread(path)
        box, probs =mtcnn.detect(img, landmarks=False)
       
        try:
            box = np.clip(box,0,max(img.shape))
            (x, y, x1, y1) = box[0]
            face = img[int(y):int(y1), int(x):int(x1)]
        except:
            face = img
        img = cv2.resize(face, (200,200), interpolation = cv2.INTER_AREA )
        img = np.array(img)
        cv2.imwrite(os.path.join("AResizedDataset", str(Label), os.path.basename(path)),img)
    except:
        img = img
    
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
   
   
