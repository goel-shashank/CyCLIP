import os
import math
import zlib
import uuid
import shelve
import tarfile
import argparse
import requests
import pandas as pd

from io import BytesIO
from tqdm import tqdm
from PIL import Image
from multiprocessing import Pool
from torchvision import transforms

transform = transforms.Compose([transforms.Resize(224, interpolation = transforms.InterpolationMode.BICUBIC), transforms.CenterCrop(224)])

def download(row):
    rfile = f"images/{zlib.crc32(row['image'].encode('utf-8')) & 0xffffffff}.png"
    file = f"{row['dir']}/{rfile}"
    
    if(os.path.isfile(file)):
        row["status"] = 200
        row["file"] = rfile
        return row

    try:
        response = requests.get(row["image"], stream = False, timeout = 10, allow_redirects = True)
        row["status"] = response.status_code
    except Exception as e:
        row["status"] = 404
        return row
        
    if(response.ok):
        try:
            response.raw.decode_content = True 
            image = Image.open(BytesIO(response.content)).convert("RGB")
            image = transform(image)
            image.save(file)
        except:
            row["status"] = 404
            return row
        
        row["file"] = rfile
        
    return row    

def apply(args):
    index, df, function = args
    df = df.apply(function, axis = 1)
    return (index, df)

def multiprocess(df, function, dir, hash): 
    with shelve.open(f"{dir}/.{hash}") as file:
        bar = tqdm(total = math.ceil(len(df) / 50))
        
        finished = set(map(int, file.keys()))
        for key in file.keys():
            bar.update()

        data = [(index, df[i:i + 50], function) for index, i in enumerate(range(0, len(df), 50)) if index not in finished]
       
        if(len(data) > 0):
            with Pool() as pool:
                for result in pool.imap_unordered(apply, data, 2):
                    file[str(result[0])] = result
                    bar.update()
        
        bar.close()
        
        keys = sorted([int(k) for k in file.keys()])
        df = pd.concat([file[str(key)][1] for key in keys])
        df = df[["file", "caption"]].rename(columns = {"file": "image"})
        
        return df

def run(options):
    os.makedirs(options.dir, exist_ok = True)
    os.makedirs(os.path.join(options.dir, "images"), exist_ok = True)
    
    df = pd.read_csv(options.file, sep = "\t", names = [ "caption", "image"])
    df["dir"] = options.dir
    df = df[options.start:options.end]
    
    df = multiprocess(df, function = download, dir = options.dir, hash = options.hash)    
    df.to_csv(f"{options.dir}/train.csv", index = False)

if(__name__ == "__main__"):
    parser = argparse.ArgumentParser()

    parser.add_argument("-f,--file", dest = "file", type = str, default = None, help = "File")
    parser.add_argument("-d,--dir", dest = "dir", type = str, default = None, help = "Directory")
    parser.add_argument("-s,--start", dest = "start", type = int, default = 0, help = "Start index")
    parser.add_argument("-e,--end", dest = "end", type = int, default = 1000000000000, help = "End index")

    options = parser.parse_args()
    options.hash = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{options.file}-{options.dir}-{options.start}-{options.end}"))
    
    run(options)
