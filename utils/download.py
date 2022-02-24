import pandas as pd
import numpy as np
import requests
import zlib
import os
import sys
import argparse
import shelve
from PIL import Image
from io import BytesIO
from multiprocessing import Pool
from tqdm import tqdm
from torchvision import transforms as T
import tarfile

transform = T.Compose([T.Resize(224, interpolation = T.InterpolationMode.BICUBIC), 
                        T.CenterCrop(224)])

def _df_split_apply(tup_arg):
    split_ind, subset, func = tup_arg
    r = subset.apply(func, axis=1)
    file_obj= tarfile.open(f"{subset.iloc[0]['folder']}/{split_ind}.tar","w") 
    for fname in r["file"]:
        try:
            file_obj.add(fname)
        except:
            continue
    file_obj.close()
    for fname in r["file"]:
        try:
            os.remove(fname)   
        except:
            pass
    return (split_ind, r)

def df_multiprocess(df, chunk_size, func, dataset_name, dir_path):
    print("Generating parts...")
    with shelve.open('%s/%s_%s_%s_results.tmp' % (dir_path, dataset_name, func.__name__, chunk_size)) as results:
 
        pbar = tqdm(total=len(df), position=0)
        # Resume:
        finished_chunks = set([int(k) for k in results.keys()])
        pbar.desc = "Resuming"
        for k in results.keys():
            pbar.update(len(results[str(k)][1]))

        pool_data = ((index, df[i:i + chunk_size], func) for index, i in enumerate(range(0, len(df), chunk_size)) if index not in finished_chunks)
        print(int(len(df) / chunk_size), "parts.", chunk_size, "per part.", "Using", os.cpu_count(), "processes")
 
        pbar.desc = "Downloading"
        with Pool() as pool:
            for i, result in enumerate(pool.imap_unordered(_df_split_apply, pool_data, 2)):
                results[str(result[0])] = result
                pbar.update(len(result[1]))
        pbar.close()

    print("Finished Downloading.")
    return

# Unique name based on url
def _file_name(row):
    return "%s/%s_%s.png" % (row['folder'], row.name, (zlib.crc32(row['image'].encode('utf-8')) & 0xffffffff))

# Don't download image, just check with a HEAD request, can't resume.
# Can use this instead of download_image to get HTTP status codes.
def check_download(row):
    fname = _file_name(row)
    try:
        # not all sites will support HEAD
        response = requests.head(row['image'], stream=False, timeout=5, allow_redirects=True)
        row['status'] = response.status_code
        row['headers'] = dict(response.headers)
    except:
        # log errors later, set error as 408 timeout
        row['status'] = 408
        return row
    if response.ok:
        row['file'] = fname
    return row

def download_image(row):
    fname = _file_name(row)
    # Skip Already downloaded, retry others later
    if os.path.isfile(fname):
        row['status'] = 200
        row['file'] = fname
        row['size'] = os.stat(row['file']).st_size
        return row

    try:
        # use smaller timeout to skip errors, but can result in failed downloads
        response = requests.get(row['image'], stream=False, timeout=10, allow_redirects=True)
        row['status'] = response.status_code
        row['headers'] = dict(response.headers)
    except Exception as e:
        # log errors later, set error as 408 timeout
        row['status'] = 408
        return row
        
    if response.ok:
        try:
            response.raw.decode_content = True 
            img = Image.open(BytesIO(response.content)).convert('RGB')
            img = transform(img)
            img.save(fname)
            # with open(fname, 'wb') as out_file:
            #     # some sites respond with gzip transport encoding
            #     response.raw.decode_content = True
            #     out_file.write(response.content)
            row['size'] = os.stat(fname).st_size
        except:
            # This is if it times out during a download or decode
            row['status'] = 408
            return row
        row['file'] = fname
    return row

def open_tsv(fname, folder):
    print("Opening %s Data File..." % fname)
    df = pd.read_csv(fname, sep='\t', names=["image","caption"])
    df['folder'] = folder
    print("Processing", len(df), " Images:")
    return df

def df_from_shelve(chunk_size, func, dataset_name, dir_path):
    print("Generating Dataframe from results...")
    with shelve.open('%s/%s_%s_%s_results.tmp' % (dir_path, dataset_name, func.__name__, chunk_size)) as results:
        keylist = sorted([int(k) for k in results.keys()])
        df = pd.concat([results[str(k)][1] for k in keylist], sort=True)
    return df

def run_download(start = None, end = None, images_per_part = 50, data_name = 'train', dir_path = None, filename = 'train.tsv'):
    os.makedirs(os.path.join(dir_path, data_name), exist_ok = True)
    filepath = f'data/{filename}'
    df = open_tsv(filepath, f'{dir_path}/{data_name}')
    df = df[start:min(end, len(df))]
    df_multiprocess(df=df, chunk_size=images_per_part, func=download_image, dataset_name=data_name, dir_path = dir_path)
    df = df_from_shelve(chunk_size=images_per_part, func=download_image, dataset_name=data_name, dir_path = dir_path)
    df.to_csv(f'{dir_path}/downloaded_{data_name}_report.tsv.gz', compression='gzip', sep='\t', header=False, index=False)
    print("Saved.")

def get_downloaded_report(dir_path = ".", data_name = 'validation'):
    df = pd.read_csv(f'{dir_path}/downloaded_{data_name}_report.tsv.gz', compression='gzip', sep='\t', names = ["captions", "file", "folder", "headers", "size", "status", "image"])
    return df

if(__name__ == "__main__"):
    parser = argparse.ArgumentParser()

    parser.add_argument("-s,--start", dest = "start", type = float, default = None, help = "Start index")
    parser.add_argument("-e,--end", dest = "end", type = float, default = None, help = "End index")

    options = parser.parse_args()
    run_download(start = int(options.start * 1e6), end = int(options.end * 1e6), dir_path = f"./data/CC12M-{options.start}-{options.end}")

