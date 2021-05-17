from http.server import BaseHTTPRequestHandler, HTTPServer
import time
from flask import Flask, flash, request, redirect, render_template
import os
from pathlib import Path
import torch
import torchvision
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import gzip
from torch.utils.data import DataLoader
import torchvision.transforms as trans
import struct
import sys
import array
from tqdm import tqdm
from . import server_utils

# Begin Server Code
#####################################################
app=Flask(__name__)
app.secret_key = "1234"

hostName = "localhost"
serverPort = 8080

### upload_form()
# render on access - upload
@app.route('/upload')
def upload_form():
    return render_template('upload.html')

### loading()
# render on access - upload
@app.route('/loading')
def loading():
    return render_template('loading.html')



### file_receiver
## file received - act
@app.route('/upload', methods=['POST'])
def file_receiver():
    if request.method != 'POST':
        flash('invalid request')
        return redirect(request.url)

    ### Handle actions relevant to user
    ### Get all user selections and verify prior to image download
    user = str(request.form.get('user'))
    dist_name = str(request.form.get('dist'))
    out_samples = int(request.form.get('out_ct'))
    latent_dim = int(request.form.get('lat_sz'))
    conv_ct = int(request.form.get('conv_ct'))
    img_size = int(request.form.get('img_sz'))
    n_epochs = int(request.form.get('epochs'))
    hide_size = int(request.form.get('hide_sz'))



    ### Validate user inputs
    if latent_dim < 1 or out_samples < 1 or n_epochs < 1:
        flash('Latent Dimension / Samples / Epochs must be > 0')
        return redirect(request.url)


    ### create dir / get path init
    parse_username(user)


    ## if valid post try file
    if 'files[]' not in request.files:
            flash('upload file missing or invalid')
            return redirect(request.url)
    files = request.files.getlist('files[]')

    user_path = "./user_files/" + user
    save_path = user_path + "/data"

    for file in files:
        if file and file_valid(file.filename):
            filename = (file.filename)
            file.save(os.path.join(save_path,  filename))
    flash('File(s) uploaded')

    gan = GAN(dist_name, out_samples, user_path, latent_dim, conv_ct, n_epochs, img_size, hide_size, filename)

    return redirect('/loading')


# Begin Server Helpers
#########################################################################

def file_valid(name):
    return True ## TODO

def parse_username(user):
    user_path = ('./user_files/' + user)
    if os.path.isdir(user):
        dir = Path(user_path)
        try:
            dir.rmdir()
        except OSError as e:
            print("Error: %s : %s" % (dir_path, e.strerror))
            print("Failed to find a known existing user bin")


    os.mkdir(user_path)
    os.mkdir(user_path + "/data")

    return user_path

def parse_images(filename):
    with gzip.open(filename, 'rb') as fh:
        magic, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))
        return np.array(array.array("B", fh.read()), dtype=np.uint8).reshape(num_data, rows, cols)


####### MAIN
#############################################
if __name__ == "__main__":
    print (sys.path)
    exit()
    app.run(host='127.0.0.1',port=4000,debug=True,threaded=True)

############################
