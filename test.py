from flask import Flask, request, render_template
from flask_cors import cross_origin
import os
from utils import get_base_url, allowed_file, and_syntax

from model import *  #(jimmy python program)


# setup the webservver
port = 12123
base_url = get_base_url(port)
app = Flask(__name__, static_url_path=base_url+'static')



link = input('enter your link: ')
message = predict_amazon(link)
print(message)