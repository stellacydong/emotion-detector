from flask import Flask, request, render_template
from flask_cors import cross_origin
import os
from utils import get_base_url, allowed_file, and_syntax

from model import *  #(jimmy python program)

# setup the webservver
#port = 12123
#base_url = get_base_url(port)
app = Flask(__name__)

IMAGE_FOLDER=os.path.join('static','images')
app.config['UPLOAD_FOLDER']=IMAGE_FOLDER

# homepage 
@app.route('/')
def home():
    return render_template('home.html')

# request test and show the result 
@app.route("/result",methods=["GET","POST"])
@cross_origin()
def result():
    if request.method=="POST":
        review = (request.form["Review"])
        prediction = predict_text(review)
        output=""
        if prediction=='Positive':
            output="Positive"
            img_filename=os.path.join(app.config['UPLOAD_FOLDER'],'thumb_up.png')
        else:
            output="Negative"
            img_filename=os.path.join(app.config['UPLOAD_FOLDER'],'thumb_down.png')
        return render_template('home.html',prediction_text=f'This is a {output} review.',image=img_filename)


# request test and show the result 
@app.route("/count",methods=["GET","POST"])
@cross_origin()
def count():
    if request.method=="POST":
        link = (request.form["link"])
        title, num_of_reviews, message = predict_amazon(link)
        if message == 'Not url':
            img_filename=os.path.join(app.config['UPLOAD_FOLDER'],'thumb_down.png')
            #word_filename=os.path.join(app.config['UPLOAD_FOLDER'], 'bar.png')
            return render_template('home.html',prediction_text='Not a url',image=img_filename)
        elif message=='Success':
            bar_filename=os.path.join(app.config['UPLOAD_FOLDER'], 'bar.png')
            pie_filename=os.path.join(app.config['UPLOAD_FOLDER'], 'pie.png')
            all_filename=os.path.join(app.config['UPLOAD_FOLDER'], 'All_wordcloud.png')
            pos_filename=os.path.join(app.config['UPLOAD_FOLDER'], 'Positive_wordcloud.png')
            neg_filename=os.path.join(app.config['UPLOAD_FOLDER'], 'Negative_wordcloud.png')
            
            return render_template('home.html',prediction_text=f'Analyzed {num_of_reviews} reviews', word_cloud_img = all_filename, pos_cloud_img = pos_filename, neg_cloud_img = neg_filename, bar_img = bar_filename, pie_img = pie_filename)
        else:
            img_filename=os.path.join(app.config['UPLOAD_FOLDER'],'thumb_down.png')
            #word_filename=os.path.join(app.config['UPLOAD_FOLDER'], 'bar.png')
            return render_template('home.html',prediction_text=f'Error when trying URL. Try to click "See all reviews". If this still doesn\'t work, the host might be broken. {message}',image=img_filename)
    return render_template("home.html")


if __name__ == "__main__":
    predict_text('LOAD MODEL!')
    # change the code.ai-camp.org to the site where you are editing this file.
    print("Try to open\n\n    https://cocalc8.ai-camp.org" + base_url + '\n\n')
    # remove debug=True when deploying it
    app.run(host = '0.0.0.0', port=port, debug=True)
    import sys; sys.exit(0)
