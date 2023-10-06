import numpy as np
from flask import Flask, render_template, request,jsonify
import pickle
app=Flask(__name__)
model= pickle.load(open('model.pkl','rb'))
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/login',methods = ['POST'])
def login():
    year=request.form["year"]
    do=request.form["do"]
    ph = request.form["ph"]
    co= request.form["co"]
    bod= request.form["bod"]
    na= request.form["na"]
    tc= request.form["tc"]
    total=[[float (do), float (ph), float (co), float (bod), float (na), float(tc),int(year)]]
    y_pred = model.predict(total)
    #y_pred =y_ pred[[0]]
    if(y_pred >=95 and y_pred<= 100):
        return render_template("index.html",showcase=f"Excellent.The predicted value is {str(y_pred)}") 
    elif(y_pred >=89 and y_pred <= 94):
        return render_template("index.html",showcase=f"Very good.The predicted value is {str(y_pred)}") 
    elif(y_pred >=80 and y_pred <= 88):
        return render_template("index.html",showcase=f"Good.The predicted value is {str(y_pred)}")
    elif(y_pred>=65 and y_pred <=79):
        return render_template("index.html", showcase =f"Fair.The predicted value is {str(y_pred)}")
    elif(y_pred<= 45 and y_pred <=64):
        return render_template("index.html", showcase =f"Marginal.The predicted value is {str(y_pred)}")
    else:
        return render_template("index.html", showcase=f"Poor.The predicted value is {str(y_pred)}")
if __name__=='__main__':
    app.run(debug=True,port=5000)







