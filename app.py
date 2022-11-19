from flask import *
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import model_from_json

app=Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT']=0


df=pd.read_csv('DATA/cancer_classification.csv')
X = df.drop('benign_0__mal_1',axis=1).values
y = df['benign_0__mal_1'].values
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=101)
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
col_names=['Clump Thickness','Uniformity of Cell Size','Uniformity of Cell Shape','Marginal Adhesion','Single Epithelial Cell Size','Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses','Class']
data= pd.read_csv('DATA/process.csv')
X_data = data.drop('Class',axis=1).values
y_data = data['Class'].values
X_train_data, X_test_data, y_train_data, y_test_data = train_test_split(X_data,y_data,test_size=0.25,random_state=101)
scaler_data = MinMaxScaler()
scaler_data.fit(X_train_data)
X_train_data = scaler_data.transform(X_train_data)


@app.route('/home/<name>/<int:age>')
def index(name,age):
    return '<h1>Hello! {} Age is {} </h1>'.format(name,age)


@app.route('/homepage')
def homepage():
    meanData = ["mean radius","mean texture","mean perimeter","mean area","mean smoothness","mean compactness","mean concavity","mean concave points","mean symmetry","mean fractal dimension"]
    errorData= ["radius error","texture error","perimeter error","area error","smoothness error","compactness error","concavity error","concave points error","symmetry error","fractal dimension error"]
    worstData=['worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness', 'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension']
    return render_template('front.html');
    

@app.route('/user')
def user():
    return render_template('user.html')
    
@app.route('/researcher')
def researcher():
    return render_template('index.html')

@app.route('/success',methods=['POST','GET'])
def print_data():
    global df,scaler
    if request.method=='POST':
        if request.form["Rname"]:
            meanData = ["mean radius","mean texture","mean perimeter","mean area","mean smoothness","mean compactness","mean concavity","mean concave points","mean symmetry","mean fractal dimension"]
            errorData= ["radius error","texture error","perimeter error","area error","smoothness error","compactness error","concavity error","concave points error","symmetry error","fractal dimension error"]
            worstData=['worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness', 'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension']
            l=list()
            value=0
            for i in range(30):
                if i<10:
                    l.append(float(request.form[meanData[i]]))
                elif i>=10 and i<20:
                    l.append(float(request.form[errorData[i-10]]))
                else:
                    l.append(float(request.form[worstData[i-20]]))
                    
                    
            """val=list(df.iloc(0)[0])
            for i in range(10,30):
                l.append(val[i])"""
            
            l=np.array(l[:30])
            l=scaler.transform([l])
            json_file=open('model_ann.json','r')
            loaded_model_json=json_file.read()
            json_file.close()
            loaded_model=model_from_json(loaded_model_json)
            loaded_model.load_weights("model_ann_weights.h5")
            pred=(loaded_model.predict(l) > 0.5).astype("int32")
            value=pred[0][0]
            name=request.form["Rname"]
            if value==0:
                return render_template("success.html",name=name)
            else:
                return render_template("failure.html",name=name)
        
      
@app.route('/succ',methods=['POST','GET'])
def print_user_data():
    if request.method=='POST':
        col_names=['Clump Thickness','Uniformity of Cell Size','Uniformity of Cell Shape','Marginal Adhesion','Single Epithelial Cell Size','Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses']
        l=list()
        for i in range(9):
            l.append(int(request.form[col_names[i]]))
        """val=list(data.iloc(0)[0])
        l=list()"""
        l=np.array(l[:9])
        l=scaler_data.transform([l])
        json_file=open('model_ann_new.json','r')
        loaded_model_json=json_file.read()
        json_file.close()
        loaded_model=model_from_json(loaded_model_json)
        loaded_model.load_weights("model_ann_new_weights.h5")
        pred=(loaded_model.predict(l) > 0.5).astype("int32")
        value=pred[0][0]
        name=request.form["Uname"]
        if value==0:
            return render_template("success.html",name=name)
        else:
            return render_template("failure.html",name=name)
        
    


if __name__=='__main__':
    app.run(debug=True)
