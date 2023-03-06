# -*- coding : utf-8-*-
# coding:utf-8
import base64
import os
from io import BytesIO
from monitor import models
from django.http import HttpResponse
from django.shortcuts import render, redirect
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

from upload import settings
import csv
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from itertools import chain


def upload(request):
    return render(request, 'upload.html', )

def preview(request):
    if request.method == "POST":
        file = request.FILES.get('file1')
        data = pd.read_csv(file)
        data_html = data.to_html(classes='table table-striped table-bordered table-hover table-condensed')

        models.Users.objects.all()
        models.Users.objects.all().values('username')  # 只取username列
        models.Users.objects.all().values_list('id', 'username')  # 取出id和username列，并生成一个列表
        models.Users.objects.get(id=1)

        context = {'loaded_data': data_html}


    return render(request, "dataflow/table.html", context)

def predict(request):
    sum_up=''
    if request.method == "POST":
        file = request.FILES.get('file1')
        data = pd.read_csv(file, encoding='gbk')  # 读取数据
        data.to_csv('templates\副本.csv', float_format='%.2f', index=False)  # 保留两位小数
        dataset = pd.read_csv('templates\副本.csv')
        dataset = dataset.drop(columns=["Unnamed: 0"])  # 删除第一列
        start = request.POST.get('start')
        end = request.POST.get('end')
        X = dataset.loc[:, start:end].values  # 取前端输入的列数据(返回列表形式)

        cols = request.POST.getlist('check_box_list')
        mod = request.POST.get('model')


        new_list = []
        for y_name in cols:
            y_index = cols.index(y_name)
            Y = dataset.loc[:, [y_name]].values  # 取cols列数据
            from sklearn.model_selection import train_test_split  # 分割训练集和测试集
            test_size1 = int(request.POST.get('test_size'))
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size1 / 100,
                                                                random_state=0)

            if mod == "偏最小二乘回归模型":
                model = PLSRegression(n_components=500)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_1d = list(chain.from_iterable(y_pred))
                y_pred_2f = [round(x, 2) for x in y_pred_1d]

            elif mod == "支持向量回归模型":
                model = svm.SVR(kernel='rbf',
                                C=20000,
                                gamma=0.1)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_2f = [round(x, 2) for x in y_pred]

            else:
                model = RandomForestRegressor(n_estimators=151,
                                              max_features=8,
                                              ccp_alpha=0.3,
                                              n_jobs=-1,
                                              warm_start=True,
                                              random_state=1,
                                              verbose=2,
                                              )
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_2f = [round(x, 2) for x in y_pred]


            y_pred_2f = [y_pred_2f[i:i + 18] for i in range(0, len(y_pred_2f), 18)]
            new_list.append(y_pred_2f)
        for i in range(len(cols)):
            cols[i]+="预测值(g/kg):"
        cols+=['','','','','','']
        new_list+=['','','','','','']
        name1=cols[0]
        name2=cols[1]
        name3=cols[2]
        name4=cols[3]
        name5=cols[4]
        name6=cols[5]
        name7=cols[6]
        new_list1=new_list[0]
        new_list2=new_list[1]
        new_list3=new_list[2]
        new_list4=new_list[3]
        new_list5=new_list[4]
        new_list6=new_list[5]
        new_list7=new_list[6]

        sum_up='本轮采用'+mod+'进行建模预测，测试集样本占比'+str(test_size1)+'%'+'，各项预测数据如下(横向)：'

        # from tempfile import TemporaryFile
        #
        # pred_data = TemporaryFile()
        # ngData = pd.read_csv('../data/namegender.csv')

        # pred_data[y_name]=y_pred_2f

        # response = HttpResponse(content_type='text/csv')
        # # attachment 代表这个csv文件作为一个附件的形式下载
        # # filename='abc.csv' 指定下载的文件名字
        # response['Content-Disposition'] = "attachment;filename='预测文件.csv'"
        # writer = csv.writer(response)
        #

        # f[y_name]=y_pred_2f
        # f.to_csv('templates\pred1.csv', mode='a', index=y_index)

        # def rf_(x_train=None, y_train=None, x_test=None):
        #     return y_pred
        #
        # def make_dataset(df, f_start=None, f_end=None, target=None, train_size=None):
        #     X = df.iloc[:, f_start:f_end]
        #     Y = df[target]
        #     x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=1 - train_size)
        #     return x_train, x_test, y_train, y_test
        # def r2(y_pred, y_test):
        #     return r2_score(y_pred, y_test)
        #
        # def rmse(y_pred, y_test):
        #     return 1 / np.sqrt(mean_squared_error(y_pred, y_test))
        #
        # def rpd(y_pred, y_test):
        #     temp = r2(y_pred, y_test)
        #     return 1 / np.sqrt(1 - temp)

    contests = {
        'name1':name1,
        'name2':name2,
        'name3':name3,
        'name4':name4,
        'name5':name5,
        'name6':name6,
        'name7':name7,
        'new_list1':new_list1,
        'new_list2':new_list2,
        'new_list3':new_list3,
        'new_list4':new_list4,
        'new_list5':new_list5,
        'new_list6':new_list6,
        'new_list7':new_list7,
        'sum_up':sum_up,
    }
    return render(request, 'upload.html', contests)


def fileDownload(request):
    if request.method == "POST":
        file = request.FILES.get('file1')
        data = pd.read_csv(file)  # 读取数据
        data.to_csv('templates\副本.csv', float_format='%.2f', index=False)  # 保留两位小数
        dataset = pd.read_csv('templates\副本.csv')
        dataset = dataset.drop(columns=["Unnamed: 0"])  # 删除第一列
        start = int(request.POST.get('start'))
        end = int(request.POST.get('end'))
        X = dataset.iloc[:, start - 1:end - 1].values  # 取前端输入的列数据(返回列表形式)

        cols = request.POST.getlist('check_box_list')
        mod = request.POST.get('model')
        #
        # list_head = ['样本编号']
        # sample_num=data.loc[:, ["Unnamed: 0"]].values
        # list_value = [list(chain.from_iterable(sample_num))]
        list_head = []
        list_value = []
        for y_name in cols:
            Y = dataset.loc[:, [y_name]].values  # 取cols列数据
            from sklearn.model_selection import train_test_split  # 分割训练集和测试集
            test_size1 = int(request.POST.get('test_size'))
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size1 / 100,
                                                                random_state=0)

            if mod == "偏最小二乘回归模型":
                model = PLSRegression(n_components=500)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_1d = list(chain.from_iterable(y_pred))
                y_pred_2f = [round(x, 2) for x in y_pred_1d]

            elif mod == "支持向量回归模型":
                model = svm.SVR(kernel='rbf',
                                C=20000,
                                gamma=0.1)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_2f = [round(x, 2) for x in y_pred]

            else:
                model = RandomForestRegressor(n_estimators=151,
                                              max_features=8,
                                              ccp_alpha=0.3,
                                              n_jobs=-1,
                                              warm_start=True,
                                              random_state=1,
                                              verbose=2,
                                              )
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_2f = [round(x, 2) for x in y_pred]

            list_head.append(y_name + '实测值')
            list_head.append(y_name + '预测值')

            list_value.append(list(chain.from_iterable(y_test)))
            list_value.append(y_pred_2f)

        from itertools import zip_longest
        export_data = zip_longest(*list_value, fillvalue='')

        response = HttpResponse(content_type='text/csv')
        # attachment 代表这个csv文件作为一个附件的形式下载
        # filename='abc.csv' 指定下载的文件名字
        response['Content-Disposition'] = "attachment;filename=file.csv"

        wr = csv.writer(response)
        wr.writerow(list_head)
        wr.writerows(export_data)

    return response

