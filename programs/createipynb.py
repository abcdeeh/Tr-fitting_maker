def createipynb(tittle,filename,FileDir,A,B,C,D,E,F):#fittingコード用
    import os
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor
    nb = nbformat.v4.new_notebook()
    code1 = """#入力部
#グラフのデータタイトル
tittle="{}"
#読み込みファイル名(txtファイル対応)
filename="{}"
#kerrの調整(元データがKerrの場合はA～Dの変数を1にE,Fを100に設定)
#Kerr=[(A/B)*(C/D)*data]/[(E-F)/2]*100[%]
#A=オシロの標準資料磁化印加時最大最小電圧差
A={}
#B=標準資料のロックインアンプ電圧値[V]
B={}
#C=サンプルLIA電圧値[V]
C={}
#D=サンプルオシロ電圧値[V]
D={}
#E=サンプル磁場印加時の正電圧値[V]
E={}
#F=サンプル磁場印加時の負電圧値[V]
F={}
#入力ここまで
import numpy as np
data2=np.loadtxt(filename,usecols=(0,1))#[delay time(ps),Kerr signal[V]]
x=data2[:,0]
y=A/B*C/D/((E-F)/2)*data2[:,1]*100
diff=np.append(np.diff(x)[0],np.diff(x))#幅取得
dt=np.array([diff[0]])#幅種類分け
for i in range (diff.shape[0]):#幅種類分け
    if 1.01*dt[dt.shape[0]-1]<diff[i] or diff[i]<0.99*dt[dt.shape[0]-1]:
        dt=np.append(dt,diff[i])
FFT_List=np.where((0.99*dt[dt.shape[0]-1]<diff)&(diff<1.01*dt[dt.shape[0]-1]))
xFFT=np.array(x[FFT_List])[y[FFT_List]!=0]
yFFT=np.array(y[FFT_List])[y[FFT_List]!=0]
x=x[y!=0]
y=y[y!=0]

print("amp:"+str(A/B*C/D/((E-F)/2)*100))
print("データの時間差(ps)")
print(dt)""".format(tittle,filename,A,B,C,D,E,F)
    code2="""from scipy import signal,interpolate
#フーリエ変換
import matplotlib.pyplot as plt
dt_FFT=(xFFT[xFFT.shape[0]-1]-xFFT[0])/xFFT.shape[0]
y_fft = np.fft.fft(yFFT)
freq = np.fft.fftfreq(yFFT.shape[0], d=dt_FFT)
Amp = np.abs(y_fft/(yFFT.shape[0]/2))
peaks_freq,_ = signal.find_peaks(Amp[0:int(yFFT.shape[0]/2)])
top_freq=peaks_freq[np.argmax(Amp[peaks_freq])]
fig4, ax4 = plt.subplots()
ax4.plot(10**3*freq[1:int(yFFT.shape[0]/2)], Amp[1:int(yFFT.shape[0]/2)])
ax4.scatter(10**3*freq[top_freq],Amp[top_freq])
ax4.set_xlabel("Freqency [GHz]")
ax4.set_ylabel("Amplitude")
ax4.set_title(tittle)
ax4.grid()
plt.show()
#抽出された歳差運動周波数がおかしかったらイコールの右側の値で調整してください。[THz]
exp_freq=freq[top_freq]"""
    code3="""from scipy.optimize import leastsq
peaks,_ = signal.find_peaks(yFFT,distance=int(0.9/exp_freq/dt[dt.shape[0]-1]))
bottoms,_=signal.find_peaks(-yFFT,distance=int(0.9/exp_freq/dt[dt.shape[0]-1]))
#入力ここから
#参考データ
#fitting参考最大値です。指定したい最大ピーク(peaks[n番目-1])若しくは最小ピーク(bottoms[n番目-1])を"="の隣に入れてください。
end=x.shape[0]-1
#入力ここまで
peaks=peaks[peaks<=end]
bottoms=bottoms[bottoms<=end]
x1=xFFT[:end+1]
y1=yFFT[:end+1]
#最大最小の補間及び中間値
#線形補間
peaksfunc=interpolate.interp1d(x1[peaks],y1[peaks],kind='linear',fill_value='extrapolate')
bottomsfunc=interpolate.interp1d(x1[bottoms],y1[bottoms],kind='linear',fill_value='extrapolate')
#中間値
k=int(x1.shape[0])
middle_y1=[]
for i in range(k):
    middle_y1.append(float((peaksfunc(x1[i])+bottomsfunc(x1[i]))/2))
#中間値fitting用残差関数
def fit_exp1(parameter, x, y):
    residual = y - (parameter[0] * np.exp(parameter[1] * x) + parameter[2])
    return residual
b_est=middle_y1[np.array(middle_y1).shape[0]-1]
A_est=(peaksfunc(0)+bottomsfunc(0))/2-b_est
k_est=-abs(np.log(abs((middle_y1[0]-b_est)/A_est))/x1[0])
parameter0 = [ A_est,k_est,middle_y1[np.array(middle_y1).shape[0]-1]]#fitting初期値
fitting_func = leastsq(fit_exp1, parameter0, args=(x1 , middle_y1))[0]#fittingパラメータ
middle_func = lambda A, k, x, b: A * np.exp(k * x) + b#プロット用 式
fig,ax = plt.subplots()
ax.plot(x, y, label="data")
ax.scatter(xFFT[peaks],yFFT[peaks], color='red', label="peaks")
ax.scatter(xFFT[bottoms],yFFT[bottoms], color='blue', label="bottoms")
ax.plot(x1,peaksfunc(x1), color='red', linestyle=":")
ax.plot(x1,bottomsfunc(x1),color='blue', linestyle=":")
ax.plot(x1, middle_y1, color="purple",label="middle", linestyle=":")
ax.plot(x1,middle_func(fitting_func[0],fitting_func[1],x1,fitting_func[2]), color='green',label="middle-fitting")
ax.legend()
ax.set_ylabel("Kerr signal [%]")
ax.set_xlabel("delay time [ps]")
ax.set_title(tittle)
plt.show()"""
    code4="""from numpy import log as ln
def exp_string(A, k):
    return "Data = %0.4f*e^(%0.4f*x)" % (A, k)
y12=[]
k=int(x1.shape[0])
for i in range(k):
    y12.append(float(y1[i]-middle_func(fitting_func[0],fitting_func[1],x1[i],fitting_func[2])))
y12=np.array(y12)
#peak and bottom
peaks1,_ = signal.find_peaks(y12,distance=int(0.9/exp_freq/dt[dt.shape[0]-1]),height=abs(np.max(y12))/100)
bottoms1,_=signal.find_peaks(-y12,distance=int(0.9/exp_freq/dt[dt.shape[0]-1]),height=abs(np.max(-y12))/100)
#fitting用残差関数
def fit_exp(parameter, x, y):
    residual = y - (parameter[0] * np.exp(parameter[1] * x))
    return residual
def model_func(parameter,x):
    return parameter[0] * np.exp(parameter[1] * x)
peaks_func=interpolate.interp1d(x1[peaks1],y12[peaks1],kind='linear',fill_value='extrapolate')
bottoms_func=interpolate.interp1d(x1[bottoms1],y12[bottoms1],kind='linear',fill_value='extrapolate')
#近似初期パラメータ
max_parameter0 = [ peaks_func(0), np.average(ln(y12[peaks1]/abs(peaks_func(0)))/x1[peaks1])]
min_parameter0 = [ bottoms_func(0), np.average(ln(abs(y12[bottoms1]/-abs(bottoms_func(0))))/x1[bottoms1])]
#fittingパラメータ
Nonlinear_peaks = leastsq(fit_exp, max_parameter0, args=(x1[peaks1], y12[peaks1]))[0]
Nonlinear_bottoms = leastsq(fit_exp, min_parameter0, args=(x1[bottoms1], y12[bottoms1]))[0]
xfunc=np.linspace(0,x1[x1.shape[0]-1],1000)
#グラフ
fig3,ax3 = plt.subplots()
ax3.plot(x1, y12, label="data")
ax3.plot(x1,peaks_func(x1), color='red', linestyle=":")
ax3.plot(x1,bottoms_func(x1), color='blue', linestyle=":")
ax3.plot(xfunc, model_func(Nonlinear_peaks, xfunc),color="orange",label="fitting-result")
ax3.plot(xfunc, model_func(Nonlinear_bottoms, xfunc),color="orange")
ax3.plot([0,x1[x1.shape[0]-1]],[0,0],color="black", linestyle=":")
ax3.scatter(x1[peaks1],y12[peaks1], color='red')
ax3.scatter(x1[bottoms1],y12[bottoms1], color='blue')
ax3.legend(loc = 'upper right')
ax3.set_xlim(0)
ax3.plot([0,x1[x1.shape[0]-1]],[model_func(Nonlinear_peaks, 0),model_func(Nonlinear_peaks, 0)],color="orange")
ax3.plot([0,x1[x1.shape[0]-1]],[model_func(Nonlinear_bottoms,0),model_func(Nonlinear_bottoms,0)],color="orange")
ax3.set_xlabel("delay time [ps]")
ax3.set_ylabel("Kerr signal [%]")
ax3.set_title(tittle+" envelope curve fitting")
plt.show()
exp_A=(abs(Nonlinear_peaks[0])+abs(Nonlinear_bottoms[0]))/2
exp_t=(Nonlinear_peaks[1]+Nonlinear_bottoms[1])/2
print("fitting-result [A: "+str(exp_A)+",1/τ:"+str(exp_t)+"[1/ps],"+exp_string(exp_A, exp_t)+"]")"""
    code5="""xfunc1=np.linspace(0,x1[x1.shape[0]-1],1000)
model_func1 = lambda A, k, Omega , delta , x : A * np.exp(k * x)*np.sin(Omega*x + delta)
def fit(p,x):
    return p[0] * np.exp(p[1] * x)*np.sin(p[2]*x+p[3])+p[4]* np.exp(p[5] * x)+p[6]
def fit_exp3(p, x, y):
    return y - fit(p,x)
fitting_parameter3=np.append([exp_A,exp_t,2*np.pi*exp_freq,0],fitting_func)
result=leastsq(fit_exp3, fitting_parameter3, args=(x1, y1))[0]
#上グラフ
fig5,ax5 = plt.subplots(figsize=[20,5])
ax5.plot(xfunc1,fit(result,xfunc1),label="LLG-Simulation",color="orange")
ax5.scatter(x1, y1, label="data")
ax5.set_xlabel("delay time [ps]")
ax5.set_ylabel("Kerr signal [%]")
ax5.set_xlim(0,x1[x1.shape[0]-1])
ax5.legend(loc = 'upper right')
ax5.set_title(tittle)
#下グラフ
fig6,ax6 = plt.subplots(figsize=[20,5])
ax6.scatter(x1, y1-middle_func(result[4],result[5],x1,result[6]), label="data")
ax6.plot(xfunc1,model_func1(result[0],result[1],result[2],result[3],xfunc1),label="LLG-Simulation",color="orange")
ax6.set_xlabel("delay time [ps]")
ax6.set_ylabel("Kerr signal [%]")
ax6.set_xlim(0,x1[x1.shape[0]-1])
ax6.legend(loc = 'upper right')
ax6.set_title(tittle)
plt.show()
print("Data=A*exp(-t/τ)*sin(ω*t+θ)+B*exp(-t/τ0)+C")
print("A:{}[%],-1/τ:{}[1/ps],ω:{}[rad/ps],θ:{}[rad],\\nB:{}[%],-1/τ0:{}[1/ps],C:{}[%]".format(abs(result[0]),result[1],result[2],result[3],result[4],result[5],result[6]))"""
    code6="""#fitting結果 保存用
#fig5.savefig("fitting-result.jpg",bbox_inches='tight',pad_inches=0.03)
#fig6.savefig("fitting-result2.jpg",bbox_inches='tight',pad_inches=0.03)"""
    code7="""import os,glob,pathlib
save_data=np.vstack([x1, y1-middle_func(result[4],result[5],x1,result[6])]).T
save_data=np.concatenate([np.array([["delay time [ps]","Kerr signal [%],,fittingresult,A[%],(-1/tau)[1/ps],omg[rad/ps],theta[rad]"]]),np.array(save_data,dtype=str)])
save_data[1][0]="{},{},,,{}".format(save_data[1][0],save_data[1][1],result[0])
save_data[1][1]="{},{},{}".format(result[1],result[2],result[3])
save_data[2][1]="{},,,B[%],(-1/tau0)[1/ps],C[%]".format(save_data[2][1])
save_data[3][0]="{},{},,".format(save_data[3][0],save_data[3][1])
save_data[3][1]="{},{},{}".format(result[4],result[5],result[6])
np.savetxt(tittle+'_result.csv', save_data, delimiter=',',fmt="%s")
np.savetxt(os.path.join(pathlib.Path(os.getcwd()).parent,"graph","data",tittle+'_result.csv'), save_data, delimiter=',',fmt="%s")"""

    nb["cells"] = [
        nbformat.v4.new_code_cell(code1),
        nbformat.v4.new_code_cell(code2),
        nbformat.v4.new_code_cell(code3),
        nbformat.v4.new_code_cell(code4),
        nbformat.v4.new_code_cell(code5),
        nbformat.v4.new_code_cell(code6),
        nbformat.v4.new_code_cell(code7)
    ]
    with open(os.path.join(*[FileDir,tittle+".ipynb"]), mode="w",encoding="utf-8") as f:
        nbformat.write(nb, f)
def createipynb1(FileDir):#グラフ用
    import os
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor
    nb = nbformat.v4.new_notebook()
    code1 = """#データ読み込み
import os,glob,csv,re
import numpy as np
current_path=os.path.join(os.getcwd(),"data")
list = np.array(os.listdir(current_path))#data内のファイル取得
list = list[np.char.endswith(list,".csv")]#listの内のcsvを抽出
num_list=np.zeros(list.shape[0],dtype="int")
for i in range(list.shape[0]):
    num_list[i]=int(re.sub(r"\D","",np.char.replace(list, '_result.csv', '')[i]))
sort=np.argsort(num_list)
list=list[sort]
num_list=num_list[sort]
name_list=np.char.replace(list, '_result.csv', '')#元の名前を抽出
graph_data=np.array(np.zeros(list.shape[0]),dtype="object")#ファイル数分の(object型なら中身は何でもOK)
for i in range(list.shape[0]):
    graph_data[i] = np.loadtxt(os.path.join(current_path,list[i]),delimiter=",",skiprows=1,usecols=(0,1))#[delay time(ps),Kerr signal(%)]
    with open(os.path.join(current_path,list[i]),'r') as f:
        result=csv.reader(f)
        result=[row for row in result]
        if i==0:
            fitt_result=np.array(np.hstack([result[1][4:8],result[3][4:7]]),dtype="float")#[A(%),1/τ(1/ps),omg(rad/ps),theta(rad),B(%),-1/tau0(1/ps),C(%)] array作成
        else:
            fitt_result=np.vstack([fitt_result,np.array(np.hstack([result[1][4:8],result[3][4:7]]),dtype="float")])#[A(%),1/τ(1/ps),omg(rad/ps),theta(rad),B(%),-1/tau0(1/ps),C(%)] array追加
fitt_result=np.vstack([num_list,fitt_result.T]).T #[name,A(%),1/τ(1/ps),omg(rad/ps),theta(rad),B(%),-1/tau0(1/ps),C(%)]"""
    code2 = """#入力ここから
#x軸最小=x_min[ps],x軸最大=x_max[ps]
x_min=0
x_max=200
#入力ここまで


import matplotlib.pyplot as plt
#グラフ設定
plt.rcParams['font.family'] = 'Times New Roman' 
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
x_func=np.linspace(x_min,x_max,10000)#理論式用Xプロット
def function(p,x):#理論式
    return p[1]*np.exp(p[2]*x)*np.sin(p[3]*x+p[4])
fig=plt.figure(figsize=(8,2*list.shape[0]))#グラフの大きさ(縦はグラフの数によって調整)
ax1=fig.add_subplot(1,1,1)#外枠の設定
ax1.set_xlabel("delay time (ps)", fontsize=30, weight='bold')#横軸の題名
ax1.set_ylabel("Kerr Signal", fontsize=30, labelpad=30, weight='bold')#縦軸の題名
ax1.tick_params(axis='x', labelsize=30)
ax1.tick_params(left=False,labelleft=False)
ax1.set_xticks(np.arange(x_min, x_max+1, 100,dtype="int"))
ax1.set_xticks(np.arange(x_min, x_max+1, 20,dtype="int"),minor="True")
ax=np.array(np.zeros(list.shape[0]),dtype="object")#中グラフ
for i in range(list.shape[0]):#データの分だけループ
    ax[i]=fig.add_subplot(list.shape[0],1,i+1)#データ用のグラフの設定
    ax[i].scatter((graph_data[i].T)[0],(graph_data[i].T)[1],color="red",zorder=3,s=100)#元データの散布図
    ax[i].plot(x_func,function(fitt_result[i],x_func),lw=4)#理論式のプロット
    ax[i].set_xlim(x_min,x_max)#x軸(最小値,最大値)
    ax[i].text(x_max*0.75,fitt_result[i][1]*0.5, name_list[i], fontsize=30)
    ax[i].axis("off")#メモリ線、枠線削除
plt.show()"""
    code3 = """fig.savefig("fitting_result.jpg",dpi=200,bbox_inches='tight',pad_inches=0.03)"""
    code4= """unit=name_list[0].replace(str(num_list[0]), "")#単位抽出部分
if unit=="deg":#単位判別 unit:fittingデータまとめの"data_result.csv"用の単位名,uint1:グラフ用の単位名
    unit="deg (°)"
    unit1="deg (°)"
if unit=="kOe":
    unit="Hext (kOe)"
    unit1="$H_{ext}$ (kOe)"
if unit=="mW":
    unit="pump power (mW)"
    unit1="pump power (mW)"
else:
    unit1=unit
fig2,ax2=plt.subplots(figsize=[6,5])#上_振幅グラフ
ax2.plot(fitt_result.T[0],np.abs(fitt_result.T[1]),'o', color='none', markersize=15, markeredgewidth=4, markeredgecolor='blue', alpha=0.8,zorder=3)#プロット
ax2.tick_params(axis='x', labelsize=20)#x軸の目盛りの文字の大きさ
ax2.tick_params(axis='y', labelsize=20)#y軸の目盛りの文字の大きさ
ax2.set_xlabel(unit1, fontsize=20)#x軸の題名
ax2.set_ylabel("amplitude (%)", fontsize=20, weight='bold')#y軸の題名
fig3,ax3=plt.subplots(figsize=[6,5])#下_角周波数グラフ
ax3.plot(fitt_result.T[0],fitt_result.T[3]*1000,'o', color='none', markersize=15, markeredgewidth=4, markeredgecolor='blue', alpha=0.8,zorder=3)#プロット
ax3.tick_params(axis='x', labelsize=20)#x軸の目盛りの文字の大きさ
ax3.tick_params(axis='y', labelsize=20)#y軸の目盛りの文字の大きさ
ax3.set_yticks(np.arange(0,max(fitt_result.T[3]*1000)*1.2+1,100))#y軸目盛線の位置 np.arange(最小値,最大値,刻み幅)
ax3.set_xlabel(unit1, fontsize=20)#x軸の題名
ax3.set_ylabel("ω (Grad/s)", fontsize=20, weight='bold')#y軸の題名
ax3.set_ylim(0,max(fitt_result.T[3]*1000)*1.2)
plt.show()
"""
    code5="""fig2.savefig("amplitude.jpg",dpi=200,bbox_inches='tight',pad_inches=0.03)#振幅上グラフ画像保存
#fig.savefig("omg.jpg",dpi=200,bbox_inches='tight',pad_inches=0.03)#角周波数下グラフ画像保存"""
    code6="""#csv保存
fitt_result_save=np.concatenate([np.array([[unit,"A[%]","(-1/tau)[1/ps]","omg[rad/ps]","theta[rad]","B[%]","(-1/tau0)[1/ps]","C[%]"]]),np.array(fitt_result,dtype=str)])#一番上の行の題名
np.savetxt('data_result.csv', fitt_result_save, delimiter=',',fmt="%s")#csv保存"""  
    nb["cells"] = [
        nbformat.v4.new_code_cell(code1),
        nbformat.v4.new_code_cell(code2),
        nbformat.v4.new_code_cell(code3),
        nbformat.v4.new_code_cell(code4),
        nbformat.v4.new_code_cell(code5),
        nbformat.v4.new_code_cell(code6)
    ]
    with open(os.path.join(*[FileDir,"graph.ipynb"]), mode="w",encoding="utf-8") as f:
        nbformat.write(nb, f)
"""
    try:
        with open(os.path.join(*[FileDir,tittle+".ipynb"]),encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        os.chdir(os.path.join(*[os.getcwd(),FileDir]))
        ep.preprocess(nb)
        os.chdir(current_path)
        with open(os.path.join(*[FileDir,tittle+".ipynb"]), 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
    except:
        pass
"""