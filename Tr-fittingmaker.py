from programs import createipynb
import os
import numpy as np
def isnumber(K,P):
	ko=0
	while ko<10:
		print(K)
		N=input(P)
		if N.isdecimal():
			N=int(N)
			return N
			break
		try:
			N=float(N)
			return N
			break
		except:
			try:
				O=eval(N)
				return N
				break
			except:
				print("Fill the number")
				print(K)
				N=input(P)
files = []
i = 0
while i < 3:
	print("Drag and drop data directory")
	path =input()
	files2=np.array(files)
	a=str(path[0])
	if a=="\'" or a=='\"':
		path=str(path[1:-2])
	del a
	while i < 3:
		if os.path.exists(path):
			break
		else:
			print("<error:The path does not exist.>")
			print("Drag and drop data's directory")
			path =input()
	try:
		f=open(path,'r')
		f.close()
		base, ext = os.path.splitext(path)
		if ext=='.txt' or ext=='':
			dir1=np.array(base.split("\\"))
			dir1=np.array(dir1[dir1.shape[0]-1].split("/"))
			dirshape=dir1.shape[0]
			DirTittle=dir1[dirshape-1]
			try:
				os.mkdir(DirTittle)
			except:
				pass
			break
	except:
		pass
	try:
		for f in os.listdir(path):
			if os.path.isfile(os.path.join(path, f)):
				base, ext = os.path.splitext(f)
				if ext=='.txt' or ext=='':
					files2=np.append(files2,f)
		if files2.shape[0]!=0:
			dir1=np.array(path.split("\\"))
			dir1=np.array(dir1[dir1.shape[0]-1].split("/"))
			dirshape=dir1.shape[0]
			DirTittle=dir1[dirshape-1]
			try:
				os.mkdir(DirTittle)
			except:
				pass
			break
	except:
		pass
while i<10:
	print("Fill A[V] value")
	A=input("オシロの標準資料磁化印加時最大最小電圧差:")
	K="Fill B[V] value"
	P="標準資料のロックインアンプ電圧値;"
	if A.isdecimal():
		A=int(A)
		B=isnumber(K,P)
		break
	try:
		A=float(A)
		B=isnumber(K,P)
		break
	except:
		try:
			O=eval(A)
			B=isnumber(K,P)
			break
		except:
			if len(A)==0:
				print("Did you measure?[y/n]")
				yn=input()
				if yn=="n":
					A=100
					B=1
					break
		print("Fill the number")
K="Fill C[V] value"
P="測定資料LIA電圧値:"
C=isnumber(K,P)
K="Fill D[V] value"
P="サンプルオシロ電圧値:"
D=isnumber(K,P)
K="Fill E[V] value"
P="サンプル磁場印加時の正電圧値:"
E=isnumber(K,P)
K="Fill F[V] value"
P="サンプル磁場印加時の負電圧値:"
F=isnumber(K,P)
if files2.shape[0]!=0:
	path1=os.path.join(path, files2[i])
else:
	path1=path
base, ext = os.path.splitext(path1)
dir1=np.array(base.split("\\"))
dir1=np.array(dir1[dir1.shape[0]-1].split("/"))
dirshape=dir1.shape[0]
Tittle=dir1[dirshape-1]
filename=Tittle+ext
FileDir=os.path.join(*[DirTittle,Tittle])
try:
	os.mkdir(FileDir)
except:
	pass
txt = open(os.path.join(*[FileDir,filename]),'w')
txt.write(open(path1,'r').read())
txt.close()
createipynb.createipynb(Tittle,filename,FileDir,A,B,C,D,E,F)
i+=1
while i < files2.shape[0]:
	path1=os.path.join(path, files2[i])
	base, ext = os.path.splitext(path1)
	dir1=np.array(base.split("\\"))
	dir1=np.array(dir1[dir1.shape[0]-1].split("/"))
	asds=dir1.shape[0]
	Tittle=dir1[asds-1]
	filename=Tittle+ext
	FileDir=os.path.join(*[DirTittle,Tittle])
	try:
		os.mkdir(FileDir)
	except:
		pass
	txt = open(os.path.join(*[FileDir,filename]),'w')
	txt.write(open(path1,'r').read())
	txt.close()
	createipynb.createipynb(Tittle,filename,FileDir,A,B,C,D,E,F)
	i+=1
os.mkdir(os.path.join(DirTittle,"graph"))
os.mkdir(os.path.join(DirTittle,"graph","data"))
createipynb.createipynb1(os.path.join(DirTittle,"graph"))