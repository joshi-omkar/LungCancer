import tkinter as tk
from tkinter import ttk, LEFT, END
from PIL import Image , ImageTk 
from tkinter.filedialog import askopenfilename
import cv2
import numpy as np
import time
import sqlite3
#import tfModel_test as tf_test
global fn
fn=""
##############################################+=============================================================
root = tk.Tk()
root.configure(background="seashell2")
#root.geometry("1300x700")


w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))
root.title("Lung cancer disease Detection using ML")


#430
#++++++++++++++++++++++++++++++++++++++++++++
#####For background Image
image2 =Image.open('c3.jpg')
image2 =image2.resize((1220,800), Image.ANTIALIAS)

background_image=ImageTk.PhotoImage(image2)
background_label = tk.Label(root, image=background_image,bd=5)

background_label.image = background_image

background_label.place(x=0, y=0) #, relwidth=1, relheight=1)
#


#frame_display = tk.LabelFrame(root, text=" --Display-- ", width=900, height=250, bd=5, font=('times', 14, ' bold '),bg="lightblue4")
#frame_display.grid(row=0, column=0, sticky='nw')
#frame_display.place(x=300, y=100)

#frame_display1 = tk.LabelFrame(root, text=" --Result-- ", width=900, height=200, bd=5, font=('times', 14, ' bold '),bg="lightblue4")
#frame_display1.grid(row=0, column=0, sticky='nw')
#frame_display1.place(x=300, y=430)

#frame_display2 = tk.LabelFrame(root, text=" --Calaries-- ", width=900, height=50, bd=5, font=('times', 14, ' bold '),bg="lightblue4")
#frame_display2.grid(row=0, column=0, sticky='nw')
#frame_display2.place(x=300, y=380)

frame_alpr = tk.LabelFrame(root, text="  ", width=820, height=900, bd=5, font=('times', 14, ' bold '),bg="#271983")
frame_alpr.grid(row=0, column=0)
frame_alpr.place(x=900, y=0)

lbl = tk.Label(root, text="Lung cancer disease Detection using ML", font=('Elephant', 35,' bold '),width=50,bg="brown",fg="white")
lbl.place(x=0, y=0)

lbl = tk.Label(frame_alpr, text='Cancer is a part of our life ', font=('Lucida Calligraphy', 15,' bold '),bg="#271983",fg="white")
lbl.place(x=170, y=100)

lbl = tk.Label(frame_alpr, text="but it's not our whole life", font=('Lucida Calligraphy', 15,' bold '),bg="#271983",fg="white")
lbl.place(x=210, y=140)




def login():

    from subprocess import call
    call(["python", "login.py"])  

def register():

    from subprocess import call
    call(["python", "registration.py"])  
   
def window():
    root.destroy()

button1 = tk.Button(frame_alpr, text=" SIGN UP ",command=register,width=15, height=1, font=('times', 15, ' bold '),bg="#3BB9FF",fg="white")
button1.place(x=200, y=350)

button2 = tk.Button(frame_alpr, text="LOGIN",command=login,width=15, height=1, font=('times', 15, ' bold '),bg="#3BB9FF",fg="white")
button2.place(x=200, y=450)




root.mainloop()