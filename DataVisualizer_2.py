
from tkinter import filedialog,messagebox
from tkinter import *
import ttkbootstrap as tb
import numpy as np
import pandas as pd
import os
import h5py
from pathlib import Path
from PIL import Image, ImageTk
from tkintertable import TableCanvas, TableModel
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import sys


# Get the current working directory
current_directory = os.path.dirname(os.path.abspath(__file__))
# !!!! YOU SHOULD WRITE DOWN THE DATA FILE DIRECTORY BELOW
datafile_directory = r"data_set" #the folder which includes h5 files of Turbofan Data

X_s_var=[]
W_var=[]
unit_list1 = [];
list_sensor=[]
list_cycle=[]
list_unit=[]
df_cycle=pd.DataFrame()
df_sensor=pd.DataFrame()
df_unit=pd.DataFrame()
df_Xs = pd.DataFrame()
df_A = pd.DataFrame(columns=["unit","cycle"])
#INSTALLATION OF BASE
root = tb.Window(themename="solar")
#root = Tk()
root.title("CMAPSS Dataset Visualizer");
root.iconbitmap(os.path.join(current_directory, "Bootstrap-Bootstrap-Bootstrap-airplane-engines.ico"))
#root.geometry("1800x1200")
root.geometry("1600x1024")
const_x = 1600/1800
const_y = 1024/1200

#NASA IMAGE
img_lbl1_x, img_lbl1_y = int(50*const_x) , int(35*const_y)  
img_nasa1 = Image.open(os.path.join(current_directory, "NASA_Worm_logo.svg.png"))
#img_nasa1 = img_nasa1.resize((245, 205), Image.LANCZOS)  # Resize the image
img_nasa1 = img_nasa1.resize((int(256*const_x), int(70*const_y)), Image.LANCZOS)  # Resize the image
img_nasa1 = ImageTk.PhotoImage(img_nasa1)
img_lbl1 = tb.Label(root,image=img_nasa1)
img_lbl1.place(x = img_lbl1_x, y = img_lbl1_y )

img_lbl2 = tb.Label(root,text="CMAPSS Dataset Visualizer",font=("Times",24,"bold italic underline"))
img_lbl2.place(x = img_lbl1_x + int(270*const_x), y = img_lbl1_y + int(30*const_y))

#OPEN FILE DIALOG

def open_file1():
    global X_s_var,W_var,W,X_s,X_v,T,Y,A,df_A,df_Xs,df_W,df_all
    root.filename = filedialog.askopenfilename(initialdir=datafile_directory,title="Select the Dataset",filetypes=(("h5 files","*.h5"),("all files","*.*")))
    file_lbl_txt = str(Path(root.filename).stem)
    with h5py.File(root.filename, 'r') as hdf:
            # Development set
            W_dev = np.array(hdf.get('W_dev'))             # W
            X_s_dev = np.array(hdf.get('X_s_dev'))         # X_s
            X_v_dev = np.array(hdf.get('X_v_dev'))         # X_v
            T_dev = np.array(hdf.get('T_dev'))             # T
            Y_dev = np.array(hdf.get('Y_dev'))             # RUL  
            A_dev = np.array(hdf.get('A_dev'))             # Auxiliary

            # Test set
            W_test = np.array(hdf.get('W_test'))           # W
            X_s_test = np.array(hdf.get('X_s_test'))       # X_s
            X_v_test = np.array(hdf.get('X_v_test'))       # X_v
            T_test = np.array(hdf.get('T_test'))           # T
            Y_test = np.array(hdf.get('Y_test'))           # RUL  
            A_test = np.array(hdf.get('A_test'))           # Auxiliary
            
            # Varnams
            W_var = np.array(hdf.get('W_var'))
            X_s_var = np.array(hdf.get('X_s_var'))  
            X_v_var = np.array(hdf.get('X_v_var')) 
            T_var = np.array(hdf.get('T_var'))
            A_var = np.array(hdf.get('A_var'))
            
            # from np.array to list dtype U4/U5
            W_var = list(np.array(W_var, dtype='U20'))
            X_s_var = list(np.array(X_s_var, dtype='U20'))  
            X_v_var = list(np.array(X_v_var, dtype='U20')) 
            T_var = list(np.array(T_var, dtype='U20'))
            A_var = list(np.array(A_var, dtype='U20'))
                              
    W = np.concatenate((W_dev, W_test), axis=0)  
    X_s = np.concatenate((X_s_dev, X_s_test), axis=0)
    X_v = np.concatenate((X_v_dev, X_v_test), axis=0)
    T = np.concatenate((T_dev, T_test), axis=0)
    Y = np.concatenate((Y_dev, Y_test), axis=0) 
    A = np.concatenate((A_dev, A_test), axis=0)
    df_A = pd.DataFrame(data=A, columns=A_var)
    df_Xs = pd.DataFrame(data=X_s, columns=X_s_var)
    df_W = pd.DataFrame(data=W, columns=W_var)
    df_all = pd.concat([df_A, df_Xs, df_W], axis=1)
    file_lbl.config(text=f"Dataset {file_lbl_txt} has been imported.")
    
    for item in X_s_var:
        listbox_var.insert(END, item);
    
file_btn_x, file_btn_y = 50,150   
file_btn = tb.Button(root,text="Open File",command=open_file1,bootstyle="success");
file_btn.place(x=file_btn_x,y=file_btn_y)

file_lbl_txt = "<== Select a dataset to analyze";
file_lbl = tb.Label(root,text=file_lbl_txt)
file_lbl.place(x=file_btn_x + 100 ,y=file_btn_y+5)


#FEATURE SELECTOR

def save_feat(list_cycle,list_unit):
    global df_cycle,df_sensor,df_unit,list_sensor,list_opvar,df_opvar,df_all,df_filtered
    df_sensor = df_Xs[list_sensor].copy()
    df_cycle = df_A[df_A["cycle"].isin(list_cycle)]["cycle"].copy()
    df_unit = df_A[df_A["unit"].isin(list_unit)]["unit"].copy()
    df_opvar =df_W[list_opvar].copy()
    # Find common index values iteratively
    common_indices = df_sensor.index
    for df in [df_cycle, df_unit]:
        common_indices = common_indices.intersection(df.index)
        
    df_filtered = df_all.loc[common_indices]
    df_filtered = df_filtered.reset_index()

    
    
    
def save_selected_sensors(items1):
    global list_sensor
    list_sensor = [items1.get(i) for i in items1.curselection()]
    print(list_sensor)
     
def save_selected_unit(items1):
    global list_unit
    list_unit = [items1.get(i) for i in items1.curselection()]
    list_unit = [float(x) for x in list_unit]
    print(list_unit)
    
def save_selected_opvar(items1):
    global list_opvar
    list_opvar = [items1.get(i) for i in items1.curselection()]
    print(list_opvar)

def save_selected_cycle(items1,items2):
    global list_cycle
    list_cycle = [float(x) for x in range(items1.amountusedvar.get(),items2.amountusedvar.get())]
    print(list_cycle)

def open_window():
    global sel_unit,sel_sensor,sel_cyc,list_cycle,list_unit,list_sensor,unit_list1
    
    f_sel = tb.Toplevel()
    f_sel.title("Feature Selection")
    f_sel.iconbitmap(os.path.join(current_directory, "Bootstrap-Bootstrap-Bootstrap-airplane-engines.ico"))
    f_sel.geometry("800x800")

    
    #######################################################
    #SELECTED SENSORS

    #Sensor Listbox
    sensor_list1 = X_s_var;


    sensor_listbox1_x, sensor_listbox1_y = 50, 50
    label_sensor = tb.Label(f_sel,text="SENSORS",font=("Helvetica",20))
    label_sensor.place(x=sensor_listbox1_x,y=sensor_listbox1_y)

    sensor_listbox1 = Listbox(f_sel,selectmode=MULTIPLE)
    sensor_listbox1.place(x= sensor_listbox1_x, y= sensor_listbox1_y+50)

    for item in sensor_list1:
        sensor_listbox1.insert(END, item);
     
    save_sen_btn1 = tb.Button(f_sel,text = "Save Sensors",command=lambda :save_selected_sensors(sensor_listbox1))
    save_sen_btn1.place(x = sensor_listbox1_x, y = sensor_listbox1_y + 250)    
    #######################################################
    #Operational Variables Listbox
    opvar_list1 = W_var;


    opvar_listbox1_x, opvar_listbox1_y = sensor_listbox1_x + 250, sensor_listbox1_y
    label_opvar = Label(f_sel,text="OPERATIONAL VARIABLES",font=("Helvetica",20))
    label_opvar.place(x=opvar_listbox1_x,y=opvar_listbox1_y)

    opvar_listbox1 = Listbox(f_sel,selectmode=MULTIPLE)
    opvar_listbox1.place(x= opvar_listbox1_x + 125, y= opvar_listbox1_y + 50)

    for item in opvar_list1:
        opvar_listbox1.insert(END, item);
        
    save_op_btn1 = tb.Button(f_sel,text = "Save Op Var",command=lambda :save_selected_opvar(opvar_listbox1))
    save_op_btn1.place(x = opvar_listbox1_x + 150, y =  sensor_listbox1_y + 250)   
    #######################################################
    #Units Listbox
    if len(W_var) != 0:
        unit_list1 = df_A["unit"].unique();


    unit_listbox1_x, unit_listbox1_y = sensor_listbox1_x, sensor_listbox1_y + 350
    label_unit = Label(f_sel,text="UNITS",font=("Helvetica",20))
    label_unit.place(x=unit_listbox1_x,y=unit_listbox1_y)

    unit_listbox1 = Listbox(f_sel,selectmode=MULTIPLE)
    unit_listbox1.place(x= unit_listbox1_x, y= unit_listbox1_y+50)

    for item in unit_list1:
        unit_listbox1.insert(END, item);
    

    
    save_unit_btn1 = tb.Button(f_sel,text = "Save Units",command=lambda :save_selected_unit(unit_listbox1))
    save_unit_btn1.place(x = unit_listbox1_x, y = unit_listbox1_y +250) 
    #######################################################
    #CYCLE RANGE SELECTOR


    #Min Cycle and Max Cycle Meters
    cycle_label_x, cycle_label_y = 400,400
    label_cycle = tb.Label(f_sel,text="CYCLE",font=("Helvetica",20))
    label_cycle.place(x=cycle_label_x,y=cycle_label_y)

    cycle1_x ,cycle1_y = cycle_label_x - 50, cycle_label_y + 80

    min_cycle_1 = tb.Meter(f_sel, bootstyle="danger",
                           subtext="Min",
                           subtextfont="-size 8",
                           meterthickness=3,
                           textfont="-size 12",
                           interactive=True,
                           metertype="semi",
                           metersize=120,
                           amounttotal=95
                           )
    min_cycle_1.place(x=cycle1_x ,y= cycle1_y)


    cycle2_x ,cycle2_y = cycle_label_x + 50, cycle_label_y + 80

    min_cycle_2 = tb.Meter(f_sel, bootstyle="danger",
                           subtext="Max",
                           subtextfont="-size 8",
                           meterthickness=3,
                           textfont="-size 12",
                           interactive=True,
                           metertype="semi",
                           metersize=120,
                           amounttotal=95
                           )
    min_cycle_2.place(x=cycle2_x + 20 ,y= cycle2_y)
    
    

    #######################################################
    save_btn1 = tb.Button(f_sel,text= "Save All and Quit",command=lambda :[save_selected_cycle(min_cycle_1,min_cycle_2),save_feat(list_cycle,list_unit),f_sel.destroy()],bootstyle="danger")
    save_btn1.place(x = 650, y = 750)
    
    


fsel_btn_x, fsel_btn_y = file_btn_x,file_btn_y+50   
fsel_btn = tb.Button(root,text="Select Features",command=open_window);
fsel_btn.place(x=fsel_btn_x,y=fsel_btn_y)


#######################################################
# Basic Analysis Tool
bstat_btn_x, bstat_btn_y = fsel_btn_x, fsel_btn_y +50


def make_Stat():
    global df_unit, df_cycle, df_sensor

    # Read and process the data from the loaded dataset

    # Create the dataframes for unit, cycle, and sensor statistics
    df_unit_Stat = df_unit.describe()
    df_cycle_Stat = df_cycle.describe()
    df_sensor_Stat = df_sensor.describe()
    df_opvar_Stat = df_opvar.describe()

    # Concatenate the dataframes to create a single dataframe for all statistics
    df_Stats = pd.concat([df_unit_Stat, df_cycle_Stat, df_sensor_Stat,df_opvar_Stat], axis=1)

    # Create a Frame to hold the TreeView and scrollbar
    frame = Frame(root)
    frame.pack(expand=FALSE)
    frame.place(x=bstat_btn_x, y=bstat_btn_y+75,width=350,height=250)
    
    # Create a vertical scrollbar
    v_scrollbar = tb.Scrollbar(frame,orient='vertical')
    v_scrollbar.pack(side=RIGHT,fill=BOTH)
    #tree.configure(yscrollcommand=v_scrollbar.set)

    # Create a horizontal scrollbar
    h_scrollbar = tb.Scrollbar(frame,orient='horizontal')
    h_scrollbar.pack(side=BOTTOM,fill=X)
    #tree.configure(xscrollcommand=h_scrollbar.set)


    # Create the TreeView
    tree = tb.Treeview(frame,yscrollcommand=v_scrollbar.set,xscrollcommand=h_scrollbar.set)
    tree.pack(expand=FALSE)
    
    h_scrollbar.config(command=tree.xview)
    v_scrollbar.config(command=tree.yview)
    # Add columns to the TreeView
    columns = ['Index'] + list(df_Stats.columns)
    tree["columns"] = columns
    tree["show"] = "headings"

    # Configure column headings
    for col in columns:
        tree.column(col, anchor="center", width=80)
        tree.heading(col, text=col)

    # Insert data into the TreeView
    for i, index_name in enumerate(df_Stats.index):
        values = [index_name] + list(df_Stats.iloc[i])
        tree.insert("", i, values=values)
    




    
bstat_btn = tb.Button(text="Stats",bootstyle="danger", command= make_Stat);
bstat_btn.place(x = bstat_btn_x,y = bstat_btn_y)


# Create a new frame for plotting
plot_frame_x, plot_frame_y = file_btn_x + int(500*const_x),file_btn_y
plot_frame = Frame(root)
plot_frame.place(x=plot_frame_x, y=plot_frame_y, width=int(1200*const_x), height=int(800*const_y))

# Function to plot sensor readings
def plot_sensor_readings():
    global df_sensor, list_sensor, df_filtered
    
        # Clear existing plots from the frame
    for widget in plot_frame.winfo_children():
        widget.destroy()

    # Number of subplots
    num_subplots = len(list_sensor)

    # Determine the number of rows and columns
    num_rows = (num_subplots + 2) // 3  # 3 plots per row
    num_cols = min(num_subplots, 3)

    # Create subplots
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 5*num_rows))

    # Flatten the axes array if there is more than one row
    if num_rows > 1:
        axes = axes.flatten()

    # Plot each column as a subplot
    for i, column in enumerate(list_sensor):
        ax = axes[i]
        df_filtered[column].plot(ax=ax, legend=True)
        ax.set_title(column)
        ax.set_xlabel("Time")
        ax.set_ylabel("Sensor Values")
        ax.legend()

    # Adjust layout
    plt.tight_layout()

    # Add overall plot title
    fig.suptitle("Sensor Readings")

    # Embed the plot in the Tkinter window
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

# Create a button to trigger the sensor readings plot
plot_sensor_btn = tb.Button(root, text="Plot Sensor Readings", command=plot_sensor_readings, bootstyle="primary")
plot_sensor_btn.place(x=plot_frame_x, y=plot_frame_y + int(900*const_y))


# Function to plot sensor readings as scatter plots
def plot_sensor_scatter():
    global df_sensor, list_sensor, df_filtered
    
        # Clear existing plots from the frame
    for widget in plot_frame.winfo_children():
        widget.destroy()

    # Number of subplots
    num_subplots = len(list_sensor)

    # Determine the number of rows and columns
    num_rows = (num_subplots + 2) // 3  # 3 plots per row
    num_cols = min(num_subplots, 3)

    # Create subplots
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 5*num_rows))

    # Flatten the axes array if there is more than one row
    if num_rows > 1:
        axes = axes.flatten()

    # Plot each column as a scatter plot
    for i, column in enumerate(list_sensor):
        ax = axes[i]
        ax.scatter(df_filtered.index,df_filtered[column], label=column)
        ax.set_title(column)
        ax.set_xlabel("Time")
        ax.set_ylabel("Sensor Values")
        ax.legend()

    # Adjust layout
    plt.tight_layout()

    # Add overall plot title
    fig.suptitle("Sensor Readings (Scatter Plots)")

    # Embed the plot in the Tkinter window
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

# Create a button to trigger the scatter plot of sensor readings
plot_sensor_scatter_btn = tb.Button(root, text="Plot Sensor Readings (Scatter)", command=plot_sensor_scatter, bootstyle="primary")
plot_sensor_scatter_btn.place(x=plot_frame_x + int(150*const_x), y=plot_frame_y + int(900*const_y))


# Function to plot sensor readings as box plots
def plot_sensor_box():
    global df_sensor, list_sensor, df_filtered

    # Clear existing plots from the frame
    for widget in plot_frame.winfo_children():
        widget.destroy()

    # Number of subplots
    num_subplots = len(list_sensor)

    # Determine the number of rows and columns
    num_rows = (num_subplots + 2) // 3  # 3 plots per row
    num_cols = min(num_subplots, 3)

    # Create subplots
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 5*num_rows))

    # Flatten the axes array if there is more than one row
    if num_rows > 1:
        axes = axes.flatten()

    # Plot each column as a box plot
    for i, column in enumerate(list_sensor):
        ax = axes[i]
        ax.boxplot(df_filtered[column])
        ax.set_title(column)
        ax.set_xlabel("Sensor")
        ax.set_ylabel("Sensor Values")

    # Adjust layout
    plt.tight_layout()

    # Add overall plot title
    fig.suptitle("Sensor Readings (Box Plots)")

    # Embed the plot in the Tkinter window
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
    
# Create a button to trigger the box plot of sensor readings
plot_sensor_box_btn = tb.Button(root, text="Plot Sensor Readings (Box)", command=plot_sensor_box, bootstyle="primary")
plot_sensor_box_btn.place(x=plot_frame_x + int(350*const_x), y=plot_frame_y + int(900*const_y))

# Function to plot operational variables readings as scatter plots
def plot_opvar_scatter():
    global list_opvar, df_filtered
    
        # Clear existing plots from the frame
    for widget in plot_frame.winfo_children():
        widget.destroy()

    # Number of subplots
    num_subplots = len(list_opvar)

    # Determine the number of rows and columns
    num_rows = (num_subplots + 1) // 2  # 2 plots per row
    num_cols = min(num_subplots, 2)

    # Create subplots
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 5*num_rows))

    # Flatten the axes array if there is more than one row
    if num_rows > 1:
        axes = axes.flatten()

    # Plot each column as a scatter plot
    for i, column in enumerate(list_opvar):
        ax = axes[i]
        ax.scatter(df_filtered.index,df_filtered[column], label=column)
        ax.set_title(column)
        ax.set_xlabel("Time")
        ax.legend()

    # Adjust layout
    plt.tight_layout()

    # Add overall plot title
    fig.suptitle("Operational Variables (Scatter Plots)")

    # Embed the plot in the Tkinter window
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

# Create a button to trigger the scatter plot of operational variables
plot_opvar_scatter_btn = tb.Button(root, text="Plot OpVar Readings (Scatter)", command=plot_opvar_scatter, bootstyle="primary")
plot_opvar_scatter_btn.place(x=plot_frame_x, y=plot_frame_y + int(975*const_y))

# Function to plot operational settings readings as box plots
def plot_opvar_box():
    global list_opvar, df_filtered
    
        # Clear existing plots from the frame
    for widget in plot_frame.winfo_children():
        widget.destroy()

    # Number of subplots
    num_subplots = len(list_opvar)

    # Determine the number of rows and columns
    num_rows = (num_subplots + 1) // 2  # 2 plots per row
    num_cols = min(num_subplots, 2)

    # Create subplots
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 5*num_rows))

    # Flatten the axes array if there is more than one row
    if num_rows > 1:
        axes = axes.flatten()

    # Plot each column as a scatter plot
    for i, column in enumerate(list_opvar):
        ax = axes[i]
        ax.boxplot(df_filtered[column])
        ax.set_title(column)
        ax.set_xlabel("Time")
        ax.legend()

    # Adjust layout
    plt.tight_layout()

    # Add overall plot title
    fig.suptitle("Operational Variables (Box)")

    # Embed the plot in the Tkinter window
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

# Create a button to trigger the scatter plot of operational variables
plot_opvar_box_btn = tb.Button(root, text="Plot OpVar Readings (Box)", command=plot_opvar_box, bootstyle="primary")
plot_opvar_box_btn.place(x=plot_frame_x+int(200*const_x), y=plot_frame_y + int(975*const_y))


# Function to plot heatmap for sensor correlations
def plot_sensor_corr():
    global list_sensor, df_sensor
    
    # Clear existing plots from the frame
    for widget in plot_frame.winfo_children():
        widget.destroy()

    # Calculate correlation matrix
    corr = df_sensor.corr()

    # Create a heatmap using Seaborn
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)

    # Add overall plot title
    ax.set_title("Sensor Correlations")

    # Embed the plot in the Tkinter window
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
    
    

# Create a button to trigger the correlation heatmap
plot_sensor_corr_btn = tb.Button(root, text="Sensor Correlations", command=plot_sensor_corr, bootstyle="primary")
plot_sensor_corr_btn.place(x=plot_frame_x+int(375*const_x), y=plot_frame_y + int(975*const_y))

# Function to plot heatmap for sensor correlations
def plot_opvar_corr():
    global list_opvar, df_opvar
    
    # Clear existing plots from the frame
    for widget in plot_frame.winfo_children():
        widget.destroy()

    # Calculate correlation matrix
    corr = df_opvar.corr()

    # Create a heatmap using Seaborn
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)

    # Add overall plot title
    ax.set_title("Operational Variables Correlations")

    # Embed the plot in the Tkinter window
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
    
    

# Create a button to trigger the correlation heatmap
plot_opvar_corr_btn = tb.Button(root, text="Op Var Correlations", command=plot_opvar_corr, bootstyle="primary")
plot_opvar_corr_btn.place(x=plot_frame_x+ int(525*const_x), y=plot_frame_y + int(975*const_y))


def plot_vartotime(entry_unit,entry_cyc,var_list):
    global df_A,df_Xs
    
    # Clear existing plots from the frame
    for widget in plot_frame.winfo_children():
        widget.destroy()
    
    try:    
        df_X_s_u_c = df_Xs.loc[(df_A.unit == float(entry_unit.get())) & (df_A.cycle == float(entry_cyc.get()))]
        df_X_s_u_c.reset_index(inplace=True, drop=True)
        sel_var = var_list.get( var_list.curselection())
        fig, ax = plt.subplots(figsize=(10, 8))
        df_X_s_u_c[sel_var].plot(ax=ax, legend=True)
        ax.set_title(sel_var)
        ax.set_xlabel("Time")
        ax.set_ylabel(sel_var)
        ax.legend()
        
        # Embed the plot in the Tkinter window
        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
    except ValueError:
        print("error",ValueError)
        

    

    

unit_entry,cyc_entry = tb.Entry(),tb.Entry()
listbox_var = Listbox()
plot_vartotime_btn = tb.Button(root,text="Time Plot",command=lambda: plot_vartotime(unit_entry,cyc_entry,listbox_var))
plot_vartotime_btn.place(x = bstat_btn_x , y = bstat_btn_y + int(800*const_y))
unit_entry_lbl = tb.Label(text="UNIT                CYCLE",font=("Times",14,"bold"))
unit_entry_lbl.place(x = bstat_btn_x , y = bstat_btn_y + int(500*const_y))
unit_entry.place(x = bstat_btn_x , y = bstat_btn_y + int(550*const_y))
cyc_entry.place(x = bstat_btn_x + int(150*const_x) , y = bstat_btn_y + int(550*const_y))
listbox_var.place(x = bstat_btn_x , y = bstat_btn_y + int(600*const_y))


def on_closing():
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        root.destroy()
        sys.exit()
# Bind the closing event to the on_closing function
root.protocol("WM_DELETE_WINDOW", on_closing)

root.mainloop()