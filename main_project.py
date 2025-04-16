import tkinter as tk
from tkinter import ttk
from tkinter import *
from tkinter import messagebox
from PIL import Image, ImageTk
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, precision_score
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.tree import DecisionTreeClassifier

# Copywrite 22k-4018, 22k-4084, 22k-8720
# Copywrite Ahmed Khalid, Adil Rizwan, Ayan Khan

#Global Varibles
to_invest = ""
model_selected = ""
other_currencies_df = pd.DataFrame



def show_main_page():
    root = Tk()
    root.title('Main Page') 
    root.geometry('800x400')

    # Defining Colors
    bg_color = 'LightSkyBlue'
    btn_bg_color = 'SteelBlue'
    btn_fg_color = 'White'
    btn_font = ('Arial', 14)
    
    Label(root, text='Crypto Investment Advisor', font=('Noto Sans CJK TC', 20, 'bold'), bg=bg_color, fg='White').pack(side=TOP, fill=X)

    middle_frame = Frame(root, bg=bg_color)
    middle_frame.pack(expand=True, fill=BOTH)

    btn_investment = Button(middle_frame, text='My Investment', font=btn_font, bg=btn_bg_color, fg=btn_fg_color, width=20, command=open_investment_page)
    btn_investment.grid(row=0, column=0, padx=10, pady=10)

    btn_predict_best = Button(middle_frame, text='Predict Best Currency', font=btn_font, bg=btn_bg_color, fg=btn_fg_color, width=20, command=open_predict_best_currency_page)
    btn_predict_best.grid(row=1, column=0, padx=10, pady=10)

    btn_predict_worst = Button(middle_frame, text='Predict Worst Currency', font=btn_font, bg=btn_bg_color, fg=btn_fg_color, width=20, command=open_predict_worst_currency_page)
    btn_predict_worst.grid(row=2, column=0, padx=10, pady=10)

    middle_frame.grid_rowconfigure((0, 1, 2), weight=1)
    middle_frame.grid_columnconfigure(0, weight=1)

    root.mainloop()


def open_investment_page():
    def proceed():
        global to_invest
        selected_option = var.get()
        if selected_option == "":
            messagebox.showerror("Error", "Please select an option.")
            investment_window.destroy()
        else:
            messagebox.showinfo("Proceed", f"Selected option: {selected_option}.")
            to_invest = selected_option
            investment_window.destroy()
            choose_model()

    investment_window = Toplevel()
    investment_window.title("Investment Options")
    investment_window.geometry("800x400")

    Label(investment_window, text="Select one currency to invest in:", font=("Arial", 12)).pack(pady=10)

    var = StringVar()

    def option_clicked(option):
        current_option = var.get()

        var.set(option)

        option_buttons[option].config(relief=SUNKEN)
        
        if current_option != "":
            option_buttons[current_option].config(relief=RAISED)
            to_invest = var.get()
            print(to_invest)

    options = ["xrp", "tezos", "binance-coin", "eos", "tether", "bitcoin",
               "stellar", "bitcoin-cash", "bitcoin-sv", "litecoin", "ethereum", "cardano"]
    
    option_frame = Frame(investment_window)
    option_frame.pack(pady=10)

    var.set("") 

    option_buttons = {}  

    row_num = 0
    col_num = 0

    for option in options:
        button = Button(option_frame, text=option, width=20, height=2, command=lambda opt=option: option_clicked(opt))
        button.grid(row=row_num, column=col_num, padx=10, pady=5)
        option_buttons[option] = button

        col_num += 1
        if col_num == 3:
            col_num = 0
            row_num += 1

    proceed_button = Button(investment_window, text="Proceed", width=20, command=proceed)
    proceed_button.pack(pady=10)

def open_predict_best_currency_page():
    def proceed():
        global model_selected
        selected_option = var.get()
        if selected_option == "":
            messagebox.showerror("Error", "Please select an option.")
            best_c_options_window.destroy()
        else:
            model_selected = selected_option
            best_c_options_window.destroy()
            check_selection()

    best_c_options_window = Toplevel()
    best_c_options_window.title("Predict Best Currency")
    best_c_options_window.geometry("800x400")

    Label(best_c_options_window, text="Select any Model that you want use in training the data:", font=("Arial", 12)).pack(pady=10)

    var = StringVar()

    def option_clicked(option):
        current_option = var.get()

        var.set(option)
        
        option_buttons[option].config(relief=SUNKEN)

        if current_option != "":
            option_buttons[current_option].config(relief=RAISED)

    options = ["Logistic Regression", "Random Forest Classifier", "Decision Tree Classifier", "Linear Regression"]
    
    option_frame = Frame(best_c_options_window)
    option_frame.pack(pady=10)

    var.set("")  

    option_buttons = {}  

    row_num = 0
    col_num = 0

    for option in options:
        button = Button(option_frame, text=option, width=20, height=2, command=lambda opt=option: option_clicked(opt))
        button.grid(row=row_num, column=col_num, padx=10, pady=5)
        option_buttons[option] = button

        col_num += 1
        if col_num == 2:
            col_num = 0
            row_num += 1

    proceed_button = Button(best_c_options_window, text="Proceed", width=20, command=proceed)
    proceed_button.pack(pady=10)

def open_predict_worst_currency_page():
    def proceed():
        global model_selected
        
        selected_option = var.get()
        if selected_option == "":
            messagebox.showerror("Error", "Please select an option.")
            worst_c_options_window.destroy()
        else:
            model_selected = selected_option
            worst_c_options_window.destroy()
            check_selection2()

    worst_c_options_window = Toplevel()
    worst_c_options_window.title("Predict Worst Currency")
    worst_c_options_window.geometry("800x400")

    Label(worst_c_options_window, text="Select any Model that you want use in training the data:", font=("Arial", 12)).pack(pady=10)

    var = StringVar()

    def option_clicked(option):
        current_option = var.get()

        var.set(option)

        option_buttons[option].config(relief=SUNKEN)

        if current_option != "":
            option_buttons[current_option].config(relief=RAISED)

    options = ["Logistic Regression", "Random Forest Classifier", "Decision Tree Classifier", "Linear Regression"]
    
    option_frame = Frame(worst_c_options_window)
    option_frame.pack(pady=10)

    var.set("")  

    option_buttons = {}  

    row_num = 0
    col_num = 0

    for option in options:
        button = Button(option_frame, text=option, width=20, height=2, command=lambda opt=option: option_clicked(opt))
        button.grid(row=row_num, column=col_num, padx=10, pady=5)
        option_buttons[option] = button

        col_num += 1
        if col_num == 2:
            col_num = 0
            row_num += 1

    proceed_button = Button(worst_c_options_window, text="Proceed", width=20, command=proceed)
    proceed_button.pack(pady=10)

def choose_model():
    def proceed():
        global model_selected
        selected_option = var.get()
        if selected_option == "":
            messagebox.showerror("Error", "Please select an option.")
            additional_options_window.destroy()
        else:
            model_selected = selected_option
            additional_options_window.destroy()
            User_Investment_Selection()

    additional_options_window = Toplevel()
    additional_options_window.title("Additional Options")
    additional_options_window.geometry("800x400")

    Label(additional_options_window, text="Select an additional option:", font=("Arial", 12)).pack(pady=10)

    var = StringVar()

    def option_clicked(option):
        
        current_option = var.get()

        var.set(option)

        option_buttons[option].config(relief=SUNKEN)


        if current_option != "":
            option_buttons[current_option].config(relief=RAISED)

    options = ["Logistic Regression", "Random Forest Classifier", "Decision Tree Classifier"]
    
    option_frame = Frame(additional_options_window)
    option_frame.pack(pady=10)

    var.set("")  

    option_buttons = {}  

    row_num = 0
    col_num = 0

    for option in options:
        button = Button(option_frame, text=option, width=20, height=2, command=lambda opt=option: option_clicked(opt))
        button.grid(row=row_num, column=col_num, padx=10, pady=5)
        option_buttons[option] = button

        col_num += 1
        if col_num == 2:
            col_num = 0
            row_num += 1

    proceed_button = Button(additional_options_window, text="Proceed", width=20, command=proceed)
    proceed_button.pack(pady=10)

def train_model_on_logistic_for_best():
    df = pd.read_csv(r'D:\Sem 4\AI\Project\consolidated_coin_data.csv')

    for column in df.columns:
        try:
            df[column] = df[column].str.replace(',', '').astype(float)
        except ValueError:
            print(f"Error converting column '{column}' to float.")

    numerical_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Market Cap']
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    X = df.drop(columns=['Currency', 'Date'])  # Features
    y = df['Currency']  # Target variable
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Training the model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print("Overall Accuracy:", accuracy)

    y_pred = model.predict(X_test)

    classification_rep = classification_report(y_test, y_pred)
    print("\nClassification Report:")
    print(classification_rep)

    confusion_mat = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(confusion_mat)

    precision_scores = precision_score(y_test, y_pred, average=None)
    currency_labels = model.classes_

    best_currency_index = precision_scores.argmax()
    best_currency = currency_labels[best_currency_index]

    print(f"The currency with the highest precision is: {best_currency}")

    labels = currency_labels
    sizes = precision_scores
    explode = [0.1 if label == best_currency else 0 for label in labels] 

    accuracy = accuracy_score(y_test, y_pred)
    
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, explode=explode, autopct='%1.1f%%', startangle=90)
    plt.axis('equal') 
    plt.title('Precision Scores for Currencies')

    return classification_rep, confusion_mat, plt, best_currency, accuracy

def show_logistic_from_best():
    classification_rep, _, plt, best_currency, model_accuracy = train_model_on_logistic_for_best()
    
    root = tk.Tk()
    root.title("Logistic Regression Prediction")
    root.geometry("930x400")

    lf_bg = 'LightSkyBlue'
    lbl_fg = 'black'
    lbl_font = ('Arial', 16, 'bold')
    sub_heading_font = ('Arial', 14, 'bold')
    text_font = ('Arial', 12)

    style = ttk.Style()
    style.configure('TFrame', background=lf_bg)
    style.configure('TLabel', foreground=lbl_fg)

    main_frame = ttk.Frame(root)
    main_frame.pack(fill='both', expand=True)

    left_frame = ttk.Frame(main_frame)
    left_frame.pack(side='left', padx=10, pady=10, fill='both', expand=True)

    classification_label = ttk.Label(left_frame, text="Classification Report", font=lbl_font, foreground=lbl_fg)
    classification_label.pack(pady=10)

    classification_scroll = Scrollbar(left_frame, orient='vertical')
    classification_text = tk.Text(left_frame, wrap='word', font=text_font, yscrollcommand=classification_scroll.set, width=50, foreground=lbl_fg)
    classification_text.insert(tk.END, classification_rep + f"\n\nAccording to Logistic Regression, {best_currency} currency will be resulting in the maximum profit gain and investing in {best_currency} is the best option right now.")
    classification_text.pack(side='left', padx=10, pady=5, fill='both', expand=True)
    classification_scroll.config(command=classification_text.yview)
    classification_scroll.pack(side='right', fill='y')

    center_frame = ttk.Frame(main_frame, width=100)
    center_frame.pack(side='left', fill='both', expand=True)

    right_frame = ttk.Frame(main_frame)
    right_frame.pack(side='left', padx=10, pady=10, fill='both', expand=True)

    pie_label = ttk.Label(right_frame, text="Precision Scores Pie Chart", font=lbl_font, foreground=lbl_fg)
    pie_label.pack(pady=10)

    canvas = FigureCanvasTkAgg(plt.gcf(), master=right_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(padx=10, pady=5, fill='both', expand=True)

    display = f" Best Currency Prediction \n\nModel: Logistic Regression\n\nIn the upcoming near future, {best_currency} is the digital currency that might result in high profits to investors according to my prediction.\n\n Please note that this is just a prediction and reality might vary accordingly. I am just an AI-Based Program that is used to predict according to its training dataset.\n\nModels Accuracy: {model_accuracy}"
    best_currency_label = ttk.Label(center_frame, text=display, font=text_font, foreground=lbl_fg, wraplength=200)
    best_currency_label.pack(pady=50, anchor='s')

    root.mainloop()


def train_model_on_random_forest_for_best():
    df = pd.read_csv(r'D:\Sem 4\AI\Project\consolidated_coin_data.csv')

    for column in df.columns:
        try:
            df[column] = df[column].str.replace(',', '').astype(float)
        except ValueError:
            print(f"Error converting column '{column}' to float.")

    numerical_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Market Cap']
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    X = df.drop(columns=['Currency', 'Date'])  # Features
    y = df['Currency']  # Target variable
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print("Overall Accuracy:", accuracy)

    y_pred = model.predict(X_test)

    classification_rep = classification_report(y_test, y_pred)
    print("\nClassification Report:")
    print(classification_rep)

    confusion_mat = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(confusion_mat)

    precision_scores = precision_score(y_test, y_pred, average=None)
    currency_labels = model.classes_

    best_currency_index = precision_scores.argmax()
    best_currency = currency_labels[best_currency_index]

    print(f"The currency with the highest precision is: {best_currency}")

    labels = currency_labels
    sizes = precision_scores
    explode = [0.1 if label == best_currency else 0 for label in labels]

    accuracy = accuracy_score(y_test, y_pred)
    
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, explode=explode, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')  
    plt.title('Precision Scores for Currencies')

    return classification_rep, confusion_mat, plt, best_currency, accuracy

def show_random_forest_from_best():
    classification_rep, _, plt, best_currency, model_accuracy = train_model_on_random_forest_for_best()
    
    root = tk.Tk()
    root.title("Random Forest Classifier Prediction")
    root.geometry("930x400")

    lf_bg = 'LightSkyBlue'
    lbl_fg = 'black'
    lbl_font = ('Arial', 16, 'bold')
    sub_heading_font = ('Arial', 14, 'bold')
    text_font = ('Arial', 12)

    style = ttk.Style()
    style.configure('TFrame', background=lf_bg)
    style.configure('TLabel', foreground=lbl_fg)

    main_frame = ttk.Frame(root)
    main_frame.pack(fill='both', expand=True)

    left_frame = ttk.Frame(main_frame)
    left_frame.pack(side='left', padx=10, pady=10, fill='both', expand=True)

    classification_label = ttk.Label(left_frame, text="Classification Report", font=lbl_font, foreground=lbl_fg)
    classification_label.pack(pady=10)

    classification_scroll = Scrollbar(left_frame, orient='vertical')
    classification_text = tk.Text(left_frame, wrap='word', font=text_font, yscrollcommand=classification_scroll.set, width=50, foreground=lbl_fg)
    classification_text.insert(tk.END, classification_rep + f"\n\nAccording to Random Forest Classifier, {best_currency} currency will be resulting in the maximum profit gain and investing in {best_currency} is best option right now.")
    classification_text.pack(side='left', padx=10, pady=5, fill='both', expand=True)
    classification_scroll.config(command=classification_text.yview)
    classification_scroll.pack(side='right', fill='y')

    center_frame = ttk.Frame(main_frame, width=100)
    center_frame.pack(side='left', fill='both', expand=True)

    right_frame = ttk.Frame(main_frame)
    right_frame.pack(side='left', padx=10, pady=10, fill='both', expand=True)

    pie_label = ttk.Label(right_frame, text="Precision Scores Pie Chart", font=lbl_font, foreground=lbl_fg)
    pie_label.pack(pady=10)

    canvas = FigureCanvasTkAgg(plt.gcf(), master=right_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(padx=10, pady=5, fill='both', expand=True)
    
    display = f" Best Currency Prediction \n\nModel: Random Forest Classifier\n\nIn the upcoming near future, {best_currency} is the digital currency that might result in high profits to investors according to my prediction.\n\n Please note that this is just a prediction and reality might vary accordingly. I am just an AI-Based Program that is used to predict according to its training dataset.\n\nModels Accuracy: {model_accuracy}"
    best_currency_label = ttk.Label(center_frame, text=display, font=text_font, foreground=lbl_fg, wraplength=200)
    best_currency_label.pack(pady=50, anchor='s')

    root.mainloop()


def train_model_decision_tree_for_best():
    df = pd.read_csv(r'D:\Sem 4\AI\Project\consolidated_coin_data.csv')

    for column in df.columns:
        try:
            df[column] = df[column].str.replace(',', '').astype(float)
        except ValueError:
            print(f"Error converting column '{column}' to float.")

    numerical_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Market Cap']
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    X = df.drop(columns=['Currency', 'Date'])  # Features
    y = df['Currency']  # Target variable
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print("Overall Accuracy:", accuracy)
    
    y_pred = model.predict(X_test)

    classification_rep = classification_report(y_test, y_pred)
    print("\nClassification Report:")
    print(classification_rep)

    confusion_mat = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(confusion_mat)

    precision_scores = precision_score(y_test, y_pred, average=None)
    currency_labels = model.classes_

    best_currency_index = precision_scores.argmax()
    best_currency = currency_labels[best_currency_index]

    print(f"The currency with the highest precision is: {best_currency}")

    labels = currency_labels
    sizes = precision_scores
    explode = [0.1 if label == best_currency else 0 for label in labels]  

    accuracy = accuracy_score(y_test, y_pred)
    
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, explode=explode, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')  
    plt.title('Precision Scores for Currencies (Decision Tree Classifier)')

    return classification_rep, confusion_mat, plt, best_currency, accuracy

def show_decision_tree_for_best():
    classification_rep, _, plt, best_currency, model_accuracy = train_model_decision_tree_for_best()

    root = tk.Tk()
    root.title("Decision Tree Classifier Prediction")
    root.geometry("930x400")

    lf_bg = 'LightSkyBlue'
    lbl_fg = 'black'
    lbl_font = ('Arial', 16, 'bold')
    sub_heading_font = ('Arial', 14, 'bold')
    text_font = ('Arial', 12)

    style = ttk.Style()
    style.configure('TFrame', background=lf_bg)
    style.configure('TLabel', foreground=lbl_fg)

    main_frame = ttk.Frame(root)
    main_frame.pack(fill='both', expand=True)

    left_frame = ttk.Frame(main_frame)
    left_frame.pack(side='left', padx=10, pady=10, fill='both', expand=True)

    classification_label = ttk.Label(left_frame, text="Classification Report", font=lbl_font, foreground=lbl_fg)
    classification_label.pack(pady=10)

    classification_scroll = Scrollbar(left_frame, orient='vertical')
    classification_text = tk.Text(left_frame, wrap='word', font=text_font, yscrollcommand=classification_scroll.set, width=50, foreground=lbl_fg)
    classification_text.insert(tk.END, classification_rep + f"\n\nAccording to Decision Tree Classification, {best_currency} currency will be resulting in the maximum profit gain and investing in {best_currency} is most suitable right now.")
    classification_text.pack(side='left', padx=10, pady=5, fill='both', expand=True)
    classification_scroll.config(command=classification_text.yview)
    classification_scroll.pack(side='right', fill='y')

    center_frame = ttk.Frame(main_frame, width=100)
    center_frame.pack(side='left', fill='both', expand=True)
    
    right_frame = ttk.Frame(main_frame)
    right_frame.pack(side='left', padx=10, pady=10, fill='both', expand=True)

    pie_label = ttk.Label(right_frame, text="Precision Scores Pie Chart", font=lbl_font, foreground=lbl_fg)
    pie_label.pack(pady=10)

    canvas = FigureCanvasTkAgg(plt.gcf(), master=right_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(padx=10, pady=5, fill='both', expand=True)

    display = f" Best Currency Prediction \n\nModel: Decision Tree Classifier\n\nIn the upcoming near future, {best_currency} is the digital currency that might result in high profits to investors according to my prediction.\n\n Please note that this is just a prediction and reality might vary accordingly. I am just an AI-Based Program that is used to predict according to its training dataset.\n\nModels Accuracy: {model_accuracy}"
    best_currency_label = ttk.Label(center_frame, text=display, font=text_font, foreground=lbl_fg, wraplength=200)
    best_currency_label.pack(pady=50, anchor='s')

    root.mainloop()


def train_model_linear_for_best(df):
    for column in df.columns:
        try:
            df[column] = df[column].str.replace(',', '').astype(float)
        except ValueError:
            print(f"Error converting column '{column}' to float.")

    numerical_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Market Cap']
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    X = df.drop(columns=['Currency', 'Date'])  
    y = df['High']  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)

    currency_labels = df['Currency'].unique()
    currency_r2_scores = {}

    for currency in currency_labels:
        currency_df = df[df['Currency'] == currency]
        X_curr = currency_df.drop(columns=['Currency', 'Date'])
        y_curr = currency_df['Close']

        r2_curr = model.score(X_curr, y_curr)
        currency_r2_scores[currency] = r2_curr
        

    best_currency = max(currency_r2_scores, key=currency_r2_scores.get)
    best_currency_r2 = currency_r2_scores[best_currency]

    return r2, best_currency, best_currency_r2

def show_linear_for_best():
    df = pd.read_csv(r'consolidated_coin_data.csv')

    r2, best_currency, best_currency_r2 = train_model_linear_for_best(df)

    root = tk.Tk()
    root.title("Linear Regression Results")
    root.geometry("930x400")

    lf_bg = 'LightSkyBlue'
    lbl_fg = 'black'
    lbl_font = ('Arial', 16, 'bold')
    sub_heading_font = ('Arial', 14, 'bold')
    text_font = ('Arial', 12)

    style = ttk.Style()
    style.configure('TFrame', background=lf_bg)
    style.configure('TLabel', foreground=lbl_fg)

    main_frame = ttk.Frame(root)
    main_frame.pack(fill='both', expand=True)

    left_frame = ttk.Frame(main_frame)
    left_frame.pack(side='left', padx=10, pady=10, fill='both', expand=True)

    r2_label = ttk.Label(left_frame, text=f"R-squared: {r2}", font=lbl_font, foreground=lbl_fg)
    r2_label.pack(pady=10)
    
    best_currency_label = ttk.Label(left_frame, text=f"Best Currency: {best_currency} with R-squared: {best_currency_r2}", font=lbl_font, foreground=lbl_fg)
    best_currency_label.pack(pady=10)

    center_frame = ttk.Frame(main_frame, width=100)
    center_frame.pack(side='left', fill='both', expand=True)

    right_frame = ttk.Frame(main_frame)
    right_frame.pack(side='left', padx=10, pady=10, fill='both', expand=True)

    display = f"Model: Logistic Regression\n\nAccording to Linear Regression, {best_currency} currency will be resulting in the maximum profit and investing in {best_currency} is the best option right now.\n\n Please note that this is just a prediction and reality might vary accordingly. I am just an AI-Based Program that is used to predict according to its training dataset."
    placeholder_label = ttk.Label(right_frame, text=display, font=lbl_font, foreground=lbl_fg, wraplength=200)
    placeholder_label.pack(pady=10)

    root.mainloop()
   

def train_model_on_logistic_for_worst():
    df = pd.read_csv(r'D:\Sem 4\AI\Project\consolidated_coin_data.csv')

    for column in df.columns:
        try:
            df[column] = df[column].str.replace(',', '').astype(float)
        except ValueError:
            print(f"Error converting column '{column}' to float.")

    numerical_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Market Cap']
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    X = df.drop(columns=['Currency', 'Date'])  # Features
    y = df['Currency']  # Target variable
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print("Overall Accuracy:", accuracy)

    y_pred = model.predict(X_test)


    classification_rep = classification_report(y_test, y_pred)
    print("\nClassification Report:")
    print(classification_rep)

    confusion_mat = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(confusion_mat)

    precision_scores = precision_score(y_test, y_pred, average=None)
    currency_labels = model.classes_

    worst_currency_index = precision_scores.argmin()
    worst_currency = currency_labels[worst_currency_index]

    print(f"The currency with the lowest precision is: {worst_currency}")

    labels = currency_labels
    sizes = precision_scores
    explode = [0.1 if label == worst_currency else 0 for label in labels] 

    accuracy = accuracy_score(y_test, y_pred)
    
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, explode=explode, autopct='%1.1f%%', startangle=90)
    plt.axis('equal') 
    plt.title('Precision Scores for Currencies')

    return classification_rep, confusion_mat, plt, worst_currency, accuracy

def show_logistic_from_worst():
    classification_rep, _, plt, worst_currency, model_accuracy = train_model_on_logistic_for_worst()
    
    root = tk.Tk()
    root.title("Logistic Regression Prediction")
    root.geometry("930x400")

    lf_bg = 'LightSkyBlue'
    lbl_fg = 'black'
    lbl_font = ('Arial', 16, 'bold')
    sub_heading_font = ('Arial', 14, 'bold')
    text_font = ('Arial', 12)

    style = ttk.Style()
    style.configure('TFrame', background=lf_bg)
    style.configure('TLabel', foreground=lbl_fg)

    main_frame = ttk.Frame(root)
    main_frame.pack(fill='both', expand=True)

    left_frame = ttk.Frame(main_frame)
    left_frame.pack(side='left', padx=10, pady=10, fill='both', expand=True)

    classification_label = ttk.Label(left_frame, text="Classification Report", font=lbl_font, foreground=lbl_fg)
    classification_label.pack(pady=10)

    classification_scroll = Scrollbar(left_frame, orient='vertical')
    classification_text = tk.Text(left_frame, wrap='word', font=text_font, yscrollcommand=classification_scroll.set, width=50, foreground=lbl_fg)
    classification_text.insert(tk.END, classification_rep + f"\n\nAccording to Logistic Regression, {worst_currency} currency will be resulting in the lowest profit and investing in {worst_currency} is the worst option right now.")
    classification_text.pack(side='left', padx=10, pady=5, fill='both', expand=True)
    classification_scroll.config(command=classification_text.yview)
    classification_scroll.pack(side='right', fill='y')

    center_frame = ttk.Frame(main_frame, width=100)
    center_frame.pack(side='left', fill='both', expand=True)

    right_frame = ttk.Frame(main_frame)
    right_frame.pack(side='left', padx=10, pady=10, fill='both', expand=True)

    pie_label = ttk.Label(right_frame, text="Precision Scores Pie Chart", font=lbl_font, foreground=lbl_fg)
    pie_label.pack(pady=10)

    canvas = FigureCanvasTkAgg(plt.gcf(), master=right_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(padx=10, pady=5, fill='both', expand=True)

    display = f" Worst Currency Prediction \n\nModel: Logistic Regression\n\nIn the upcoming near future, {worst_currency} is the digital currency that might result in very low profits to investors according to my prediction.\n\n Please note that this is just a prediction and reality might vary accordingly. I am just an AI-Based Program that is used to predict according to its training dataset.\n\nModels Accuracy : {model_accuracy}"
    best_currency_label = ttk.Label(center_frame, text=display, font=text_font, foreground=lbl_fg, wraplength=200)
    best_currency_label.pack(pady=50, anchor='s')

    root.mainloop()


def train_model_on_decisionTree_for_worst():
    df = pd.read_csv(r'D:\Sem 4\AI\Project\consolidated_coin_data.csv')

    for column in df.columns:
        try:
            df[column] = df[column].str.replace(',', '').astype(float)
        except ValueError:
            print(f"Error converting column '{column}' to float.")

    numerical_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Market Cap']
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    X = df.drop(columns=['Currency', 'Date'])  # Features
    y = df['Currency']  # Target variable
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print("Overall Accuracy:", accuracy)


    y_pred = model.predict(X_test)

    classification_rep = classification_report(y_test, y_pred)
    print("\nClassification Report:")
    print(classification_rep)

    confusion_mat = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(confusion_mat)

    precision_scores = precision_score(y_test, y_pred, average=None)
    currency_labels = model.classes_

    worst_currency_index = precision_scores.argmin()
    worst_currency = currency_labels[worst_currency_index]

    print(f"The currency with the lowest precision is: {worst_currency}")

    labels = currency_labels
    sizes = precision_scores
    explode = [0.1 if label == worst_currency else 0 for label in labels] 

    accuracy = accuracy_score(y_test, y_pred)
    
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, explode=explode, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')  
    plt.title('Precision Scores for Currencies')

    return classification_rep, confusion_mat, plt, worst_currency, accuracy

def show_decisionTree_for_worst():
    classification_rep, _, plt, worst_currency, model_accuracy = train_model_on_decisionTree_for_worst()

    root = tk.Tk()
    root.title("Decision Tree Classifier Prediction")
    root.geometry("930x400")

    lf_bg = 'LightSkyBlue'
    lbl_fg = 'black'
    lbl_font = ('Arial', 16, 'bold')
    sub_heading_font = ('Arial', 14, 'bold')
    text_font = ('Arial', 12)

    style = ttk.Style()
    style.configure('TFrame', background=lf_bg)
    style.configure('TLabel', foreground=lbl_fg)

    main_frame = ttk.Frame(root)
    main_frame.pack(fill='both', expand=True)

    left_frame = ttk.Frame(main_frame)
    left_frame.pack(side='left', padx=10, pady=10, fill='both', expand=True)

    classification_label = ttk.Label(left_frame, text="Classification Report", font=lbl_font, foreground=lbl_fg)
    classification_label.pack(pady=10)

    classification_scroll = Scrollbar(left_frame, orient='vertical')
    classification_text = tk.Text(left_frame, wrap='word', font=text_font, yscrollcommand=classification_scroll.set, width=50, foreground=lbl_fg)
    classification_text.insert(tk.END, classification_rep + f"\n\nAccording to Decision Tree Classifier, {worst_currency} currency will be resulting in the lowest profit and investing in {worst_currency} is the worst option right now.")
    classification_text.pack(side='left', padx=10, pady=5, fill='both', expand=True)
    classification_scroll.config(command=classification_text.yview)
    classification_scroll.pack(side='right', fill='y')

    center_frame = ttk.Frame(main_frame, width=100)
    center_frame.pack(side='left', fill='both', expand=True)

    right_frame = ttk.Frame(main_frame)
    right_frame.pack(side='left', padx=10, pady=10, fill='both', expand=True)

    pie_label = ttk.Label(right_frame, text="Precision Scores Pie Chart", font=lbl_font, foreground=lbl_fg)
    pie_label.pack(pady=10)

    canvas = FigureCanvasTkAgg(plt.gcf(), master=right_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(padx=10, pady=5, fill='both', expand=True)

    display = f" Worst Currency Prediction \n\nModel: Decision Tree Classifier\n\nIn the upcoming near future, {worst_currency} is the digital currency that might result in very low profits to investors according to my prediction.\n\n Please note that this is just a prediction and reality might vary accordingly. I am just an AI-Based Program that is used to predict according to its training dataset.\n\nModels Accuracy : {model_accuracy}"
    best_currency_label = ttk.Label(center_frame, text=display, font=text_font, foreground=lbl_fg, wraplength=200)
    best_currency_label.pack(pady=50, anchor='s')

    root.mainloop()


def train_model_on_forest_for_worst():
    df = pd.read_csv(r'D:\Sem 4\AI\Project\consolidated_coin_data.csv')

    for column in df.columns:
        try:
            df[column] = df[column].str.replace(',', '').astype(float)
        except ValueError:
            print(f"Error converting column '{column}' to float.")

    numerical_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Market Cap']
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    X = df.drop(columns=['Currency', 'Date'])  # Features
    y = df['Currency']  # Target variable
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print("Overall Accuracy:", accuracy)

    y_pred = model.predict(X_test)

    classification_rep = classification_report(y_test, y_pred)
    print("\nClassification Report:")
    print(classification_rep)

    confusion_mat = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(confusion_mat)

    precision_scores = precision_score(y_test, y_pred, average=None)
    currency_labels = model.classes_

    worst_currency_index = precision_scores.argmin()
    worst_currency = currency_labels[worst_currency_index]

    print(f"The currency with the lowest precision is: {worst_currency}")

    labels = currency_labels
    sizes = precision_scores
    explode = [0.1 if label == worst_currency else 0 for label in labels]  

    accuracy = accuracy_score(y_test, y_pred)
    
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, explode=explode, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')  
    plt.title('Precision Scores for Currencies')

    return classification_rep, confusion_mat, plt, worst_currency, accuracy

def show_forest_for_worst():
    classification_rep, _, plt, worst_currency, model_accuracy = train_model_on_forest_for_worst()

    root = tk.Tk()
    root.title("Random Forest Classifier Prediction")
    root.geometry("930x400")

    lf_bg = 'LightSkyBlue'
    lbl_fg = 'black'
    lbl_font = ('Arial', 16, 'bold')
    sub_heading_font = ('Arial', 14, 'bold')
    text_font = ('Arial', 12)

    style = ttk.Style()
    style.configure('TFrame', background=lf_bg)
    style.configure('TLabel', foreground=lbl_fg)

    main_frame = ttk.Frame(root)
    main_frame.pack(fill='both', expand=True)

    left_frame = ttk.Frame(main_frame)
    left_frame.pack(side='left', padx=10, pady=10, fill='both', expand=True)

    classification_label = ttk.Label(left_frame, text="Classification Report", font=lbl_font, foreground=lbl_fg)
    classification_label.pack(pady=10)

    classification_scroll = Scrollbar(left_frame, orient='vertical')
    classification_text = tk.Text(left_frame, wrap='word', font=text_font, yscrollcommand=classification_scroll.set, width=50, foreground=lbl_fg)
    classification_text.insert(tk.END, classification_rep + f"\n\nAccording to Random Forest Classifier, {worst_currency} currency will be resulting in the lowest profit and investing in {worst_currency} is the worst option right now.")
    classification_text.pack(side='left', padx=10, pady=5, fill='both', expand=True)
    classification_scroll.config(command=classification_text.yview)
    classification_scroll.pack(side='right', fill='y')

    center_frame = ttk.Frame(main_frame, width=100)
    center_frame.pack(side='left', fill='both', expand=True)

    right_frame = ttk.Frame(main_frame)
    right_frame.pack(side='left', padx=10, pady=10, fill='both', expand=True)

    pie_label = ttk.Label(right_frame, text="Precision Scores Pie Chart", font=lbl_font, foreground=lbl_fg)
    pie_label.pack(pady=10)

    canvas = FigureCanvasTkAgg(plt.gcf(), master=right_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(padx=10, pady=5, fill='both', expand=True)

    display = f" Worst Currency Prediction \n\nModel: Random Forest Classifier\n\nIn the upcoming near future, {worst_currency} is the digital currency that might result in very low profits to investors according to my prediction.\n\n Please note that this is just a prediction and reality might vary accordingly. I am just an AI-Based Program that is used to predict according to its training dataset.\n\nModels Accuracy : {model_accuracy}"
    best_currency_label = ttk.Label(center_frame, text=display, font=text_font, foreground=lbl_fg, wraplength=200)
    best_currency_label.pack(pady=50, anchor='s')

    root.mainloop()

def train_model_on_linear_for_worst(df):
    for column in df.columns:
        try:
            df[column] = df[column].str.replace(',', '').astype(float)
        except ValueError:
            print(f"Error converting column '{column}' to float.")

    numerical_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Market Cap']
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    X = df.drop(columns=['Currency', 'Date'])  # Features
    y = df['Low']  # Target variable
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)

    currency_labels = df['Currency'].unique()
    currency_r2_scores = {}

    for currency in currency_labels:
        currency_df = df[df['Currency'] == currency]
        X_curr = currency_df.drop(columns=['Currency', 'Date'])
        y_curr = currency_df['High']

        r2_curr = model.score(X_curr, y_curr)
        currency_r2_scores[currency] = r2_curr

    best_currency = max(currency_r2_scores, key=currency_r2_scores.get)
    best_currency_r2 = currency_r2_scores[best_currency]

    worst_currency = min(currency_r2_scores, key=currency_r2_scores.get)
    worst_currency_r2 = currency_r2_scores[worst_currency]

    return r2, best_currency, best_currency_r2, worst_currency, worst_currency_r2

def show_linear_for_worst():
    df = pd.read_csv(r'consolidated_coin_data.csv')

    r2, best_currency, best_currency_r2, worst_currency, worst_currency_r2 = train_model_on_linear_for_worst(df)

    root = tk.Tk()
    root.title("Linear Regression Results")
    root.geometry("930x400")

    lf_bg = 'LightSkyBlue'
    lbl_fg = 'black'
    lbl_font = ('Arial', 16, 'bold')
    sub_heading_font = ('Arial', 14, 'bold')
    text_font = ('Arial', 12)

    style = ttk.Style()
    style.configure('TFrame', background=lf_bg)
    style.configure('TLabel', foreground=lbl_fg)

    main_frame = ttk.Frame(root)
    main_frame.pack(fill='both', expand=True)

    left_frame = ttk.Frame(main_frame)
    left_frame.pack(side='left', padx=10, pady=10, fill='both', expand=True)

    r2_label = ttk.Label(left_frame, text=f"R-squared: {r2}", font=lbl_font, foreground=lbl_fg)
    r2_label.pack(pady=10)

    worst_currency_label = ttk.Label(left_frame, text=f"Worst Currency: {worst_currency} with R-squared: {worst_currency_r2}", font=lbl_font, foreground=lbl_fg)
    worst_currency_label.pack(pady=10)

    center_frame = ttk.Frame(main_frame, width=100)
    center_frame.pack(side='left', fill='both', expand=True)

    right_frame = ttk.Frame(main_frame)
    right_frame.pack(side='left', padx=10, pady=10, fill='both', expand=True)

    display = f"Model: Logistic Regression\n\nAccording to Linear Regression, {worst_currency} currency will be resulting in the minimum profit and investing in {worst_currency} is the worst option right now.\n\n Please note that this is just a prediction and reality might vary accordingly. I am just an AI-Based Program that is used to predict according to its training dataset."
    placeholder_label = ttk.Label(right_frame, text=display, font=lbl_font, foreground=lbl_fg, wraplength=200)
    placeholder_label.pack(pady=10)

    root.mainloop()


def calculate_precision_and_accuracy_for_all_currencies(model, X_test, y_test):
    y_pred = model.predict(X_test)

    precision_scores = precision_score(y_test, y_pred, average=None)
    currency_labels = model.classes_

    accuracy_scores = []
    for currency in currency_labels:
        currency_mask = (y_test == currency)
        accuracy = accuracy_score(y_test[currency_mask], y_pred[currency_mask])
        accuracy_scores.append(accuracy)

    print("Precision and Accuracy Scores for all unique currencies:")
    for currency, precision, accuracy in zip(currency_labels, precision_scores, accuracy_scores):
        print(f"{currency}: Precision - {precision}, Accuracy - {accuracy}")

    return precision_scores, accuracy_scores, currency_labels

def train_model_on_logistic_user_choice(X_train, X_test, y_train, y_test):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print("Overall Accuracy:", accuracy)

    y_pred = model.predict(X_test)

    classification_rep = classification_report(y_test, y_pred)
    print("\nClassification Report:")
    print(classification_rep)

    confusion_mat = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(confusion_mat)

    # Getting Predictions for all Unique Crypto Coins
    precision_scores, accuracy_scores, currency_labels = calculate_precision_and_accuracy_for_all_currencies(model, X_test, y_test)

    # Find the currency with the highest precision
    best_currency_index = precision_scores.argmax()
    best_currency = currency_labels[best_currency_index]
    print(f"The currency with the highest precision is: {best_currency}")

    labels = currency_labels
    sizes = precision_scores
    explode = [0.1 if precision > 0.5 else 0 for precision in precision_scores]  

    accuracy = accuracy_score(y_test, y_pred)
    
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, explode=explode, autopct='%1.1f%%', startangle=90)
    plt.axis('equal') 
    plt.title('Precision Scores for Currencies')

    return classification_rep, confusion_mat, plt, best_currency, accuracy, currency_labels, precision_scores, accuracy_scores

def show_logistic_user_choice():
    df = pd.read_csv(r'D:\Sem 4\AI\Project\consolidated_coin_data.csv')

    for column in df.columns:
        try:
            df[column] = df[column].str.replace(',', '').astype(float)
        except ValueError:
            print(f"Error converting column '{column}' to float.")

    numerical_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Market Cap']
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    X = df.drop(columns=['Currency', 'Date'])  # Features
    y = df['Currency']  # Target variable
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    classification_rep, _, plt, best_currency, model_accuracy, currency_labels, precision_scores, accuracy_scores = train_model_on_logistic_user_choice(X_train, X_test, y_train, y_test)

    root = tk.Tk()
    root.title("Logistic Regression Prediction")
    root.geometry("930x400")

    lf_bg = 'LightSkyBlue'
    lbl_fg = 'black'
    lbl_font = ('Arial', 16, 'bold')
    sub_heading_font = ('Arial', 14, 'bold')
    text_font = ('Arial', 12)

    style = ttk.Style()
    style.configure('TFrame', background=lf_bg)
    style.configure('TLabel', foreground=lbl_fg)

    main_frame = ttk.Frame(root)
    main_frame.pack(fill='both', expand=True)

    left_frame = ttk.Frame(main_frame)
    left_frame.pack(side='left', padx=10, pady=10, fill='both', expand=True)

    accuracy_precision_label = ttk.Label(left_frame, text="Accuracy and Precision for all unique currencies", font=lbl_font, foreground=lbl_fg)
    accuracy_precision_label.pack(pady=10)

    acc_prec_text = ""
    for currency, precision, accuracy in zip(currency_labels, precision_scores, accuracy_scores):
        acc_prec_text += f"{currency}: Precision - {precision}, Accuracy - {accuracy}\n"

    acc_prec_scroll = Scrollbar(left_frame, orient='vertical')
    acc_prec_text_widget = tk.Text(left_frame, wrap='word', font=text_font, yscrollcommand=acc_prec_scroll.set, foreground=lbl_fg)
    acc_prec_text_widget.insert(tk.END, acc_prec_text)
    acc_prec_text_widget.pack(side='left', padx=10, pady=5, fill='both', expand=True)
    acc_prec_scroll.config(command=acc_prec_text_widget.yview)
    acc_prec_scroll.pack(side='right', fill='y')

    center_frame = ttk.Frame(main_frame, width=100)
    center_frame.pack(side='left', fill='both', expand=True)

    right_frame = ttk.Frame(main_frame)
    right_frame.pack(side='left', padx=10, pady=10, fill='both', expand=True)

    pie_label = ttk.Label(right_frame, text="Precision Scores Pie Chart", font=lbl_font, foreground=lbl_fg)
    pie_label.pack(pady=10)

    canvas = FigureCanvasTkAgg(plt.gcf(), master=right_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(padx=10, pady=5, fill='both', expand=True)

    display = f" User Investment Analysis \n\nModel: Logistic Regression\n\nThis is the chart that shows my predictions. Currencies with a higher chance of profit are exploded, and are recommended to be invested in.\n\n Please note that this is just a prediction and reality might vary accordingly. I am just an AI-Based Program that is used to predict according to its training dataset.\n\nModels Accuracy: {model_accuracy}"
    best_currency_label = ttk.Label(center_frame, text=display, font=text_font, foreground=lbl_fg, wraplength=200)
    best_currency_label.pack(pady=50, anchor='s')

    root.mainloop()



def train_model_on_forest_user_choice(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print("Overall Accuracy:", accuracy)

    y_pred = model.predict(X_test)

    classification_rep = classification_report(y_test, y_pred)
    print("\nClassification Report:")
    print(classification_rep)

    confusion_mat = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(confusion_mat)

    # Getting Predictions for all Unique Crypto Coins
    precision_scores, accuracy_scores, currency_labels = calculate_precision_and_accuracy_for_all_currencies(model, X_test, y_test)

    # Find the currency with the highest precision
    best_currency_index = precision_scores.argmax()
    best_currency = currency_labels[best_currency_index]
    print(f"The currency with the highest precision is: {best_currency}")

    labels = currency_labels
    sizes = precision_scores
    explode = [0.1 if precision > 0.5 else 0 for precision in precision_scores]  

    accuracy = accuracy_score(y_test, y_pred)
    
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, explode=explode, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')  
    plt.title('Precision Scores for Currencies')

    return classification_rep, confusion_mat, plt, best_currency, accuracy, currency_labels, precision_scores, accuracy_scores

def show_rforest_user_choice():
    df = pd.read_csv(r'D:\Sem 4\AI\Project\consolidated_coin_data.csv')

    for column in df.columns:
        try:
            df[column] = df[column].str.replace(',', '').astype(float)
        except ValueError:
            print(f"Error converting column '{column}' to float.")

    numerical_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Market Cap']
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    X = df.drop(columns=['Currency', 'Date'])  # Features
    y = df['Currency']  # Target variable
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    classification_rep, _, plt, best_currency, model_accuracy, currency_labels, precision_scores, accuracy_scores = train_model_on_forest_user_choice(X_train, X_test, y_train, y_test)

    root = tk.Tk()
    root.title("Random Forest Classifier Prediction")
    root.geometry("930x400")

    lf_bg = 'LightSkyBlue'
    lbl_fg = 'black'
    lbl_font = ('Arial', 16, 'bold')
    sub_heading_font = ('Arial', 14, 'bold')
    text_font = ('Arial', 12)

    style = ttk.Style()
    style.configure('TFrame', background=lf_bg)
    style.configure('TLabel', foreground=lbl_fg)

    main_frame = ttk.Frame(root)
    main_frame.pack(fill='both', expand=True)

    left_frame = ttk.Frame(main_frame)
    left_frame.pack(side='left', padx=10, pady=10, fill='both', expand=True)

    accuracy_precision_label = ttk.Label(left_frame, text="Accuracy and Precision for all unique currencies", font=lbl_font, foreground=lbl_fg)
    accuracy_precision_label.pack(pady=10)

    acc_prec_text = ""
    for currency, precision, accuracy in zip(currency_labels, precision_scores, accuracy_scores):
        acc_prec_text += f"{currency}: Precision - {precision}, Accuracy - {accuracy}\n"

    acc_prec_scroll = Scrollbar(left_frame, orient='vertical')
    acc_prec_text_widget = tk.Text(left_frame, wrap='word', font=text_font, yscrollcommand=acc_prec_scroll.set, foreground=lbl_fg)
    acc_prec_text_widget.insert(tk.END, acc_prec_text)
    acc_prec_text_widget.pack(side='left', padx=10, pady=5, fill='both', expand=True)
    acc_prec_scroll.config(command=acc_prec_text_widget.yview)
    acc_prec_scroll.pack(side='right', fill='y')

    center_frame = ttk.Frame(main_frame, width=100)
    center_frame.pack(side='left', fill='both', expand=True)

    right_frame = ttk.Frame(main_frame)
    right_frame.pack(side='left', padx=10, pady=10, fill='both', expand=True)

    pie_label = ttk.Label(right_frame, text="Precision Scores Pie Chart", font=lbl_font, foreground=lbl_fg)
    pie_label.pack(pady=10)

    canvas = FigureCanvasTkAgg(plt.gcf(), master=right_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(padx=10, pady=5, fill='both', expand=True)

    display = f" User Investment Analysis \n\nModel: Random Forest Classifier\n\nThis is the chart that shows my predictions. Currencies with a higher chance of profit are exploded, and are recommended to be invested in.\n\n Please note that this is just a prediction and reality might vary accordingly. I am just an AI-Based Program that is used to predict according to its training dataset.\n\nModels Accuracy: {model_accuracy}"
    best_currency_label = ttk.Label(center_frame, text=display, font=text_font, foreground=lbl_fg, wraplength=200)
    best_currency_label.pack(pady=50, anchor='s')

    root.mainloop()


def train_model_on_tree_user_choice(X_train, X_test, y_train, y_test):
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print("Overall Accuracy:", accuracy)

    y_pred = model.predict(X_test)

    classification_rep = classification_report(y_test, y_pred)
    print("\nClassification Report:")
    print(classification_rep)

    confusion_mat = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(confusion_mat)

    # Getting Predictions for all Unique Crypto Coins
    precision_scores, accuracy_scores, currency_labels = calculate_precision_and_accuracy_for_all_currencies(model, X_test, y_test)

    # Find the currency with the highest precision
    best_currency_index = precision_scores.argmax()
    best_currency = currency_labels[best_currency_index]
    print(f"The currency with the highest precision is: {best_currency}")

    labels = currency_labels
    sizes = precision_scores
    explode = [0.1 if precision > 0.5 else 0 for precision in precision_scores]  

    accuracy = accuracy_score(y_test, y_pred)
    
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, explode=explode, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')  
    plt.title('Precision Scores for Currencies')

    return classification_rep, confusion_mat, plt, best_currency, accuracy, currency_labels, precision_scores, accuracy_scores

def show_d_t_user_choice():
    df = pd.read_csv(r'D:\Sem 4\AI\Project\consolidated_coin_data.csv')

    for column in df.columns:
        try:
            df[column] = df[column].str.replace(',', '').astype(float)
        except ValueError:
            print(f"Error converting column '{column}' to float.")

    numerical_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Market Cap']
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    X = df.drop(columns=['Currency', 'Date'])  # Features
    y = df['Currency']  # Target variable
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    classification_rep, _, plt, best_currency, model_accuracy, currency_labels, precision_scores, accuracy_scores = train_model_on_tree_user_choice(X_train, X_test, y_train, y_test)

    root = tk.Tk()
    root.title("Random Forest Classifier Prediction")
    root.geometry("930x400")

    lf_bg = 'LightSkyBlue'
    lbl_fg = 'black'
    lbl_font = ('Arial', 16, 'bold')
    sub_heading_font = ('Arial', 14, 'bold')
    text_font = ('Arial', 12)

    style = ttk.Style()
    style.configure('TFrame', background=lf_bg)
    style.configure('TLabel', foreground=lbl_fg)

    main_frame = ttk.Frame(root)
    main_frame.pack(fill='both', expand=True)

    left_frame = ttk.Frame(main_frame)
    left_frame.pack(side='left', padx=10, pady=10, fill='both', expand=True)

    accuracy_precision_label = ttk.Label(left_frame, text="Accuracy and Precision for all unique currencies", font=lbl_font, foreground=lbl_fg)
    accuracy_precision_label.pack(pady=10)

    acc_prec_text = ""
    for currency, precision, accuracy in zip(currency_labels, precision_scores, accuracy_scores):
        acc_prec_text += f"{currency}: Precision - {precision}, Accuracy - {accuracy}\n"

    acc_prec_scroll = Scrollbar(left_frame, orient='vertical')
    acc_prec_text_widget = tk.Text(left_frame, wrap='word', font=text_font, yscrollcommand=acc_prec_scroll.set, foreground=lbl_fg)
    acc_prec_text_widget.insert(tk.END, acc_prec_text)
    acc_prec_text_widget.pack(side='left', padx=10, pady=5, fill='both', expand=True)
    acc_prec_scroll.config(command=acc_prec_text_widget.yview)
    acc_prec_scroll.pack(side='right', fill='y')

    center_frame = ttk.Frame(main_frame, width=100)
    center_frame.pack(side='left', fill='both', expand=True)

    right_frame = ttk.Frame(main_frame)
    right_frame.pack(side='left', padx=10, pady=10, fill='both', expand=True)

    pie_label = ttk.Label(right_frame, text="Precision Scores Pie Chart", font=lbl_font, foreground=lbl_fg)
    pie_label.pack(pady=10)

    canvas = FigureCanvasTkAgg(plt.gcf(), master=right_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(padx=10, pady=5, fill='both', expand=True)

    display = f" User Investment Analysis \n\nModel: Decision Tree Classifier\n\nThis is the chart that shows my predictions. Currencies with a higher chance of profit are exploded, and are recommended to be invested in.\n\n Please note that this is just a prediction and reality might vary accordingly. I am just an AI-Based Program that is used to predict according to its training dataset.\n\nModels Accuracy: {model_accuracy}"
    best_currency_label = ttk.Label(center_frame, text=display, font=text_font, foreground=lbl_fg, wraplength=200)
    best_currency_label.pack(pady=50, anchor='s')

    root.mainloop()









# CHECK SELECTION
def User_Investment_Selection():
    global model_selected
    
    if model_selected == "Logistic Regression":
        show_logistic_user_choice()
        
    elif model_selected == "Random Forest Classifier":
        show_rforest_user_choice()
        
    elif model_selected == "Decision Tree Classifier":
        show_d_t_user_choice()
                

def check_selection():
    global model_selected
    
    if model_selected == "Logistic Regression":
        show_logistic_from_best()
    
    elif model_selected == "Random Forest Classifier":
        show_random_forest_from_best()
        
    elif model_selected == "Decision Tree Classifier":
        show_decision_tree_for_best()
        
    elif model_selected == "Linear Regression":
        show_linear_for_best()
        

def check_selection2 ():
    global model_selected
    
    if model_selected == "Logistic Regression":
        show_logistic_from_worst()
        
    elif model_selected == "Decision Tree Classifier":
        show_decisionTree_for_worst()
        
    elif model_selected == "Random Forest Classifier":
        show_forest_for_worst()
        
    elif model_selected == "Linear Regression":
        show_linear_for_worst()

    
def show_loading_page():
    root = Tk()
    root.geometry("750x311")
    root.title("Loading...")
    
    image_path = "C:\Users\Rizwan\Desktop\AI LAB\AI Project\AI_Loading.png"
    img = Image.open(image_path)
    img = img.resize((750, 311), Image.ANTIALIAS if "ANTIALIAS" in dir(Image) else Image.BILINEAR)
    photo = ImageTk.PhotoImage(img)
    
    label = Label(root, image=photo)
    label.image = photo
    label.pack()
    
    root.after(5000, lambda: [root.destroy(), show_main_page()])
    
    root.mainloop()
   
   
 #int main()
if __name__ == "__main__":
    show_loading_page()