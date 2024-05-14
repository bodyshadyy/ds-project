import tkinter as tk
import customtkinter as ctk
from matplotlib.figure import Figure
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import ListedColormap
from tkinter import ttk, filedialog
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso,LogisticRegression
from sklearn.neighbors import KNeighborsRegressor,KNeighborsClassifier
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.naive_bayes import GaussianNB

class SampleApp(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.title("Data science project")
        self.geometry("600x500")
        ctk.set_appearance_mode("System")  # Modes: "System" (default), "Dark", "Light"
        ctk.set_default_color_theme("blue")  # Themes: "blue" (default), "green", "dark-blue"


        self.home_page = HomePage(self)
        self.page1 = Page1(self)
        self.page2 = Page2(self)
        self.page3 = Page3(self)
        self.page4 = Page4(self)

        self.show_home_page()

    def show_home_page(self):
        self.page1.pack_forget()
        self.page2.pack_forget()
        self.page3.pack_forget()
        self.page4.pack_forget()
        self.home_page.pack()

    def show_page1(self):
        self.home_page.pack_forget()
        self.page1.pack()

    def show_page2(self):
        self.home_page.pack_forget()
        self.page2.pack()

    def show_page3(self):
        self.home_page.pack_forget()
        self.page3.pack()
    def show_page4(self):
        self.home_page.pack_forget()
        self.page4.pack()

class HomePage(ctk.CTkFrame):
    def __init__(self, master):
        ctk.CTkFrame.__init__(self, master)


        self.button4 = ctk.CTkButton(self, text="show csv data ", command=master.show_page4)
        self.button4.pack(pady=10)


        self.button1 = ctk.CTkButton(self, text="regression", command=master.show_page2)
        self.button1.pack(pady=10)

        self.button2 = ctk.CTkButton(self, text="classification", command=master.show_page3)
        self.button2.pack(pady=10)

        self.button3 = ctk.CTkButton(self, text="cluster", command=master.show_page1)
        self.button3.pack(pady=10)


class Page4(ctk.CTkFrame):
    def __init__(self, master):
        ctk.CTkFrame.__init__(self, master)
        self.header = ctk.CTkFrame(self)
        self.header.pack(fill="x")
        self.back_button = ctk.CTkButton(self.header, text="back", command=master.show_home_page)
        self.back_button.pack(side="left")

        self.page2 = ctk.CTkLabel(self.header, text="csv", font=("Helvetica", 18))
        self.page2.pack(side="left", fill="both", expand=True)

        button_frame = ctk.CTkFrame(self)
        button_frame.pack(fill=ctk.X)

        # Button to open file dialog
        self.open_button = ctk.CTkButton(button_frame, text="Open CSV File", command=self.load_csv)
        self.open_button.pack(padx=5, pady=5)

        # Frame for the table with scrollbars
        self.table_frame = ctk.CTkFrame(self,border_color="blue")
        self.table_frame.pack(fill=ctk.BOTH, expand=True,padx=5)

        # Canvas and Scrollbars
        self.canvas = ctk.CTkCanvas(self.table_frame)
        self.canvas.pack(side=ctk.LEFT, fill=ctk.BOTH, expand=True)

        self.scrollbar_x = ctk.CTkScrollbar(self.table_frame, orientation=ctk.HORIZONTAL, command=self.canvas.xview)
        self.scrollbar_x.pack(side=ctk.BOTTOM, fill=ctk.X)
        self.scrollbar_y = ctk.CTkScrollbar(self.table_frame, orientation=ctk.VERTICAL, command=self.canvas.yview)
        self.scrollbar_y.pack(side=ctk.RIGHT, fill=ctk.Y)
        self.canvas.configure(xscrollcommand=self.scrollbar_x.set, yscrollcommand=self.scrollbar_y.set)
        self.canvas.bind('<Configure>', self.on_canvas_configure)

        self.table_container = ctk.CTkFrame(self.canvas)
        self.canvas.create_window((0, 0), window=self.table_container, anchor="nw")

    def on_canvas_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox('all'))

    def load_csv(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if file_path:
            try:
                self.display_csv(file_path)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to read file\n{str(e)}")

    def display_csv(self, file_path):
        for widget in self.table_container.winfo_children():
            widget.destroy()

        df = pd.read_csv(file_path)
        if df.empty:
            messagebox.showwarning("Warning", "The CSV file is empty")
            return

        # Create the table headers
        columns = df.columns
        for i, column in enumerate(columns):
            label = ctk.CTkLabel(self.table_container, text=column)
            label.grid(row=0, column=i, sticky="nsew", padx=10)

        # Create the table rows
        for i, row in df.iterrows():
            for j, value in enumerate(row):
                label = ctk.CTkLabel(self.table_container, text=value)
                label.grid(row=i+1, column=j, sticky="nsew", padx=10)

        # Make the table cells expand with the window
        for i in range(len(columns)):
            self.table_container.grid_columnconfigure(i, weight=1)
        for i in range(len(df)):
            self.table_container.grid_rowconfigure(i, weight=1)

        self.canvas.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox('all'))




class Page1(ctk.CTkFrame):
    def __init__(self, master):
        ctk.CTkFrame.__init__(self, master)

        self.header = ctk.CTkLabel(self, text="clustering", font=("Helvetica", 18))
        self.header.grid(row=0, column=0, columnspan=1)

        self.back_button = ctk.CTkButton(self, text="Back", command=master.show_home_page)
        self.back_button.grid(row=0, column=0, columnspan=1, sticky='w')
        self.btn_load = ctk.CTkButton(self, text="Load CSV",command=self.load_csv)
        self.btn_load.grid(row=1)
        self.canvas = tk.Canvas(self, width=600, height=400)
        self.canvas.grid(row=2)
    def load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.data = pd.read_csv(file_path)
            self.visualize_clusters()

    def visualize_clusters(self):
        k = 8  # Number of clusters (you can adjust this)
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(self.data)
        labels = kmeans.predict(self.data)
        centroids = kmeans.cluster_centers_

        self.canvas.delete("all")
        fig, ax = plt.subplots(figsize=(6, 4))

        colors = ['r', 'g', 'b', 'y', 'c', 'm','k','pink']
        for i in range(k):
            points = np.array([self.data.iloc[j].values for j in range(len(self.data)) if labels[j] == i])
            ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])

        ax.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=200, c='black')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('K-Means Clustering')

        canvas = FigureCanvasTkAgg(fig, master=self.canvas)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)


class GraphFrame(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)

        # Matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('y')
        self.ax.set_title('Regression Plot')

        # Canvas to display plot
        self.plot_canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.plot_canvas.draw()
        self.plot_canvas.get_tk_widget().pack(expand=True, fill=tk.BOTH)

    def update_plot(self, X_train, X_test, y_train, y_test, x_vals, y_vals,
                    mse_train, mse_test):
        self.ax.clear()
        self.ax.scatter(X_train, y_train, color='blue', label='Train Data')
        self.ax.scatter(X_test, y_test, color='red', label='Test Data')
        self.ax.plot(x_vals, y_vals, color='green', label='Regression Line')
        self.ax.text(0.5,
                     0.9,
                     f'MSE Train: {mse_train:.2f}\nMSE Test: {mse_test:.2f}',
                     transform=self.ax.transAxes,
                     fontsize=10)
        self.ax.legend()

        self.plot_canvas.draw()

class Page2(ctk.CTkFrame):
    def __init__(self, master):
        ctk.CTkFrame.__init__(self, master)

        self.header=ctk.CTkFrame(self)
        self.header.pack(fill="x")
        self.back_button = ctk.CTkButton(self.header, text="back", command=master.show_home_page)
        self.back_button.pack(side="left")

        self.page2 = ctk.CTkLabel(self.header, text="regression", font=("Helvetica", 18))
        self.page2.pack(side="left", fill="both", expand=True)


        self.selected_regressor = tk.StringVar()
        self.selected_regressor.set("linear")
        self.encode = tk.StringVar()
        self.encode.set("label")
        self.degree = tk.IntVar()
        self.degree.set(1)  # Default degree
        self.model_dic = {'ridge': Ridge(), 'lasso': Lasso(), 'KNN': KNeighborsRegressor(), 'linear': LinearRegression(),
                          'random_forest': RandomForestRegressor()}

        self.create_widgets2()
        plot_button = ctk.CTkButton(self,
                                 text="Plot Regression from CSV",
                                 command= self.plot_regression_from_csv)
        plot_button.pack()
        self.graph_frame = GraphFrame(self)
        self.graph_frame.pack(expand=True, fill="both")

        self.canvas = tk.Canvas(self, width=600, height=400)
        self.canvas.pack()
    def create_widgets2(self):
        # Notebook for tabs
        notebook = ttk.Notebook(self)
        notebook.pack(expand=True, fill="both")

        # Tab for importing CSV
        csv_tab = ctk.CTkFrame(notebook)
        notebook.add(csv_tab, text="Import CSV")
        self.create_csv_tab(csv_tab)

        # Tab for regression types
        regression_tab = ctk.CTkFrame(notebook)
        notebook.add(regression_tab, text="Regression")
        self.create_regression_tab(regression_tab)

        # Tab for regularization types
        regularization_tab = ctk.CTkFrame(notebook)
        notebook.add(regularization_tab, text="Regularization")
        self.create_regularization_tab(regularization_tab)

        # Tab for preprocessing
        preprocess_tab = ctk.CTkFrame(notebook)
        notebook.add(preprocess_tab, text="Preprocess")
        self.create_preprocess_tab(preprocess_tab)

    def create_csv_tab(self, parent):
        # Frame for importing CSV
        csv_frame = ctk.CTkFrame(parent)
        csv_frame.pack(pady=10)

        csv_label = ctk.CTkLabel(csv_frame, text="Select CSV file:")
        csv_label.grid(row=0, column=0, padx=10)

        self.csv_path_var = tk.StringVar()
        csv_entry = ctk.CTkEntry(csv_frame,
                              textvariable=self.csv_path_var,
                              state="readonly",
                              width=40)
        csv_entry.grid(row=0, column=1, padx=10)

        csv_button = ctk.CTkButton(csv_frame,
                                text="Browse",
                                command=self.select_csv_file)
        csv_button.grid(row=0, column=2, padx=10)

    def create_regression_tab(self, parent):
        # Frame for the regression radio buttons
        regression_frame = ctk.CTkFrame(parent)
        regression_frame.pack(pady=10)

        linear_radio = ctk.CTkRadioButton(regression_frame,
                                       text="Linear Regression",
                                       variable=self.selected_regressor,
                                       value="linear")
        linear_radio.grid(row=0, column=0, padx=10)

        random_forest_radio = ctk.CTkRadioButton(regression_frame,
                                              text="Random Forest Regression",
                                              variable=self.selected_regressor,
                                              value="random_forest")
        random_forest_radio.grid(
            row=1,
            column=0,
        )
        KNN_button = ctk.CTkRadioButton(regression_frame,
                                     text="KNN Regression",
                                     variable=self.selected_regressor,
                                     value="KNN")
        KNN_button.grid(row=1, column=1)

    def create_regularization_tab(self, parent):
        # Frame for the regularization radio buttons
        regularization_frame = ctk.CTkFrame(parent)
        regularization_frame.pack(pady=10)

        ridge_radio = ctk.CTkRadioButton(regularization_frame,
                                      text="Ridge Regression",
                                      variable=self.selected_regressor,
                                      value="ridge")
        ridge_radio.grid(row=0, column=0, padx=10)

        lasso_radio = ctk.CTkRadioButton(regularization_frame,
                                      text="Lasso Regression",
                                      variable=self.selected_regressor,
                                      value="lasso")
        lasso_radio.grid(row=0, column=1, padx=10)

    def create_preprocess_tab(self, parent):

        # Frame for preprocessing options
        preprocess_frame = ctk.CTkFrame(parent)
        preprocess_frame.pack(pady=10)

        # Checkboxes for preprocessing options
        self.scale_checkbox = ttk.Checkbutton(preprocess_frame,
                                              text="Standard Scale",
                                              variable=tk.BooleanVar())
        self.scale_checkbox.grid(row=0, column=0, padx=10)

        # Degree of polynomial
        degree_label = ctk.CTkLabel(preprocess_frame, text="Polynomial Degree:")
        degree_label.grid(row=1, column=0, padx=10)

        degree_spinbox = ttk.Spinbox(preprocess_frame,
                                     from_=1,
                                     to=10,
                                     textvariable=self.degree,
                                     width=5)
        degree_spinbox.grid(row=1, column=1)

        # Encoder selection
        label_encoder = ctk.CTkRadioButton(preprocess_frame, text='Label Encoder', variable=self.encode, value='label')
        label_encoder.grid(row=2, column=0, padx=10)

        hot_encoder = ctk.CTkRadioButton(preprocess_frame, text='One Hot Encoder', variable=self.encode, value='hot')
        hot_encoder.grid(row=2, column=1, padx=10)

    def select_csv_file(self):
        csv_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        self.csv_path_var.set(csv_path)

    def preprocess_data(self):
        csv_path = self.csv_path_var.get()
        if not csv_path:
            return

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(e)
            return

        # Handle categorical variables
        if self.encode.get() == 'label':
            encoder = LabelEncoder()
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = encoder.fit_transform(df[col])
        elif self.encode.get() == 'hot':
            df = pd.get_dummies(df, columns=[col for col in df.columns if df[col].dtype == 'object'], drop_first=True)

        # Handle missing values
        imputer = SimpleImputer(strategy='mean')
        df_filled = pd.DataFrame(imputer.fit_transform(df))

        X = df_filled.iloc[:, :-1].values
        y = df_filled.iloc[:, -1].values

        # Apply PCA to reduce dimensionality
        pca = PCA(n_components=0.95)  # Retain 95% of variance
        X = pca.fit_transform(X)

        # Apply selected preprocessing options
        if self.scale_checkbox.instate(['selected']):
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=0.2,
                                                            random_state=42)

        return X_train, X_test, y_train, y_test, X, y

    def plot_regression_from_csv(self):
        # Preprocess data
        X_train, X_test, y_train, y_test, X, y = self.preprocess_data()
        if X_train is None:
            return

        # Create regression model
        model = make_pipeline(PolynomialFeatures(degree=self.degree.get()),
                              self.model_dic[self.selected_regressor.get()])

        model.fit(X_train, y_train)

        # Calculate MSE for train and test data
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        mse_train = mean_squared_error(y_train, y_train_pred)
        mse_test = mean_squared_error(y_test, y_test_pred)

        # Plot the data and regression line using Matplotlib
        x_vals = np.linspace(X_train.min(), X_train.max(), 100)
        y_vals = model.predict(x_vals.reshape(-1, 1))

        # Update the plot
        self.graph_frame.update_plot(X_train, X_test, y_train, y_test, x_vals,
                                     y_vals, mse_train, mse_test)


class MatplotlibFigure:
    def __init__(self, master):
        self.master = master
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=master)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        self.mse_train = None
        self.mse_test = None

    def update_plot(self, X_train, y_train, X_test, y_test, classifier):
        self.ax.clear()

        # Plot training points
        cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
        cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

        h = .02  # Step size in the mesh
        x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
        y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        self.ax.pcolormesh(xx, yy, Z, cmap=cmap_light)

        # Plot the training points
        self.ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap_bold, marker='o', label='Training Data')

        # Plot testing points with colors based on their classification
        self.ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cmap_bold, marker='x', label='Test Data')



        # Plot the testing points
        y_test_pred = classifier.predict(X_test)
        incorrect_indices = np.where(y_test != y_test_pred)[0]
        correct_indices = np.where(y_test == y_test_pred)[0]

        # Plot incorrectly classified points in black
        self.ax.scatter(X_test[incorrect_indices, 0], X_test[incorrect_indices, 1], c='k', marker='x', s=30, label='Misclassified')

        # Plot correctly classified points
        self.ax.scatter(X_test[correct_indices, 0], X_test[correct_indices, 1], c=y_test[correct_indices], cmap=cmap_bold, edgecolor='k', s=30, marker='o', label='Correctly Classified')

        # Calculate MSE for train and test sets
        y_train_pred = classifier.predict(X_train)
        mse_train = np.mean((y_train - y_train_pred) ** 2)
        mse_test = np.mean((y_test - y_test_pred) ** 2)
        self.mse_train = mse_train
        self.mse_test = mse_test

        # Update the legend
        legend_text = f"Train MSE: {mse_train:.2f}, Test MSE: {mse_test:.2f}"
        self.ax.legend([legend_text], loc='lower left')

        self.canvas.draw()
class Page3(ctk.CTkFrame):
    def __init__(self, master):
        ctk.CTkFrame.__init__(self, master)
        self.header=ctk.CTkFrame(self)
        self.header.pack(fill="x")
        self.back_button = ctk.CTkButton(self.header, text="back", command=master.show_home_page)
        self.back_button.pack(side="left")

        self.page2 = ctk.CTkLabel(self.header, text="classification", font=("Helvetica", 18))
        self.page2.pack(side="left", fill="both", expand=True)



        classify_button = ctk.CTkButton(self, text="Classify", command=self.classify_and_plot)
        classify_button.pack(padx=10, pady=10)

        # Tabs
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill='both', expand=True)

        csv_tab = ctk.CTkFrame(self.notebook)
        self.notebook.add(csv_tab, text="Import CSV")
        self.create_csv_tab(csv_tab)

        self.tab2 = ModelSelectionTab(self.notebook)
        self.notebook.add(self.tab2, text='Model Selection')
        self.matplotlib_figure = MatplotlibFigure(self)


        self.classifier = None


    def create_csv_tab(self, parent):
        # Frame for importing CSV
        csv_frame = ctk.CTkFrame(parent)
        csv_frame.pack(pady=10)

        csv_label = ctk.CTkLabel(csv_frame, text="Select CSV file:")
        csv_label.grid(row=0, column=0, padx=10)

        self.csv_path_var = tk.StringVar()
        csv_entry = ctk.CTkEntry(csv_frame,
                              textvariable=self.csv_path_var,
                              state="readonly",
                              width=40)
        csv_entry.grid(row=0, column=1, padx=10)

        csv_button = ctk.CTkButton(csv_frame,
                                text="Browse",
                                command=self.select_csv_file)
        csv_button.grid(row=0, column=2, padx=10)
    def select_csv_file(self):
        csv_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        self.csv_path_var.set(csv_path)

    def classify_and_plot(self):
        csv_path = self.csv_path_var.get()
        if not csv_path:
            return

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(e)
            return

        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_pca, y_encoded, test_size=0.3, random_state=42)

        selected_model = self.tab2.model_selection.get()
        if selected_model == "Logistic Regression":
            self.classifier = LogisticRegression()
        elif selected_model == "Random Forest":
            self.classifier = RandomForestClassifier()
        elif selected_model == "K-Nearest Neighbors":
            self.classifier = KNeighborsClassifier()
        elif selected_model == "Gaussian Naive Bayes":
            self.classifier = GaussianNB()
        elif selected_model == "Decision Tree":
            self.classifier = DecisionTreeClassifier()

        self.classifier.fit(X_train, y_train)
        y_pred = self.classifier.predict(X_test)

        if not hasattr(self, 'matplotlib_figure'):
            self.matplotlib_figure = MatplotlibFigure(self.root)

        self.matplotlib_figure.update_plot(X_train, y_train, X_test, y_test, self.classifier) 

class ModelSelectionTab(ctk.CTkFrame):

    def __init__(self, master):
        super().__init__(master)

        self.model_selection = tk.StringVar(value="Logistic Regression")

        label = ctk.CTkLabel(self, text="Select Classification Model:")
        label.pack(padx=10, pady=10)

        models = ["Logistic Regression", "Random Forest", "K-Nearest Neighbors", "Gaussian Naive Bayes","Decision Tree"]
        for model in models:
            radio = ctk.CTkRadioButton(self, text=model, variable=self.model_selection, value=model)
            radio.pack(side="top", padx=10, pady=5, anchor=tk.W)



if __name__ == "__main__":
    app = SampleApp()
    app.mainloop()
