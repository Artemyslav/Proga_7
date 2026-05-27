import tkinter as tk
from tkinter import messagebox, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

df_raw = None
df_work = None
fig = None
canvas = None
current_chart = "line"

def load_and_preprocess_data(filepath='data.csv'):
    global df_raw, df_work
    try:
        df_raw = pd.read_csv(filepath)
        
        if 'ts' in df_raw.columns:
            df_raw['analysis_date'] = pd.to_datetime(df_raw['ts'], unit='s')
        elif 'analysis_date' in df_raw.columns:
            df_raw['analysis_date'] = pd.to_datetime(df_raw['analysis_date'])
        
        if 'glu' in df_raw.columns:
            df_raw['glu'] = df_raw['glu'].clip(20, 500)
        
        if 'wbc' in df_raw.columns:
            df_raw.loc[df_raw['wbc'] < 0, 'wbc'] = 0
        
        if 'hgb' in df_raw.columns:
            hgb_95 = df_raw['hgb'].quantile(0.95)
            df_raw['hgb'] = df_raw['hgb'].clip(upper=hgb_95)
        
        if 'age' in df_raw.columns:
            df_raw['age'] = df_raw['age'].clip(0, 120)
            if 'patient_age_group' in df_raw.columns:
                df_raw['patient_age_group'] = pd.cut(
                    df_raw['age'], 
                    bins=[0, 18, 60, 121],
                    labels=['дети', 'взрослые', 'пожилые'], 
                    right=False
                )
        
        num_cols = ['hgb', 'wbc', 'plt', 'age']
        for col in num_cols:
            if col in df_raw.columns:
                df_raw[col].fillna(df_raw[col].median(), inplace=True)
        
        cat_cols = ['lab_type', 'region', 'patient_age_group']
        for col in cat_cols:
            if col in df_raw.columns:
                df_raw[col] = df_raw[col].astype('category')
        
        df_work = df_raw.copy()
        print("Данные загружены и очищены")
        return True
        
    except FileNotFoundError:
        messagebox.showerror("Ошибка", f"Файл {filepath} не найден")
        return False
    except Exception as e:
        messagebox.showerror("Ошибка", f"Не удалось загрузить данные: {e}")
        return False

def main():
    global df_raw, df_work, fig, canvas, current_chart

    if not load_and_preprocess_data('data.csv'):
        return

    root = tk.Tk()
    root.title("Дашборд: Анализ крови (Вариант 32)")
    root.geometry("1300x800")
    root.configure(bg="#f0f2f5")

    filter_region = tk.StringVar(value="Все")
    filter_lab_type = tk.StringVar(value="Все")
    filter_age_group = tk.StringVar(value="Все")
    smoothing_window = tk.IntVar(value=30)

    sns.set_theme(style="darkgrid")
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig = plt.Figure(figsize=(11, 6.5), dpi=100)
    plot_frame = tk.Frame(root, bg="white", relief=tk.SUNKEN, bd=1)
    plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    toolbar = NavigationToolbar2Tk(canvas, plot_frame)
    toolbar.update()
    toolbar.pack(side=tk.TOP, fill=tk.X)

    def apply_filters():
        global df_work
        filtered_df = df_raw.copy()
        if filter_region.get() != "Все":
            filtered_df = filtered_df[filtered_df['region'] == filter_region.get()]
        if filter_lab_type.get() != "Все":
            filtered_df = filtered_df[filtered_df['lab_type'] == filter_lab_type.get()]
        if filter_age_group.get() != "Все":
            filtered_df = filtered_df[filtered_df['patient_age_group'] == filter_age_group.get()]
        df_work = filtered_df

    def clear_figure():
        fig.clear()

    def plot_line():
        if df_work.empty:
            messagebox.showwarning("Нет данных", "Нет данных с выбранными фильтрами")
            return
        clear_figure()
        ax = fig.add_subplot(111)
        plot_df = df_work.sort_values('analysis_date').copy()
        window = smoothing_window.get()
        plot_df['glu_ma'] = plot_df['glu'].rolling(window=window, min_periods=1).mean()
        ax.plot(plot_df['analysis_date'], plot_df['glu_ma'], color='b', linewidth=1.5)
        ax.set_title(f'Скользящее среднее глюкозы (окно = {window})', fontsize=12)
        ax.set_xlabel('Дата анализа')
        ax.set_ylabel('Глюкоза, ммоль/л')
        ax.grid(True, linestyle='--', alpha=0.6)
        plt.xticks(rotation=45)
        fig.tight_layout()
        canvas.draw_idle()

    def plot_bar():
        if df_work.empty:
            messagebox.showwarning("Нет данных", "Нет данных с выбранными фильтрами")
            return
        clear_figure()
        ax = fig.add_subplot(111)
        
        grouped = df_work.groupby(['patient_age_group', 'region'], observed=False)['glu'].mean().reset_index()
        
        sns.barplot(data=grouped, x='patient_age_group', y='glu', hue='region', 
                    ax=ax, palette='Set2', errorbar=None)
        
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f', fontsize=9, padding=2)
        
        ax.set_title('Средний уровень глюкозы: возрастная группа × регион', fontsize=12)
        ax.set_xlabel('Возрастная группа')
        ax.set_ylabel('Средняя глюкоза, ммоль/л')
        ax.legend(title='Регион', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)
        fig.tight_layout()
        canvas.draw_idle()

    def plot_scatter():
        if df_work.empty:
            messagebox.showwarning("Нет данных", "Нет данных с выбранными фильтрами")
            return
        clear_figure()
        ax = fig.add_subplot(111)
        sns.scatterplot(data=df_work, x='age', y='glu', hue='lab_type', 
                        alpha=0.6, ax=ax, palette='Set1', s=30)
        ax.set_title('Зависимость глюкозы от возраста пациентов', fontsize=12)
        ax.set_xlabel('Возраст, лет')
        ax.set_ylabel('Глюкоза, ммоль/л')
        ax.legend(title='Тип лаборатории', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.6)
        fig.tight_layout()
        canvas.draw_idle()

    def plot_histogram():
        if df_work.empty:
            messagebox.showwarning("Нет данных", "Нет данных с выбранными фильтрами")
            return
        clear_figure()
        ax = fig.add_subplot(111)
        
        sns.histplot(data=df_work, x='glu', hue='patient_age_group', 
                     kde=True, ax=ax, alpha=0.5, bins=30, element='step')
        
        ax.axvline(x=7.8, color='red', linestyle='--', linewidth=2, label='Норма (7.8 ммоль/л)')
        
        ax.set_title('Распределение уровня глюкозы по возрастным группам', fontsize=12)
        ax.set_xlabel('Глюкоза, ммоль/л')
        ax.set_ylabel('Плотность / Частота')
        ax.legend(title='Возрастная группа', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.6)
        fig.tight_layout()
        canvas.draw_idle()

    def refresh_dashboard():
        apply_filters()
        if current_chart == "line":
            plot_line()
        elif current_chart == "bar":
            plot_bar()
        elif current_chart == "scatter":
            plot_scatter()
        elif current_chart == "histogram":
            plot_histogram()

    def export_plot():
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG файлы", "*.png"), ("PDF файлы", "*.pdf")]
        )
        if path:
            try:
                fig.savefig(path, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Экспорт", f"График сохранён: {path}")
            except Exception as e:
                messagebox.showerror("Ошибка", str(e))

    def set_chart(chart_type):
        global current_chart
        current_chart = chart_type
        refresh_dashboard()

    def change_smoothing_window(*args):
        if current_chart == "line":
            refresh_dashboard()

    filter_frame = tk.LabelFrame(root, text="Фильтры данных", bg="#f0f2f5", font=('Arial', 10, 'bold'), padx=10, pady=5)
    filter_frame.pack(fill=tk.X, padx=10, pady=5)

    tk.Label(filter_frame, text="Регион:", bg="#f0f2f5").grid(row=0, column=0, padx=5, pady=5, sticky='w')
    region_menu = tk.OptionMenu(filter_frame, filter_region, "Все")
    region_menu.grid(row=0, column=1, padx=5, pady=5, sticky='ew')

    tk.Label(filter_frame, text="Тип лаборатории:", bg="#f0f2f5").grid(row=0, column=2, padx=5, pady=5, sticky='w')
    lab_type_menu = tk.OptionMenu(filter_frame, filter_lab_type, "Все")
    lab_type_menu.grid(row=0, column=3, padx=5, pady=5, sticky='ew')

    tk.Label(filter_frame, text="Возрастная группа:", bg="#f0f2f5").grid(row=0, column=4, padx=5, pady=5, sticky='w')
    age_group_menu = tk.OptionMenu(filter_frame, filter_age_group, "Все")
    age_group_menu.grid(row=0, column=5, padx=5, pady=5, sticky='ew')

    filter_frame.columnconfigure(1, weight=1)
    filter_frame.columnconfigure(3, weight=1)
    filter_frame.columnconfigure(5, weight=1)

    if 'region' in df_raw.columns:
        regions = ["Все"] + sorted(df_raw['region'].cat.categories.tolist())
        region_menu['menu'].delete(0, 'end')
        for r in regions:
            region_menu['menu'].add_command(label=r, command=tk._setit(filter_region, r))

    if 'lab_type' in df_raw.columns:
        lab_types = ["Все"] + sorted(df_raw['lab_type'].cat.categories.tolist())
        lab_type_menu['menu'].delete(0, 'end')
        for lt in lab_types:
            lab_type_menu['menu'].add_command(label=lt, command=tk._setit(filter_lab_type, lt))

    if 'patient_age_group' in df_raw.columns:
        age_groups = ["Все"] + sorted(df_raw['patient_age_group'].cat.categories.tolist())
        age_group_menu['menu'].delete(0, 'end')
        for ag in age_groups:
            age_group_menu['menu'].add_command(label=ag, command=tk._setit(filter_age_group, ag))

    control_frame = tk.Frame(root, bg="#f0f2f5")
    control_frame.pack(fill=tk.X, padx=10, pady=5)

    tk.Button(control_frame, text="Линейный график", command=lambda: set_chart("line"), 
              width=14, bg="#4CAF50", fg="white").pack(side=tk.LEFT, padx=4)
    tk.Button(control_frame, text="Столбчатая диаграмма", command=lambda: set_chart("bar"), 
              width=16, bg="#2196F3", fg="white").pack(side=tk.LEFT, padx=4)
    tk.Button(control_frame, text="Точечная диаграмма", command=lambda: set_chart("scatter"), 
              width=14, bg="#FF9800", fg="white").pack(side=tk.LEFT, padx=4)
    tk.Button(control_frame, text="Гистограмма", command=lambda: set_chart("histogram"), 
              width=12, bg="#9C27B0", fg="white").pack(side=tk.LEFT, padx=4)

    tk.Label(control_frame, text=" | ", bg="#f0f2f5", font=('Arial', 12)).pack(side=tk.LEFT, padx=10)
    tk.Label(control_frame, text="Окно сглаживания:", bg="#f0f2f5").pack(side=tk.LEFT, padx=5)
    tk.Spinbox(control_frame, from_=5, to=100, textvariable=smoothing_window, 
               width=5, command=change_smoothing_window).pack(side=tk.LEFT, padx=5)

    tk.Button(control_frame, text="Обновить", command=refresh_dashboard, 
              width=12, bg="#607D8B", fg="white").pack(side=tk.RIGHT, padx=4)
    tk.Button(control_frame, text="Экспорт", command=export_plot, 
              width=12, bg="#795548", fg="white").pack(side=tk.RIGHT, padx=4)

    filter_region.trace('w', lambda *args: refresh_dashboard())
    filter_lab_type.trace('w', lambda *args: refresh_dashboard())
    filter_age_group.trace('w', lambda *args: refresh_dashboard())

    refresh_dashboard()
    root.mainloop()

if __name__ == "__main__":
    main()