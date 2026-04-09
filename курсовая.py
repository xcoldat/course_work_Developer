# -*- coding: utf-8 -*-
"""
Курсовая работа: Информационно-аналитическая система
Вариант 20: Компания разработчик программного обеспечения
Продукты: ERP, CRM, СУБД
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.optimize import linprog
import os

# ----------------------------------------------------------------------
# Глобальные настройки графики
plt.rcParams['font.size'] = 9
plt.style.use('seaborn-v0_8-whitegrid')

# ----------------------------------------------------------------------
# Главное окно приложения
class MainApp(tk.Tk):
    """Основное окно с меню выбора задач."""
    def __init__(self):
        super().__init__()
        self.title("Информационно-аналитическая система (Компания разработчик ПО)")
        self.geometry("550x400")
        self.resizable(False, False)

        # Заголовок
        ttk.Label(self, text="Главное меню", font=("Arial", 14, "bold")).pack(pady=15)

        # Кнопки вызова модулей
        ttk.Button(self, text="1. Оценка технического уровня продукции",
                   command=self.open_evaluation, width=45).pack(pady=5)
        ttk.Button(self, text="2. Прогноз динамики показателей",
                   command=self.open_forecast, width=45).pack(pady=5)
        ttk.Button(self, text="3. Оптимизация распределения ресурсов",
                   command=self.open_optimization, width=45).pack(pady=5)
        ttk.Button(self, text="4. Визуализация данных (дашборд)",
                   command=self.open_dashboard, width=45).pack(pady=5)
        ttk.Button(self, text="Выход", command=self.quit, width=45).pack(pady=20)

    def open_evaluation(self):
        """Открыть окно оценки технического уровня."""
        EvaluationWindow(self)

    def open_forecast(self):
        """Открыть окно прогнозирования."""
        ForecastWindow(self)

    def open_optimization(self):
        """Открыть окно оптимизации."""
        OptimizationWindow(self)

    def open_dashboard(self):
        """Открыть окно дашборда."""
        DashboardWindow(self)


# ----------------------------------------------------------------------
# Модуль 1: Оценка технического уровня
class EvaluationWindow(tk.Toplevel):
    """Окно для оценки качества продукции (ERP, CRM, СУБД)."""
    def __init__(self, master):
        super().__init__(master)
        self.title("Оценка технического уровня продукции")
        self.geometry("900x700")
        self.minsize(700, 500)

        # Данные: список словарей с характеристиками образцов
        self.samples = []          # Каждый элемент: {'name': ..., 'values': [x1, x2, ...]}
        self.characteristics = []  # Названия характеристик (зависят от типа продукта)
        self.etalon_values = []    # Эталонные (идеальные) значения характеристик

        # Переменная для типа продукта
        self.product_type = tk.StringVar(value="ERP")

        # Создание интерфейса
        self.create_widgets()

    def create_widgets(self):
        """Построить элементы управления."""
        # Верхняя панель выбора типа
        top_frame = ttk.Frame(self)
        top_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(top_frame, text="Тип продукции:").pack(side=tk.LEFT)
        type_combo = ttk.Combobox(top_frame, textvariable=self.product_type,
                                  values=["ERP", "CRM", "СУБД"], state="readonly", width=10)
        type_combo.pack(side=tk.LEFT, padx=5)
        type_combo.bind("<<ComboboxSelected>>", self.on_type_changed)

        ttk.Button(top_frame, text="Загрузить из CSV", command=self.load_csv).pack(side=tk.LEFT, padx=10)
        ttk.Button(top_frame, text="Сбросить данные", command=self.reset_data).pack(side=tk.LEFT)

        # Панель ручного ввода
        input_frame = ttk.LabelFrame(self, text="Ручной ввод образца")
        input_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(input_frame, text="Название:").grid(row=0, column=0, padx=5, pady=5, sticky='e')
        self.entry_name = tk.Entry(input_frame, width=20)
        self.entry_name.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(input_frame, text="Характеристики (через запятую):").grid(row=0, column=2, padx=5, pady=5, sticky='e')
        self.entry_values = tk.Entry(input_frame, width=50)
        self.entry_values.grid(row=0, column=3, padx=5, pady=5)

        ttk.Button(input_frame, text="Добавить образец", command=self.add_sample).grid(row=0, column=4, padx=10)

        # Таблица для отображения добавленных образцов
        table_frame = ttk.LabelFrame(self, text="Список образцов")
        table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.tree = ttk.Treeview(table_frame, show='headings', height=6)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=scrollbar.set)

        # Кнопка выполнения оценки
        ttk.Button(self, text="Выполнить оценку и построить диаграммы",
                   command=self.evaluate_and_plot).pack(pady=10)

        # Область для графиков
        self.plot_frame = ttk.Frame(self)
        self.plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Инициализация характеристик по умолчанию
        self.on_type_changed()

    def on_type_changed(self, event=None):
        """Обновить названия характеристик и эталон при смене типа продукта."""
        ptype = self.product_type.get()
        if ptype == "ERP":
            self.characteristics = ["Модульность (кол-во)", "Время отклика (мс)",
                                    "Макс. пользователей", "Надёжность (%)", "Стоимость (тыс. руб)"]
            self.etalon_values = [10, 100, 1000, 99.9, 5000]  # эталонные значения
        elif ptype == "CRM":
            self.characteristics = ["Интеграций с API", "Время синхронизации (с)",
                                    "Объём данных (ГБ)", "Удобство (балл)", "Цена (тыс. руб)"]
            self.etalon_values = [20, 2, 100, 9.5, 3000]
        elif ptype == "СУБД":
            self.characteristics = ["Производительность (tps)", "Поддержка SQL стандартов",
                                    "Масштабируемость", "Безопасность (сертификаты)", "Цена (тыс. руб)"]
            self.etalon_values = [5000, 5, 10, 4, 4000]

        # Обновить заголовки таблицы
        self.update_tree_columns()
        # Очистить данные, т.к. характеристики изменились
        self.reset_data()

    def update_tree_columns(self):
        """Настроить столбцы таблицы в соответствии с характеристиками."""
        self.tree.delete(*self.tree.get_children())
        cols = ["Название"] + self.characteristics
        self.tree["columns"] = cols
        self.tree.column("#0", width=0, stretch=False)
        for col in cols:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=120, anchor='center')

    def reset_data(self):
        """Очистить список образцов и таблицу."""
        self.samples = []
        self.tree.delete(*self.tree.get_children())

    def load_csv(self):
        """Загрузить данные из CSV-файла. Формат: первая колонка - название, остальные - значения характеристик."""
        filepath = filedialog.askopenfilename(
            title="Выберите CSV файл с характеристиками",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not filepath:
            return
        try:
            df = pd.read_csv(filepath)
            if df.shape[1] != len(self.characteristics) + 1:
                messagebox.showerror("Ошибка", f"Файл должен содержать {len(self.characteristics)+1} колонок: название и {len(self.characteristics)} характеристик.")
                return
            # Очищаем текущие данные и добавляем из файла
            self.samples = []
            for _, row in df.iterrows():
                name = str(row.iloc[0])
                values = [float(row.iloc[i+1]) for i in range(len(self.characteristics))]
                self.samples.append({"name": name, "values": values})
            self.refresh_table()
            messagebox.showinfo("Готово", f"Загружено {len(self.samples)} образцов.")
        except Exception as e:
            messagebox.showerror("Ошибка чтения", f"Не удалось прочитать файл:\n{e}")

    def add_sample(self):
        """Добавить образец из ручного ввода."""
        name = self.entry_name.get().strip()
        values_str = self.entry_values.get().strip()
        if not name:
            messagebox.showwarning("Внимание", "Введите название образца.")
            return
        if not values_str:
            messagebox.showwarning("Внимание", "Введите значения характеристик.")
            return
        try:
            values = [float(x.strip()) for x in values_str.split(',')]
            if len(values) != len(self.characteristics):
                messagebox.showerror("Ошибка", f"Должно быть {len(self.characteristics)} значений, введено {len(values)}.")
                return
            self.samples.append({"name": name, "values": values})
            self.refresh_table()
            self.entry_name.delete(0, tk.END)
            self.entry_values.delete(0, tk.END)
        except ValueError:
            messagebox.showerror("Ошибка", "Некорректные числовые значения.")

    def refresh_table(self):
        """Обновить таблицу на основе self.samples."""
        self.tree.delete(*self.tree.get_children())
        for sample in self.samples:
            row = [sample["name"]] + [str(v) for v in sample["values"]]
            self.tree.insert("", tk.END, values=row)

    def evaluate_and_plot(self):
        """Рассчитать относительные показатели, построить радиальную и столбчатую диаграммы."""
        if not self.samples:
            messagebox.showwarning("Внимание", "Нет данных для оценки. Добавьте образцы.")
            return

        # Эталонные значения
        etalon = np.array(self.etalon_values, dtype=float)

        # Для каждого образца считаем относительные показатели:
        # Для характеристик, где больше - лучше: относительное = значение / эталон (ограничено 1.0)
        # Для цены (последняя характеристика) меньше - лучше: относительное = эталон / значение (ограничено 1.0)
        # Здесь предполагаем, что все кроме последней - "больше-лучше", последняя - "цена (меньше-лучше)".
        rel_indicators = []
        names = []
        for sample in self.samples:
            vals = np.array(sample["values"], dtype=float)
            rel = np.zeros(len(vals))
            for i in range(len(vals)-1):  # все кроме последней
                rel[i] = min(vals[i] / etalon[i], 1.0) if etalon[i] != 0 else 1.0
            # цена
            if etalon[-1] != 0:
                rel[-1] = min(etalon[-1] / vals[-1], 1.0)
            else:
                rel[-1] = 1.0
            rel_indicators.append(rel)
            names.append(sample["name"])

        # Расчёт технического уровня (качества) как среднее арифметическое относительных показателей
        quality_scores = [np.mean(rel) for rel in rel_indicators]

        # Очистка предыдущих графиков
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        # Создание фигуры с двумя подграфиками
        fig = plt.Figure(figsize=(9, 4), dpi=100)
        # Радиальная диаграмма
        ax1 = fig.add_subplot(121, projection='polar')
        self._draw_radar(ax1, rel_indicators, names)

        # Столбчатая диаграмма (сортировка по убыванию качества)
        ax2 = fig.add_subplot(122)
        sorted_indices = np.argsort(quality_scores)[::-1]
        sorted_names = [names[i] for i in sorted_indices]
        sorted_scores = [quality_scores[i] for i in sorted_indices]
        bars = ax2.bar(sorted_names, sorted_scores, color='skyblue', edgecolor='navy')
        ax2.set_ylim(0, 1.05)
        ax2.set_ylabel("Технический уровень")
        ax2.set_title("Сравнение технического уровня")
        ax2.tick_params(axis='x', rotation=30)
        # Добавить подписи значений над столбцами
        for bar, score in zip(bars, sorted_scores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f"{score:.3f}", ha='center', va='bottom', fontsize=8)

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _draw_radar(self, ax, rel_indicators, names):
        """Нарисовать радиальную диаграмму с заданными относительными показателями."""
        # Подготовка углов (по количеству характеристик)
        num_vars = len(self.characteristics)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # замыкание

        # Для каждого образца
        for i, rel in enumerate(rel_indicators):
            values = rel.tolist()
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, label=names[i])
            ax.fill(angles, values, alpha=0.1)

        # Настройка осей
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(self.characteristics, fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_title("Относительные показатели", pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=8)


# ----------------------------------------------------------------------
# Модуль 2: Прогноз динамики показателей
class ForecastWindow(tk.Toplevel):
    """Окно прогнозирования методом линейной регрессии."""
    def __init__(self, master):
        super().__init__(master)
        self.title("Прогноз динамики показателей")
        self.geometry("800x600")
        self.minsize(600, 400)

        self.data = None          # DataFrame с колонками 'period', 'value'
        self.forecast_result = None

        self.create_widgets()

    def create_widgets(self):
        # Панель управления
        control_frame = ttk.LabelFrame(self, text="Управление")
        control_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(control_frame, text="Загрузить данные из CSV", command=self.load_data).grid(row=0, column=0, padx=5, pady=5)
        ttk.Label(control_frame, text="Периодов для прогноза:").grid(row=0, column=1, padx=5, pady=5)
        self.periods_entry = tk.Entry(control_frame, width=5)
        self.periods_entry.insert(0, "3")
        self.periods_entry.grid(row=0, column=2, padx=5, pady=5)
        ttk.Button(control_frame, text="Выполнить прогноз", command=self.run_forecast).grid(row=0, column=3, padx=5, pady=5)

        # Таблица для отображения загруженных данных
        table_frame = ttk.LabelFrame(self, text="Исходные данные")
        table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.tree = ttk.Treeview(table_frame, show='headings', height=8)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=scrollbar.set)

        # Область графика
        self.plot_frame = ttk.Frame(self)
        self.plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

    def load_data(self):
        """Загрузить временной ряд из CSV. Ожидаются колонки 'period' и 'value'."""
        filepath = filedialog.askopenfilename(
            title="Выберите CSV файл с данными",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not filepath:
            return
        try:
            df = pd.read_csv(filepath)
            if 'period' not in df.columns or 'value' not in df.columns:
                messagebox.showerror("Ошибка", "Файл должен содержать колонки 'period' и 'value'.")
                return
            self.data = df[['period', 'value']].copy()
            # Приведение period к строке или числу (для оси X)
            self.refresh_table()
        except Exception as e:
            messagebox.showerror("Ошибка чтения", str(e))

    def refresh_table(self):
        """Отобразить данные в таблице."""
        self.tree.delete(*self.tree.get_children())
        if self.data is None:
            return
        self.tree["columns"] = ["period", "value"]
        self.tree.column("#0", width=0, stretch=False)
        self.tree.heading("period", text="Период")
        self.tree.heading("value", text="Значение")
        self.tree.column("period", width=150, anchor='center')
        self.tree.column("value", width=150, anchor='center')
        for _, row in self.data.iterrows():
            self.tree.insert("", tk.END, values=(row['period'], row['value']))

    def run_forecast(self):
        """Выполнить прогноз линейной регрессией."""
        if self.data is None or self.data.empty:
            messagebox.showwarning("Нет данных", "Сначала загрузите данные.")
            return
        try:
            n_periods = int(self.periods_entry.get())
            if n_periods <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Ошибка", "Введите положительное целое число периодов.")
            return

        # Подготовка числовых индексов
        x = np.arange(len(self.data)).reshape(-1, 1)
        y = self.data['value'].values

        # Линейная регрессия
        coeffs = np.polyfit(x.flatten(), y, 1)
        slope, intercept = coeffs

        # Прогноз
        last_idx = len(self.data)
        future_x = np.arange(last_idx, last_idx + n_periods)
        future_y = slope * future_x + intercept

        # Подготовка для визуализации
        all_x = np.arange(last_idx + n_periods)
        trend_line = slope * all_x + intercept

        # Очистка области графика
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        fig = plt.Figure(figsize=(8, 4), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(x.flatten(), y, 'bo-', label='Исторические данные')
        ax.plot(all_x, trend_line, 'g--', label='Тренд')
        ax.plot(future_x, future_y, 'ro', label='Прогноз')
        ax.set_xlabel("Период (индекс)")
        ax.set_ylabel("Значение показателя")
        ax.set_title("Прогноз динамики показателя")
        ax.legend()
        ax.grid(True)

        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Сохраняем результат для возможного использования в дашборде
        self.forecast_result = {"future_x": future_x, "future_y": future_y}

        # Вывод численного прогноза
        pred_text = "\n".join([f"Период {i+1}: {val:.2f}" for i, val in enumerate(future_y)])
        messagebox.showinfo("Результаты прогноза", f"Прогноз на {n_periods} периодов:\n{pred_text}")


# ----------------------------------------------------------------------
# Модуль 3: Оптимизация распределения ресурсов
class OptimizationWindow(tk.Toplevel):
    """Окно решения задачи линейного программирования."""
    def __init__(self, master):
        super().__init__(master)
        self.title("Оптимизация распределения ресурсов")
        self.geometry("700x600")
        self.minsize(500, 400)

        self.create_widgets()

    def create_widgets(self):
        # Пояснение
        info_text = """Задача: распределить бюджет между тремя проектами (ERP, CRM, СУБД),
чтобы максимизировать суммарную прибыль при ограничениях на минимальные
и максимальные затраты, а также общий бюджет."""
        ttk.Label(self, text=info_text, wraplength=600, justify='left').pack(pady=10)

        # Ручной ввод параметров
        param_frame = ttk.LabelFrame(self, text="Параметры задачи")
        param_frame.pack(fill=tk.X, padx=20, pady=10)

        # Общий бюджет
        ttk.Label(param_frame, text="Общий бюджет (тыс. руб):").grid(row=0, column=0, padx=5, pady=5, sticky='e')
        self.budget_entry = tk.Entry(param_frame, width=10)
        self.budget_entry.insert(0, "1000")
        self.budget_entry.grid(row=0, column=1, padx=5, pady=5, sticky='w')

        # Прибыль на единицу затрат для каждого проекта
        ttk.Label(param_frame, text="Прибыльность (ERP, CRM, СУБД):").grid(row=1, column=0, padx=5, pady=5, sticky='e')
        self.profit_entry = tk.Entry(param_frame, width=25)
        self.profit_entry.insert(0, "5, 4, 6")
        self.profit_entry.grid(row=1, column=1, padx=5, pady=5, sticky='w')

        # Минимальные вложения
        ttk.Label(param_frame, text="Мин. вложения (ERP, CRM, СУБД):").grid(row=2, column=0, padx=5, pady=5, sticky='e')
        self.min_entry = tk.Entry(param_frame, width=25)
        self.min_entry.insert(0, "100, 100, 100")
        self.min_entry.grid(row=2, column=1, padx=5, pady=5, sticky='w')

        # Максимальные вложения
        ttk.Label(param_frame, text="Макс. вложения (ERP, CRM, СУБД):").grid(row=3, column=0, padx=5, pady=5, sticky='e')
        self.max_entry = tk.Entry(param_frame, width=25)
        self.max_entry.insert(0, "600, 600, 600")
        self.max_entry.grid(row=3, column=1, padx=5, pady=5, sticky='w')

        ttk.Button(self, text="Решить задачу", command=self.solve).pack(pady=10)

        # Область для вывода результатов
        result_frame = ttk.LabelFrame(self, text="Результаты")
        result_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        self.result_text = tk.Text(result_frame, height=12, wrap=tk.WORD)
        self.result_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def solve(self):
        """Решить задачу линейного программирования и вывести результат."""
        try:
            budget = float(self.budget_entry.get())
            profits = [float(x.strip()) for x in self.profit_entry.get().split(',')]
            mins = [float(x.strip()) for x in self.min_entry.get().split(',')]
            maxs = [float(x.strip()) for x in self.max_entry.get().split(',')]

            if len(profits) != 3 or len(mins) != 3 or len(maxs) != 3:
                raise ValueError("Введите ровно 3 значения через запятую.")

        except ValueError as e:
            messagebox.showerror("Ошибка ввода", f"Некорректные числовые данные:\n{e}")
            return

        # Целевая функция: минимизация отрицательной прибыли (т.к. linprog минимизирует)
        c = [-p for p in profits]  # коэффициенты целевой функции

        # Ограничения:
        # 1. x1 + x2 + x3 <= budget
        A_ub = [[1, 1, 1]]
        b_ub = [budget]

        # Границы переменных
        bounds = [(mins[i], maxs[i]) for i in range(3)]

        # Решение
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

        self.result_text.delete(1.0, tk.END)
        if res.success:
            x = res.x
            total_profit = -res.fun
            result_str = "Оптимальное распределение бюджета (тыс. руб):\n"
            result_str += f"  ERP:  {x[0]:.2f}\n"
            result_str += f"  CRM:  {x[1]:.2f}\n"
            result_str += f"  СУБД: {x[2]:.2f}\n"
            result_str += f"Суммарные затраты: {sum(x):.2f} (из {budget})\n"
            result_str += f"Максимальная прибыль: {total_profit:.2f}\n"
            self.result_text.insert(tk.END, result_str)
        else:
            self.result_text.insert(tk.END, f"Решение не найдено.\nСообщение: {res.message}")


# ----------------------------------------------------------------------
# Модуль 4: Дашборд (визуализация)
class DashboardWindow(tk.Toplevel):
    """Обобщённая визуализация результатов всех модулей."""
    def __init__(self, master):
        super().__init__(master)
        self.title("Визуализация данных (дашборд)")
        self.geometry("900x600")
        self.minsize(700, 500)

        self.create_widgets()

    def create_widgets(self):
        ttk.Label(self, text="Обобщённая визуализация", font=('Arial', 12, 'bold')).pack(pady=5)

        # Кнопка для обновления (построения демонстрационных графиков)
        ttk.Button(self, text="Показать демо-графики", command=self.show_demo).pack(pady=5)

        self.plot_frame = ttk.Frame(self)
        self.plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # При создании сразу показать демо
        self.show_demo()

    def show_demo(self):
        """Создать три демонстрационных графика, иллюстрирующих работу системы."""
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        fig = plt.Figure(figsize=(9, 6), dpi=100)
        # График 1: пример радиальной диаграммы (оценка)
        ax1 = fig.add_subplot(221, projection='polar')
        self._demo_radar(ax1)

        # График 2: пример прогноза
        ax2 = fig.add_subplot(222)
        self._demo_forecast(ax2)

        # График 3: пример оптимизации (столбчатая диаграмма распределения)
        ax3 = fig.add_subplot(223)
        self._demo_optimization(ax3)

        # График 4: общая информация
        ax4 = fig.add_subplot(224)
        ax4.axis('off')
        text = "Информационно-аналитическая система\nВариант 20: Компания разработчик ПО\n\n"
        text += "Функции:\n- Оценка качества ERP/CRM/СУБД\n- Прогноз показателей\n- Оптимизация ресурсов"
        ax4.text(0.5, 0.5, text, ha='center', va='center', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _demo_radar(self, ax):
        """Демонстрационная радиальная диаграмма."""
        categories = ['Модульность', 'Отклик', 'Пользователи', 'Надёжность', 'Цена']
        N = len(categories)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]

        # Пример двух образцов
        values1 = [0.9, 0.8, 0.7, 0.95, 0.6]
        values2 = [0.7, 0.9, 0.6, 0.85, 0.8]
        values1 += values1[:1]
        values2 += values2[:1]

        ax.plot(angles, values1, 'o-', linewidth=2, label='ERP Сириус')
        ax.fill(angles, values1, alpha=0.1)
        ax.plot(angles, values2, 'o-', linewidth=2, label='ERP Вектор')
        ax.fill(angles, values2, alpha=0.1)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title("Пример оценки качества ERP")
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))

    def _demo_forecast(self, ax):
        """Демонстрационный линейный прогноз."""
        x = np.arange(6)
        y = np.array([100, 115, 130, 145, 160, 175]) + np.random.normal(0, 5, 6)
        ax.plot(x, y, 'bo-', label='История')
        coeffs = np.polyfit(x, y, 1)
        trend = np.polyval(coeffs, np.arange(9))
        ax.plot(np.arange(9), trend, 'g--', label='Тренд')
        ax.plot([6,7,8], trend[6:], 'ro', label='Прогноз')
        ax.set_xlabel("Месяц")
        ax.set_ylabel("Продажи")
        ax.set_title("Прогноз продаж")
        ax.legend()
        ax.grid(True)

    def _demo_optimization(self, ax):
        """Демонстрационная диаграмма распределения бюджета."""
        projects = ['ERP', 'CRM', 'СУБД']
        allocation = [350, 250, 400]
        bars = ax.bar(projects, allocation, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax.set_ylabel("Бюджет, тыс. руб")
        ax.set_title("Оптимальное распределение бюджета")
        for bar, val in zip(bars, allocation):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    str(val), ha='center', va='bottom')


# ----------------------------------------------------------------------
# Точка входа
if __name__ == "__main__":
    app = MainApp()
    app.mainloop()