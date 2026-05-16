"""
Информационно-аналитическая система: SaaS-метрики
Архитектура: MVC, Strategy Pattern
Задачи: Композитный индекс → Прогнозная модель выручки → Сценарный анализ (What-If)
Стек: Python 3.8+, pandas, numpy, matplotlib, tkinter, scikit-learn
"""
import os, sys, json, warnings, datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

# ==================== КОНФИГУРАЦИЯ ====================
FEATURES = [
    "avg_subscription_price",
    "active_users_count",
    "is_major_update",
    "support_response_time",
    "feature_requests_satisfied_percent"
]
# 1 = больше лучше, -1 = меньше лучше
DIRECTIONS = [1, 1, 1, -1, 1]  
TARGET = "monthly_subscription_revenue"
CONTROLLABLE = ["support_response_time", "feature_requests_satisfied_percent"]

RU_NAMES = {
    "avg_subscription_price": "Цена подписки (руб.)",
    "active_users_count": "Активные пользователи (тыс.)",
    "is_major_update": "Крупное обновление (0/1)",
    "support_response_time": "Время ответа поддержки (мин.)",
    "feature_requests_satisfied_percent": "Выполнение запросов (%)",
    TARGET: "Выручка от подписок (тыс. руб.)"
}

# ==================== ЯДРА РАСЧЁТОВ (Strategy) ====================
class IndexEngine:
    @staticmethod
    def normalize(df: pd.DataFrame, features: list, directions: list) -> pd.DataFrame:
        df_norm = df.copy()
        for i, col in enumerate(features):
            mn, mx = df[col].min(), df[col].max()
            if mx == mn:
                df_norm[col] = 1.0
                continue
            vals = (df[col] - mn) / (mx - mn)
            if directions[i] == -1:
                vals = 1.0 - vals
            df_norm[col] = vals
        return df_norm

    @staticmethod
    def calculate_composite_index(df_norm: pd.DataFrame, features: list, weights: np.ndarray) -> pd.Series:
        return df_norm[features] @ weights

class ForecastEngine:
    @staticmethod
    def train_model(df: pd.DataFrame, features: list, target: str) -> dict:
        X, y = df[features].values, df[target].values
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        return {
            "model": model,
            "r2": r2,
            "mae": mae,
            "coeffs": {f: c for f, c in zip(features, model.coef_)},
            "intercept": model.intercept_,
            "actual": y,
            "predicted": y_pred
        }

# ==================== ИНТЕРФЕЙС И КОНТРОЛЛЕР (MVC) ====================
class SaaSAnalyticsApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SaaS Аналитика: Индекс → Модель → Сценарии")
        self.root.geometry("1280x950")
        self.df = self.df_norm = None
        self.index_scores = None
        self.model_info = None
        self.weights = np.ones(len(FEATURES)) / len(FEATURES)

        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        self._build_ui()

    def _on_closing(self):
        plt.close('all')
        self.root.destroy()
        self.root.quit()
        sys.exit(0)

    def _build_ui(self):
        # Настройка стиля для таблицы
        style = ttk.Style()
        style.configure("Treeview", font=('Segoe UI', 9), rowheight=25)
        style.configure("Treeview.Heading", font=('Segoe UI', 9, 'bold'))

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(expand=True, fill='both', padx=10, pady=10)
        self.tabs = [ttk.Frame(self.notebook) for _ in range(3)]
        
        titles = [
            "Оценка композитного индекса",
            "Прогнозная модель показателей",
            "Сценарный анализ улучшений"
        ]
        for t, title in zip(self.tabs, titles):
            self.notebook.add(t, text=title)
            
        self._setup_tab_index()
        self._setup_tab_forecast()
        self._setup_tab_scenario()

    # ==================== ВКЛАДКА 1: ИНДЕКС ====================
    def _setup_tab_index(self):
        f = ttk.Frame(self.tabs[0])
        f.pack(fill='both', expand=True, padx=10, pady=5)
        
        top = ttk.Frame(f)
        top.pack(fill='x', pady=5)
        ttk.Button(top, text="📂 Загрузить данные", command=self._load_data).pack(side='left', padx=5)
        ttk.Button(top, text="▶ Рассчитать индекс", command=self._calc_index).pack(side='left', padx=5)
        ttk.Button(top, text="📤 Экспорт JSON", command=self._export_json).pack(side='right', padx=5)
        self.lbl_status = ttk.Label(top, text="Ожидание данных...", foreground="gray")
        self.lbl_status.pack(side='right', padx=10)

        # Таблица со скроллбаром
        tree_frame = ttk.Frame(f)
        tree_frame.pack(fill='both', expand=True, pady=5)
        
        self.tree = ttk.Treeview(tree_frame, show='headings', height=10)
        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        self.tree.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')
        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)

        # Графики
        self.fig_idx, (self.ax_r, self.ax_b) = plt.subplots(1, 2, figsize=(10, 4))
        self.canvas_idx = FigureCanvasTkAgg(self.fig_idx, f)
        self.canvas_idx.get_tk_widget().pack(fill='both', expand=True, pady=5)

    # ==================== ВКЛАДКА 2: ПРОГНОЗ ====================
    def _setup_tab_forecast(self):
        f = ttk.LabelFrame(self.tabs[1], text="Регрессионная модель: Метрики → Выручка")
        f.pack(fill='both', expand=True, padx=10, pady=5)
        
        info_frame = ttk.Frame(f)
        info_frame.pack(fill='x', pady=10)
        self.lbl_model = ttk.Label(info_frame, text="Загрузите CSV для обучения модели", foreground="gray", justify='left')
        self.lbl_model.pack(fill='x', padx=10)
        
        ttk.Button(f, text="🔄 Обучить / Обновить модель", command=self._train_and_show_forecast).pack(pady=5)

        self.fig_fc, self.ax_fc = plt.subplots(figsize=(9, 4))
        self.canvas_fc = FigureCanvasTkAgg(self.fig_fc, f)
        self.canvas_fc.get_tk_widget().pack(fill='both', expand=True, padx=5, pady=5)

    # ==================== ВКЛАДКА 3: СЦЕНАРИИ ====================
    def _setup_tab_scenario(self):
        f = ttk.LabelFrame(self.tabs[2], text="Симулятор: что будет, если улучшить управляемые метрики?")
        f.pack(fill='both', expand=True, padx=10, pady=5)
        
        base_frame = ttk.Frame(f)
        base_frame.pack(fill='x', pady=5)
        ttk.Label(base_frame, text="Базовые значения (средние по выборке):").pack(anchor='w', padx=10)
        
        self.entries = {}
        ctrl_frame = ttk.Frame(f)
        ctrl_frame.pack(fill='x', pady=5, padx=10)
        
        for i, met in enumerate(CONTROLLABLE):
            row = ttk.Frame(ctrl_frame)
            row.pack(fill='x', pady=3)
            ttk.Label(row, text=f"{RU_NAMES[met]}:").pack(side='left')
            ent = ttk.Entry(row, width=10)
            ent.pack(side='left', padx=10)
            self.entries[met] = ent
            ttk.Label(row, text="(мин–макс:)").pack(side='left', padx=5)
            rng = ttk.Label(row, text="")
            rng.pack(side='left')
            self.entries[f"{met}_range"] = rng

        ttk.Button(f, text="▶ Рассчитать сценарий", command=self._run_scenario).pack(pady=10)
        self.lbl_scenario = ttk.Label(f, text="Введите параметры и нажмите кнопку", foreground="blue", justify='left')
        self.lbl_scenario.pack(pady=5)
        
        self.fig_sc, self.ax_sc = plt.subplots(figsize=(9, 3))
        self.canvas_sc = FigureCanvasTkAgg(self.fig_sc, f)
        self.canvas_sc.get_tk_widget().pack(fill='both', expand=True)

    # ==================== ОБРАБОТЧИКИ ====================
    def _load_data(self):
        fp = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not fp: return

        # Сброс состояния всех вкладок
        self.df = self.df_norm = None
        self.index_scores = None
        self.model_info = None

        self.tree.delete(*self.tree.get_children())

        self.ax_r.clear()
        self.ax_b.clear()
        self.ax_fc.clear()
        self.ax_sc.clear()

        self.lbl_status.config(text="Ожидание данных...", foreground="gray")
        self.lbl_model.config(text="Загрузите CSV для обучения модели", foreground="gray")
        self.lbl_scenario.config(text="Введите параметры и нажмите кнопку", foreground="blue")

        self.canvas_idx.draw()
        self.canvas_fc.draw()
        self.canvas_sc.draw()

        try:
            df = pd.read_csv(fp, sep=None, engine='python', encoding='utf-8-sig')
            missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
            if missing:
                raise ValueError(f"❌ Отсутствуют столбцы: {', '.join(missing)}")
            
            self.df = df[FEATURES + [TARGET]].copy()
            
            # Заполняем поля с точностью до 3 знаков
            for met in CONTROLLABLE:
                self.entries[met].delete(0, tk.END)
                self.entries[met].insert(0, f"{self.df[met].mean():.3f}")
                self.entries[f"{met}_range"].config(text=f"{self.df[met].min():.1f} – {self.df[met].max():.1f}")
                
            self._refresh_table()
            self.lbl_status.config(text=f"✅ Загружено {len(self.df)} наблюдений", foreground="green")
            messagebox.showinfo("Успех", "Датасет загружен. Рассчитайте индекс или обучите модель.")
        except Exception as e:
            messagebox.showerror("Ошибка загрузки", str(e))

    def _refresh_table(self):
        if self.df is None: return
        self.tree.delete(*self.tree.get_children())
        cols = list(self.df.columns)
        self.tree["columns"] = cols
        
        col_widths = {
            "avg_subscription_price": 130,
            "active_users_count": 130,
            "is_major_update": 90,
            "support_response_time": 140,
            "feature_requests_satisfied_percent": 140,
            "monthly_subscription_revenue": 150
        }
        
        for c in cols:
            self.tree.heading(c, text=RU_NAMES.get(c, c))
            width = col_widths.get(c, 120)
            self.tree.column(c, width=width, anchor='center')
        
        for _, row in self.df.iterrows():
            values = []
            for i, col in enumerate(cols):
                val = row[col]
                if isinstance(val, (int, float)):
                    values.append(f"{val:.1f}" if col != "is_major_update" else str(int(val)))
                else:
                    values.append(str(val))
            self.tree.insert("", "end", values=values)

    def _calc_index(self):
        if self.df is None:
            messagebox.showwarning("Внимание", "Сначала загрузите CSV")
            return
            
        self.df_norm = IndexEngine.normalize(self.df, FEATURES, DIRECTIONS)
        self.index_scores = IndexEngine.calculate_composite_index(self.df_norm, FEATURES, self.weights)
        
        # --- Радарная диаграмма ---
        self.ax_r.clear()
        angles = np.linspace(0, 2*np.pi, len(FEATURES), endpoint=False).tolist()
        angles += angles[:1]
        self.ax_r.set_xticks(angles[:-1])
        
        chart_labels = []
        for f in FEATURES:
            name = RU_NAMES[f]
            if len(name) > 18:
                parts = name.split()
                mid = len(parts) // 2
                chart_labels.append(" ".join(parts[:mid]) + "\n" + " ".join(parts[mid:]))
            else:
                chart_labels.append(name)

        self.ax_r.set_xticklabels(chart_labels, fontsize=9, ha='center', va='top')
        self.ax_r.set_ylim(0, 1.15)
        self.ax_r.tick_params(axis='x', pad=15)
        
        for idx in range(min(6, len(self.df_norm))):
            row = self.df_norm.iloc[idx]
            v = row[FEATURES].tolist() + [row[FEATURES].tolist()[0]]
            self.ax_r.plot(angles, v, 'o-', linewidth=2, label=f"Наблюдение {idx}")
            self.ax_r.fill(angles, v, alpha=0.2)
        self.ax_r.legend(loc='upper right', fontsize=7, bbox_to_anchor=(1.25, 1.15))
        self.ax_r.set_title("Профили метрик (нормализованные)")

        # --- Столбчатая диаграмма ---
        self.ax_b.clear()
        df_plot = pd.DataFrame({"№": self.df.index, "Индекс": self.index_scores.values}).sort_values("Индекс", ascending=False)
        n_rows = len(df_plot)
        step = max(1, n_rows // 14)
        
        self.ax_b.barh(df_plot["№"].astype(str), df_plot["Индекс"], color=plt.cm.viridis(np.linspace(0.3, 0.9, n_rows)))
        self.ax_b.set_xlabel("Композитный индекс")
        self.ax_b.set_title("Ранжирование наблюдений")
        self.ax_b.set_xlim(0, 1)
        
        self.ax_b.set_yticks(range(0, n_rows, step))
        self.ax_b.set_yticklabels(df_plot["№"].astype(str).iloc[::step], fontsize=7)
        
        for i, v in enumerate(df_plot["Индекс"]):
            if i % step == 0:
                self.ax_b.text(v + 0.01, i, f"{v:.3f}", va='center', fontsize=7)
            
        self.fig_idx.tight_layout(pad=2.0)
        self.canvas_idx.draw()
        self.lbl_status.config(text="✅ Индекс рассчитан", foreground="black")

    def _train_and_show_forecast(self):
        if self.df is None:
            messagebox.showwarning("Внимание", "Загрузите данные")
            return
        self.model_info = ForecastEngine.train_model(self.df, FEATURES, TARGET)
        
        txt = f"📊 Модель: LinearRegression (scikit-learn)\n"
        txt += f"R² = {self.model_info['r2']:.3f} | MAE = {self.model_info['mae']:.2f} тыс. руб.\n"
        txt += "Влияние метрик на выручку (коэффициенты):\n"
        sorted_coeffs = sorted(self.model_info['coeffs'].items(), key=lambda x: abs(x[1]), reverse=True)
        for f, c in sorted_coeffs:
            txt += f"  • {RU_NAMES[f]}: {c:+.2f}\n"
        self.lbl_model.config(text=txt, foreground="black")
        
        self.ax_fc.clear()
        self.ax_fc.scatter(self.model_info['actual'], self.model_info['predicted'], alpha=0.6, color="#3498db", s=40)
        min_v, max_v = min(self.model_info['actual']), max(self.model_info['actual'])
        self.ax_fc.plot([min_v, max_v], [min_v, max_v], 'r--', linewidth=2, label="Идеальное совпадение")
        self.ax_fc.set_xlabel("Фактическая выручка")
        self.ax_fc.set_ylabel("Прогноз модели")
        self.ax_fc.set_title(f"Факт vs Прогноз (R²={self.model_info['r2']:.3f})")
        self.ax_fc.legend()
        self.ax_fc.grid(True, alpha=0.3)
        
        self.fig_fc.tight_layout()
        self.canvas_fc.draw()

    def _run_scenario(self):
        if self.model_info is None:
            messagebox.showwarning("Внимание", "Сначала обучите модель")
            return
        try:
            base = {f: self.df[f].mean() for f in FEATURES}
            scenario = base.copy()
            for met in CONTROLLABLE:
                val = float(self.entries[met].get())
                scenario[met] = val
                
            X_base = np.array([[base[f] for f in FEATURES]])
            X_sc = np.array([[scenario[f] for f in FEATURES]])
            
            pred_base = self.model_info['model'].predict(X_base)[0]
            pred_sc = self.model_info['model'].predict(X_sc)[0]
            delta = pred_sc - pred_base
            
            EPS = 0.1 
            
            if delta > EPS:
                sign = "↑"
                scenario_color = "#27ae60"
                txt_color = "green"
            elif delta < -EPS:
                sign = "↓"
                scenario_color = "#e74c3c"
                txt_color = "red"
            else:
                sign = "≈"
                scenario_color = "#95a5a6"
                txt_color = "gray"
            
            txt = f"Базовый прогноз: {pred_base:.0f} тыс. руб.\n"
            txt += f"Сценарий: {pred_sc:.0f} тыс. руб. {sign} {abs(delta):.0f} тыс. руб.\n"
            txt += f"Изменения: {', '.join([f'{RU_NAMES[m]}: {base[m]:.3f} → {scenario[m]:.3f}' for m in CONTROLLABLE])}"
            self.lbl_scenario.config(text=txt, foreground=txt_color)
            
            self.ax_sc.clear()
            self.ax_sc.bar(["Базовый", "Сценарий"], [pred_base, pred_sc], 
                           color=["#95a5a6", scenario_color])
            self.ax_sc.set_ylabel("Выручка (тыс. руб.)")
            self.ax_sc.set_title(f"Сравнение сценариев ({sign} {abs(delta):.0f})")
            self.ax_sc.grid(axis='y', alpha=0.3)
            
            self.fig_sc.tight_layout()
            self.canvas_sc.draw()
        except Exception as e:
            messagebox.showerror("Ошибка сценария", f"Проверьте ввод чисел:\n{e}")

    def _export_json(self):
        if self.index_scores is None:
            messagebox.showwarning("Внимание", "Сначала рассчитайте индекс")
            return
        fp = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
        if not fp: return
        report = {
            "system": "SaaS Analytics (Композитный индекс)",
            "date": datetime.datetime.now().isoformat(),
            "weights": self.weights.tolist(),
            "forecast_r2": self.model_info["r2"] if self.model_info else None,
            "composite_index": {int(idx): round(val, 4) for idx, val in self.index_scores.items()}
        }
        with open(fp, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        messagebox.showinfo("Успех", f"Отчёт сохранён в {fp}")

if __name__ == "__main__":
    root = tk.Tk()
    app = SaaSAnalyticsApp(root)
    root.mainloop()