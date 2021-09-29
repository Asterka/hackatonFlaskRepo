# import warnings
import json
import numpy as np
from numba import njit, prange, types, config  # it is better to install numba with conda (for llvm support)
from multiprocessing import Pool, cpu_count as cores_number
from matplotlib.figure import Figure
import os


max_resolution = 1_000_000  # maximum number of budget plans possibly considered, can be increased if needed
# it increases optimization time complexity by O(log(max_resolution)) multiplier and consumes
# O(max_resolution * n_approaches * number_of_cores) memory

# total solution has a time complexity of O(n_approaches ^ risks_number * log(budget_plans_number) / cores_number),
# where budget_plans_number <= max_resolution (depends on the costs table data)

# dicts for pretty table instead of raw numbers at front
dmg_lvls_decoding = {'0': 'Несущественные последствия',
                     '1': 'Умеренные последствия',
                     '2': 'Существенные последствия',
                     '3': 'Катастрофические последствия'}

probability_decoding = {'0': 'Нулевая',
                        '1': 'Низкая',
                        '2': 'Средняя',
                        '3': 'Высокая'}

# dicts for pretty table to raw numbers conversion from json
dmg_lvls_inversed = {discription: number for number, discription in dmg_lvls_decoding.items()}
probability_inversed = {discription: number for number, discription in probability_decoding.items()}


class Singleton(type):
    # singleton for a BaseClass
    # actually no instances of the BaseClass are created
    # but for the future purposes it might be useful
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class NumpyEncoder(json.JSONEncoder):
    # used to encode 2d and 3d numpy array (tables) to json format for front
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class Table:
    """deferred class for all 3 tables:
       risks (amount of damage and probability for each option),
       costs (cost of each option)
       reasoning (reason of each option)
       Each of 3 tables has it's data for each risk_source and for each level of security (#levels = n_approaches)
    """
    def __init__(self, n_risks_sources=12, n_approaches=5, default_risks_names=None):
        self.data = np.empty(0)
        self.n_risks_sources = n_risks_sources
        self.n_approaches = n_approaches
        if not default_risks_names:
            self.risks_names = np.array(['Выход из строя серверного оборудования в офисе Заказчика',
                                         'Потеря доступа к серверу',
                                         'Проблема с ПО', 'Выход из строя ПК',
                                         'Потеря данных при передаче с буровой площадки в офис Заказчика',
                                         'Потеря данных при передаче внутри буровой площадки',
                                         'Не согласованность каналов передачи данных между подрядчиками',
                                         'Отказ/выход из строя оборудования (автоматики/датчиков)',
                                         'Обрыв ЛЭП',
                                         'Проблема с каналом связи',
                                         'Неисправность датчиков/др. оборудования',
                                         'Некорректная обработка ПО'], dtype=str)
        else:
            self.risks_names = default_risks_names

    def update_value(self, new_val, x, y, z=None):
        """
        Updates value of the table element and updates is_relative flag
        to show that old calculations are incorrect now
        :param new_val: new value
        :param x: x coordinate (row index)
        :param y: y coordinate (column index)
        :param z: z coordinate for risks table
        :return:
        """
        # updates value of the table element and updates is_relative flag
        # to show that old calculations are incorrect now
        if not z:
            self.data[x, y] = new_val
        elif self.data.shape == 3:
            self.data[x, y, z] = new_val
        else:
            Warning('update_value error: z is not None while given 2d table')
        BaseClass.is_relevant = False

    def add_row(self):
        """
        Adds new last row to the table
        :return:
        """
        if len(self.data.shape) == 2:
            self.data = np.append(self.data, np.zeros((1, self.data.shape[1]), dtype=self.data.dtype), axis=0)
        else:
            self.data = np.append(self.data, np.zeros((1, self.data.shape[1], self.data.shape[2]), dtype=self.data.dtype), axis=0)

    def add_column(self):
        """
        Adds new last column to the table
        :return:
        """
        if len(self.data.shape) == 2:
            self.data = np.append(self.data, np.zeros((self.data.shape[0], 1), dtype=self.data.dtype), axis=1)
        else:
            self.data = np.append(self.data, np.zeros((self.data.shape[0], 1, self.data.shape[2]), dtype=self.data.dtype), axis=1)

    def delete_row(self, row_n=-1):
        """
        Deletes row from the table
        :param row_n: index of a row to delete, if last then can be column_n = -1
        :return:
        """
        if row_n >= self.data.shape[0]:
            # warnings.warn(f'error (out of bounds) while trying to delete a row {row_n} out of {self.data.shape[0]}')
            pass
        elif row_n != -1:
            self.data = self.data[[i for i in range(self.data.shape[0]) if i != row_n]]
        else:
            self.data = self.data[:-1]

    def delete_column(self, column_n=-1):
        """
        Deletes column from the table
        :param column_n: index of a column to delete, if last then can be column_n = -1
        :return:
        """
        if column_n >= self.data.shape[1]:
            # warnings.warn(f'error (out of bounds) while trying to delete a column {column_n} out of {self.data.shape[1]}')
            pass
        elif column_n != -1:
            self.data = self.data[:, [i for i in range(self.data.shape[1]) if i != column_n]]
        else:
            self.data = self.data[:, :-1]

    def convert_numpy_to_json(self):
        """
        Converts a table to json format
        :return: json dump data
        """
        to_dump = self.data[:]
        row_with_approach_names = np.arange(self.n_approaches + 1).astype(str).astype(object)
        row_with_approach_names[0] = 'Названия рисков \\ Уровни проработки'

        if len(self.data.shape) == 3:
            # in case of risks table
            to_dump = self.data.astype(int).astype(np.object)[:]
            damage_lvl, probability_lvl = to_dump[:, :, 0], to_dump[:, :, 1]
            for lvl in dmg_lvls_decoding.keys():
                damage_lvl[np.where(damage_lvl == int(lvl))] = dmg_lvls_decoding[lvl]

            for prob in probability_decoding.keys():
                probability_lvl[np.where(probability_lvl == int(prob))] = probability_decoding[prob]
            to_dump = np.empty(self.data.shape[:2], dtype=np.object)
            for i in range(to_dump.shape[0]):
                for j in range(to_dump.shape[1]):
                    to_dump[i, j] = damage_lvl[i, j] + ', ' + probability_lvl[i, j]
        column_wth_risks_names = self.risks_names[:, None]
        to_dump = np.append(column_wth_risks_names, np.transpose(to_dump, axes=(1, 0)), axis=1)  # adds header column

        to_dump = np.append(row_with_approach_names[None, :], to_dump, axis=0)  # adds header row
        return json.dumps(to_dump.astype(str), cls=NumpyEncoder)

    def read_from_json(self, json_data, table_name):
        """
        reconstruct a table from json format and turn it to numpy array
        :param json_data: data
        :param table_name: 3 options: risks, costs, reasoning (depends on a table type)
        :return: actually returning value can be ignored.
                 Returns True in case of success, False in case of an unexpected table type and an exception in case of
                 an error, which will be caught at the server.py part
        """
        #
        json_load = json.loads(json_data)

        to_data = np.asarray(json_load)[:, 1:]
        # not [1:, 1:] because first row is already skipped by the front part
        if table_name == 'risks':
            new_to_data = np.empty((to_data.shape[0], to_data.shape[1], 2), dtype=float)
            for i in range(to_data.shape[0]):
                for j in range(to_data.shape[1]):

                    pair = to_data[i, j].split(', ')

                    new_to_data[i, j, 0] = float(dmg_lvls_inversed[pair[0]])
                    new_to_data[i, j, 1] = float(probability_inversed[pair[1]])
            to_data = new_to_data.transpose((1, 0, 2))
            self.data = to_data.astype(float)  # comment this line for testing this function
            return True
        elif table_name == 'costs':
            to_data = to_data.transpose((1, 0))
            self.data = to_data.astype(float)  # comment this line for testing this function
            return True
        elif table_name == 'reasoning':
            to_data = to_data.transpose((1, 0))
            self.data = to_data  # comment this line for testing this function
            return True
        else:
            return False


class RiskTable(Table):
    """
        RiskTable is a Table (n_approaches x n_risk_sources x 2), explanation below
    """
    def __init__(self, given_init_risks=None):
        super(RiskTable, self).__init__()
        if not given_init_risks:
            # default shape (5: for each security lvl approach including base approach, 12: for each risk_source, 2: damage & probability)

            # security lvl dimensions:
            # 0: 0 ур. проработки
            # 1: 1 ур. проработки
            # 2: 2 ур. проработки
            # 3: 3 ур. проработки
            # 4: 4 ур. проработки

            # risk_source dimensions:
            # 0: Несоблюдение условий размещения и функционирования серверного и телекоммуникационного оборудования
            # 1: Потеря доступа к серверу
            # 2: Проблема с ПО
            # 3: Потеря данных при передаче с буровой площадки в офис Заказчика
            # 4: Потеря данных при передаче внутри буровой площадки
            # 5: Не согласованность каналов передачи данных между подрядчиками
            # 6: Ошибка в алгоритме приёма/передачи данных
            # 7: Отказ/выход из строя оборудования (автоматики/датчиков)
            # 8: Обрыв ЛЭП
            # 9: Проблема с каналом связи
            # 10: Неисправность датчиков/др. оборудования
            # 11: Некорректная обработка ПО

            # damage & probability value options:
            # damage lvl:
            # 0: Несущественные последствия
            # 1: Умеренные последствия
            # 2: Существенные последствия
            # 3: Катастрофические последствия

            # probability:
            # 0: Нулевая
            # 1: Низкая
            # 2: Средняя
            # 3: Высокая

            self.data = np.empty((self.n_approaches, self.n_risks_sources, 2))

            self.data[0, 0] = [2, 3]
            self.data[0, 1] = [2, 2]
            self.data[0, 2] = [2, 3]
            self.data[0, 3] = [2, 3]
            self.data[0, 4] = [2, 3]
            self.data[0, 5] = [2, 3]
            self.data[0, 6] = [2, 3]
            self.data[0, 7] = [3, 2]
            self.data[0, 8] = [2, 3]
            self.data[0, 9] = [2, 3]
            self.data[0, 10] = [3, 2]
            self.data[0, 11] = [2, 3]

            self.data[1, 0] = [1, 3]
            self.data[1, 1] = [2, 1]
            self.data[1, 2] = [1, 3]
            self.data[1, 3] = [1, 3]
            self.data[1, 4] = [1, 3]
            self.data[1, 5] = [1, 2]
            self.data[1, 6] = [1, 3]
            self.data[1, 7] = [1, 3]
            self.data[1, 8] = [1, 3]
            self.data[1, 9] = [2, 2]
            self.data[1, 10] = [1, 3]
            self.data[1, 11] = [2, 2]

            self.data[2, 0] = [1, 3]
            self.data[2, 1] = [1, 1]
            self.data[2, 2] = [1, 2]
            self.data[2, 3] = [1, 2]
            self.data[2, 4] = [1, 2]
            self.data[2, 5] = [1, 1]
            self.data[2, 6] = [1, 2]
            self.data[2, 7] = [1, 2]
            self.data[2, 8] = [1, 2]
            self.data[2, 9] = [1, 2]
            self.data[2, 10] = [1, 2]
            self.data[2, 11] = [1, 1]

            self.data[3, 0] = [1, 1]
            self.data[3, 1] = [1, 0]
            self.data[3, 2] = [1, 1]
            self.data[3, 3] = [1, 1]
            self.data[3, 4] = [1, 1]
            self.data[3, 5] = [1, 1]
            self.data[3, 6] = [1, 1]
            self.data[3, 7] = [1, 1]
            self.data[3, 8] = [1, 0]
            self.data[3, 9] = [0, 2]
            self.data[3, 10] = [1, 1]
            self.data[3, 11] = [1, 1]

            self.data[4, 0] = [0, 0]
            self.data[4, 1] = [1, 0]
            self.data[4, 2] = [1, 0]
            self.data[4, 3] = [0, 0]
            self.data[4, 4] = [0, 0]
            self.data[4, 5] = [1, 1]
            self.data[4, 6] = [1, 1]
            self.data[4, 7] = [0, 1]
            self.data[4, 8] = [1, 0]
            self.data[4, 9] = [0, 2]
            self.data[4, 10] = [0, 1]
            self.data[4, 11] = [1, 0]
        else:
            self.data = given_init_risks[:]


class CostsTable(Table):
    """
        CostsTable is a Table (n_approaches x n_risk_sources) for cost for each risk management option
    """

    def __init__(self, given_init_costs=None):
        """
        If given_init_costs are not given assigns all the values from our default table
        :param given_init_costs:
        """
        super(CostsTable, self).__init__()
        if not given_init_costs:
            self.data = np.empty((self.n_approaches, self.n_risks_sources))
            self.data[0] = np.zeros(self.n_risks_sources)
            self.data[1] = [2.5, 3, 3.5, 2., 1.5, 1., 2., 4., 5., 1., 1.5, 2.]
            self.data[2] = [3.125, 3.75, 4.375, 2.5, 1.875, 1.25, 2.5, 5., 6.25, 1.25, 1.875, 2.5]
            self.data[3] = [4.375, 5.25, 6.125, 3.5, 2.625, 1.75, 3.5, 7., 8.75, 1.75, 2.625, 3.5]
            self.data[4] = [7.875, 9.45, 11.025, 6.3, 4.725, 3.15, 6.3, 12.6, 15.75, 3.15, 4.725, 6.3]
        else:
            self.data = given_init_costs[:]


class Reasoning(Table):
    """
    Reasons for each of the problems list located according to the necessary complexity level of solution and
    corresponding risk
    """
    def __init__(self, default=True, init_reasoning=None):
        """
        if default use our values, else either init with the given init_reasoning or use an empty table
        :param default:
        :param init_reasoning:
        """
        super(Reasoning, self).__init__()
        if default and not init_reasoning:
            self.data = np.empty((self.n_approaches, self.n_risks_sources), dtype=object)

            self.data[0] = ["-", "-", "-", "-",
                            "-", "-", "-", "-",
                            "-", "-", "-", "-"]
            self.data[1] = ["Сбои в работе серверного оборудования ", "Эпидемиологическая обстановка в стране",
                            "Несанкционированный вход в ПО сторонних пользователей", "Отсутствие сигнала мобильной связи",
                            "Отсутствие коммуникационных каналов между участниками процесса строительства скважины",
                            "Несовместимость форм передачи данных", "Передача недостоверной информации", "Потеря данных - отсутствие возможности сохранения данных",
                            "Сбои в работе локальной сети", "Передача недостоверной информации", "Сбой настроек ПО", "-"]
            self.data[2] = ["Потеря связи с основным сервером, сбои в работе сервера",
                            "Невозможность получения данных с некорпоративных устрйоств (сетей)",
                            "Истечение срока действия лицензионного ключа", "Изменение зоны покрытия (аарийные работы, климатические условия)",
                            "Сбои в работе локальной сети", "Несовместимость каналов передачи данных",
                            "Невозможность восстановления работоспособности датчика", "Отсутствие возможности продолжения работы в краткосрочном периоде",
                            "Задержка передачи данных", "Невозможность восстановления корректной передачи данных", "-", "-"]
            self.data[3] = ["Выход из строя сервера", "Неоснащенность персоналанеобходимыми устройствами для удаленной работы",
                            "Незапланированный выход ПК и отсутствие быстрой возможности восстановления работоспособности основного ПК",
                            "Полное отсутствие сигнала мобильной связи у всех операторов связи",
                            "Выход из строя основного канала связи  ", "-", "Превышение сроков эксплуатации и ТО", "Отсутствие возможности продолжения работы в долгосрочном периоде",
                            "-", "Превышение сроков эксплуатации и ТО", "-", "-"]
            self.data[4] = ["Потеря данных с севрерного оборудования; Перезагруженность сервера данными", "-",
                            "Попытка взлома и утечки корпоративной информации", "Выход из сторя основного спутникового канала связи",
                            "Выход из сторя основного спутникового канала связи", "-", "Отсутствие тех. обслуживания датчиков и оборудования",
                            "-", "-", "Отсутствие тех. обслуживания датчиков и оборудования", "-", "-"]

        elif not init_reasoning:
            self.data = np.empty((self.n_approaches, self.n_risks_sources), dtype=str)
        else:
            self.data = np.array(init_reasoning, dtype=str)


class BaseClass(metaclass=Singleton):
    """
       Base class, stores all the tables data and uses them for calculations, updates and optimal curve visualization
    """
    risks_table = RiskTable()  # risks_table: n_approaches x n_risk_sources x 2
    costs_table = CostsTable()  # costs_table: n_approaches x n_risk_sources
    reasons_table = Reasoning()  # reasons_table: n_approaches x n_risk_sources

    max_costs = None  # different budget plans which are considered for optimization, sorted is the ascending order
    optimal_risks = None  # risks of the optimal strategies for all budget plans
    optimal_costs = None  # costs of the optimal strategies for all budget plans
    optimal_strategies = None  # the optimal strategies for all budget plans
    # (the best set of levels of approaches for each risk)
    is_relevant = None  # flag to identify if calculations are relevant (changes when the table is edited)
    risk_cost = 2.  # coefficient of the expensiveness of risk damage cost compared to risk management investments
    steps_for_calc = -1  # number of different budget plans (evenly spaced from the cheapest to the most expensive)
    # if steps_for_calc = -1 then the best resolution is used for the plot (all possible budget plans from the given
    # table: from min budget to max budget with the step = min difference between costs of approaches across all levels)

    # the solution has a time complexity of ~
    # n_approaches ^ n_risks * log(steps_for_calc) / n_cores (for parallel execution)

    use_numba = True  # if True then numba optimization will be used for multiprocessing

    @classmethod
    def save_optimal_strategy_curve(cls):
        """
            Represents minimal risk for each available budget and optimal point for a given risk_cost field of the class
        :return:
        """
        fig = Figure()
        ax = fig.add_subplot(111)
        ax.plot(cls.max_costs, cls.optimal_risks, c='orange')

        optimal_point = np.argmin(
            cls.risk_cost * cls.optimal_risks + cls.optimal_costs)  # consider cost of data as twice cost of the solution
        ax.scatter(cls.max_costs[optimal_point], cls.optimal_risks[optimal_point], color='g')
        ax.set_ylim(top=cls.optimal_risks[0] * 1.2, bottom=0)

        ax.set_xlabel('Стоимость баз. ед.')
        ax.set_ylabel('Риск в ед. риска')
        ax.title.set_text('Риск при оптимальном наборе решений с ограничением на бюджет')
        fig.savefig(os.path.join(os.path.curdir, '..', 'hackaton-front', 'dist', 'hackaton-front', 'optimal_strategy_curve.png'))
        return  # fig

    @classmethod
    def optimize_for_all_costs(cls, costs_list=None, n_steps=None, multiprocessing_mode=True):
        """
        Calculates minimal risk for each available budget and optimal point for a given risk_cost field of the class
        Solution is simply brute force (that can be optimized by going through each combination just once, not for each
        cost: that was done for simple parallelization with numba, numba can be replaced with another lib with
        an opportunity to explicitly change number of workers to optimize this way)

        As soon as it is MVP we didn't consider smart update of the calculations after updating values: all calculations
        repeats to update the relevant solution

        :param costs_list: list of costs,
               by default: costs from min cost to max cost evenly spaced with steps_for_calc steps,
               if steps_for_calc = -1 then we use the minimum step
        :param n_steps: number of steps for optimization (by default as many steps as there are costs_list elements)
        :param multiprocessing_mode: if True then run parallel optimization else optimized single-core
        :return:
        """
        assert cls.risks_table.data.shape[:-1] == cls.costs_table.data.shape, f"Shape doesn't match: risks " \
                                                                              f"{cls.risks_table.data.shape[:-1]} != " \
                                                                              f"{cls.costs_table.data.shape}"
        if not costs_list or not n_steps:
            if not cls.max_costs is None and not cls.optimal_risks is None and not cls.optimal_costs is None:
                cls.save_optimal_strategy_curve()
            min_theoretical_cost = cls.costs_table.data[0].sum()
            max_theoretical_cost = cls.costs_table.data[-1].sum()
            min_step = float('inf')

            # defining interval and minimum step for final plot with optimal data for a given budget
            plane_costs = cls.costs_table.data.ravel()
            for i, j in np.ndindex(plane_costs.shape[0], plane_costs.shape[0]):
                cur_step = abs(plane_costs[i] - plane_costs[j])
                if 0 < cur_step < min_step:
                    min_step = cur_step

            if cls.steps_for_calc == -1:
                n_steps = int((max_theoretical_cost - min_theoretical_cost) / min_step) + 1
                n_steps = min(n_steps, max_resolution)
            else:
                n_steps = min(cls.steps_for_calc, max_resolution)
            max_costs = np.linspace(min_theoretical_cost, max_theoretical_cost, n_steps)

        else:
            max_costs = sorted(list(costs_list)[:])

        # initializing global variables for all functions below (numba can't access classes fields)
        costs = BaseClass.costs_table.data[:]
        risks = BaseClass.risks_table.data[:]
        n_risks_sources = BaseClass.costs_table.n_risks_sources
        n_approaches = BaseClass.costs_table.n_approaches
        # print('max_costs: \n', max_costs)

        @njit
        def calc_cost_and_risk(risk_strategy, max_curr_cost=2147483647., opt_strategy_score=2147483647.):
            """
            calculates cost and risk of the single strategy
            :param risk_strategy: 12 different risk sources (each has value from 0 to 4 according to level of security)
            :param max_curr_cost: max cost that this strategy can have (if it is more expensive strategy will not be considered)
            :param opt_strategy_score: optimal found strategy risk, more risky strategy will not be considered
            :return:
            """
            cost = 0
            curr_risk = 0
            for i in range(len(risk_strategy)):
                cost += costs[risk_strategy[i], i]
                curr_risk += risks[risk_strategy[i], i, 0] * risks[risk_strategy[i], i, 1]
                if cost > max_curr_cost or curr_risk > opt_strategy_score:
                    break
            return cost, curr_risk

        @njit
        def binsearch_njitted(a, x, lo=0, hi=None):
            """Return the index where to insert item x in list a, assuming a is sorted in ascending order.

            The return value i is such that all e in a[:i] have e < x, and all e in
            a[i:] have e >= x.
            """

            if hi is None:
                hi = len(a)
            while lo < hi:
                mid = (lo + hi) // 2
                if a[mid] < x:
                    lo = mid + 1
                else:
                    hi = mid
            return lo

        @njit
        def calc_opt_strategy_score_singlecore_optimized():
            """
            calculates the best risk management strategy, cost and risk for all ranges of budget plans that are considered
            according to the data, the solution is implemented in a brute force fashion;
            All calculations run on a single core, but optimized with numba, actually much faster than a multiprocessing
            version on a small number of cores (less than 20).

            :return: for each considered budget plan:
                                                    optimal risk amount,
                                                    optimal cost of the solution,
                                                    the best strategy
            """
            inf = 2147483647
            optimal_risks_ = np.full_like(max_costs, inf)
            optimal_costs_ = np.full_like(max_costs, inf)
            best_strategy = np.full((max_costs.shape[0], n_risks_sources), fill_value=-1)
            options = n_approaches ** n_risks_sources  # all possible strategies
            for risk_strategy_base10 in range(options):
                # brute force loop through all of the options only once for each: updating relevant cost best value everytime
                risk_strategy = base_repr(risk_strategy_base10, n_approaches, n_risks_sources)  # this "hides" all the nested loops in a single line
                cost, curr_risk = calc_cost_and_risk(risk_strategy)
                ind = binsearch_njitted(max_costs, cost)
                # update best value for the fixed cost if risk is the best for it
                # invariant that cost1 > cost2 => risk1 > risk2 is satisfied by default because of how the loop goes
                if curr_risk < optimal_risks_[ind]:
                    optimal_risks_[ind] = curr_risk
                    optimal_costs_[ind] = cost
                    best_strategy[ind] = risk_strategy

            # fill all the missing untouched risk management plans (some budgets can be still not visited)
            fill_index = 0
            best_risk = inf
            best_cost = inf
            best_strat = np.empty(n_risks_sources, dtype=types.int64)
            while fill_index < max_costs.shape[0]:
                if optimal_risks_[fill_index] < best_risk:
                    best_risk = optimal_risks_[fill_index]
                    best_cost = optimal_costs_[fill_index]
                    best_strat = best_strategy[fill_index]
                else:
                    best_strategy[fill_index] = best_strat[:]
                    optimal_risks_[fill_index] = best_risk
                    optimal_costs_[fill_index] = best_cost
                fill_index += 1
            return optimal_risks_, optimal_costs_, best_strategy

        @njit(parallel=True)
        def calc_opt_strategy_score_multiprocessing_optimized(threads=config.NUMBA_DEFAULT_NUM_THREADS,
                                                              costs_local=costs, risks_local=risks):
            """
            Parallel version of the calc_opt_strategy_score_singlecore_optimized with numba njit optimization

            :return: for each considered budget plan:
                                                    optimal risk amount,
                                                    optimal cost of the solution,
                                                    the best strategy
            """
            inf = 2147483647.
            local_max_costs = max_costs[:]
            # the solution in this case is similar to a single core, but the task (all the combinations) is divided
            # into parts, 1 part for each worker, after that all their solutions are combined
            # init:
            thread_optimal_risks = np.full(shape=(threads + 1, local_max_costs.shape[0]), fill_value=inf)
            thread_optimal_costs = np.full(shape=(threads + 1, local_max_costs.shape[0]), fill_value=inf)
            thread_best_strategy = np.full(shape=(threads + 1, local_max_costs.shape[0], n_risks_sources), fill_value=-1)

            global_optimal_risks, global_optimal_costs = np.full(local_max_costs.shape[0], inf), \
                                                         np.full(local_max_costs.shape[0], inf)
            global_best_strategy = np.full((local_max_costs.shape[0], n_risks_sources), -1)

            all_options = n_approaches ** n_risks_sources
            if all_options <= threads:  # then it will be much faster on a single core with numba
                return calc_opt_strategy_score_singlecore_optimized()

            # splitting the task into batches for workers:
            risk_strategy_batches = np.empty(threads + 1, dtype=types.int64)
            increment = all_options//threads
            risk_strategy_batches[:-1] = (np.arange(threads) + 1) * increment
            risk_strategy_batches[-1] = all_options + 1

            # parallel execution
            for thread in prange(threads):
                for risk_strategy_base10 in range(risk_strategy_batches[thread] - increment, risk_strategy_batches[thread]):
                    # brute force loop through all of the options only once for each: updating relevant cost best value everytime

                    # Converts x from base 10 to base "base",
                    # this code is used to encode all states of the brute force (more description at base_repr method below)
                    # base_repr analogue from numpy (base_repr is not supported by numba)
                    # base_repr implementation from below is not used to avoid an array creation at the method which
                    # breaks the parallel optimization in numba
                    max_deg = 0
                    next_digit = 1
                    while risk_strategy_base10 // next_digit > 0:
                        max_deg += 1
                        next_digit *= n_approaches

                    if risk_strategy_base10 == 0:
                        risk_strategy = np.zeros(n_risks_sources, dtype='i4')
                    else:
                        risk_strategy = np.zeros(n_risks_sources, dtype='i4')
                        m = next_digit // n_approaches
                        for digit in range(max_deg):
                            d = risk_strategy_base10 // m
                            risk_strategy[max_deg - digit - 1] = d
                            risk_strategy_base10 = risk_strategy_base10 - d * m
                            m /= n_approaches

                    cost = 0.
                    curr_risk = 0.
                    for risk_index in range(len(risk_strategy)):
                        cost += costs_local[risk_strategy[risk_index], risk_index]
                        curr_risk += risks_local[risk_strategy[risk_index], risk_index, 0] * risks_local[risk_strategy[risk_index], risk_index, 1]

                    ind = binsearch_njitted(max_costs, cost)
                    # update best value for the fixed cost if risk is the best for it
                    # invariant that cost1 > cost2 => risk1 > risk2 is satisfied by default because of how the loop goes
                    if curr_risk < thread_optimal_risks[thread, ind]:
                        thread_optimal_risks[thread, ind] = curr_risk
                        thread_optimal_costs[thread, ind] = cost
                        thread_best_strategy[thread, ind] = risk_strategy

            # calculate the last batch that is not full
            if not all_options % threads:
                for last_risk_strategy_base10 in range(risk_strategy_batches[-2] + 1, risk_strategy_batches[-1]):
                    risk_strategy = base_repr(last_risk_strategy_base10, n_approaches, n_risks_sources)
                    cost, curr_risk = calc_cost_and_risk(risk_strategy)
                    ind = binsearch(local_max_costs, cost)
                    if curr_risk < thread_optimal_risks[-1, ind]:
                        thread_optimal_risks[-1, ind] = curr_risk
                        thread_optimal_costs[-1, ind] = cost
                        thread_best_strategy[-1, ind] = risk_strategy

            # combine the result of all of the workers into one optimal and best solution for all budget plans
            for cost_index in range(local_max_costs.shape[0]):
                for thread in range(threads + 1):
                    if thread_optimal_risks[thread, cost_index] < global_optimal_risks[cost_index]:
                        global_optimal_risks[cost_index] = thread_optimal_risks[thread, cost_index]
                        global_optimal_costs[cost_index] = thread_optimal_costs[thread, cost_index]
                        global_best_strategy[cost_index] = thread_best_strategy[thread, cost_index]

            # fill all the missing untouched risk management plans (some budgets can be still not visited)
            fill_index = 0
            best_risk = inf
            best_cost = inf
            best_strat = np.empty(n_risks_sources, dtype=types.int64)
            while fill_index < max_costs.shape[0]:
                if global_optimal_risks[fill_index] < best_risk:
                    best_risk = global_optimal_risks[fill_index]
                    best_cost = global_optimal_costs[fill_index]
                    best_strat = global_best_strategy[fill_index]
                else:
                    global_best_strategy[fill_index] = best_strat[:]
                    global_optimal_risks[fill_index] = best_risk
                    global_optimal_costs[fill_index] = best_cost
                fill_index += 1

            return global_optimal_risks, global_optimal_costs, global_best_strategy

        def calc_opt_strategy_score_parallel_raw_python(threads=cores_number()):
            """
            Parallel version of the calc_opt_strategy_score_singlecore_optimized (without numba optimization)

            :return: for each considered budget plan:
                                                    optimal risk amount,
                                                    optimal cost of the solution,
                                                    the best strategy
            """
            inf = 2147483647.
            local_max_costs = max_costs[:]
            # the solution in this case is similar to a single core, but the task (all the combinations) is divided
            # into parts, 1 part for each worker, after that all their solutions are combined
            # init:
            thread_optimal_risks = np.full(shape=(threads + 1, local_max_costs.shape[0]), fill_value=inf)
            thread_optimal_costs = np.full(shape=(threads + 1, local_max_costs.shape[0]), fill_value=inf)
            thread_best_strategy = np.full(shape=(threads + 1, local_max_costs.shape[0], n_risks_sources), fill_value=-1)

            global_optimal_risks, global_optimal_costs = np.full(local_max_costs.shape[0], inf), \
                                                         np.full(local_max_costs.shape[0], inf)
            global_best_strategy = np.full((local_max_costs.shape[0], n_risks_sources), -1)

            all_options = n_approaches ** n_risks_sources
            if all_options <= threads:  # then it will be much faster on a single core with numba
                return calc_opt_strategy_score_singlecore_optimized()
            # splitting the task into batches for workers:
            risk_strategy_batches = np.empty(threads + 1, dtype=int)
            increment = all_options//threads
            risk_strategy_batches[:-1] = (np.arange(threads) + 1) * increment
            risk_strategy_batches[-1] = all_options + 1

            # parallel execution
            with Pool() as pool:
                arguments = zip(np.repeat(local_max_costs[None, ...], threads, axis=0),
                                risk_strategy_batches[:-1],
                                np.full(threads, local_max_costs.shape[0]), np.full(threads, increment),
                                np.full(threads, n_risks_sources), np.full(threads, n_approaches),
                                np.repeat(costs[None, ...], threads, axis=0),
                                np.repeat(risks[None, ...], threads, axis=0))  # preparing input data for each worker

                result = pool.starmap(search_for_batch_best_solution, arguments)

            for element in range(len(result)):
                thread_optimal_risks[element], thread_optimal_costs[element], thread_best_strategy[element] = result[element]

            # calculate the last batch that is not full
            if not all_options % threads:
                for last_risk_strategy_base10 in range(risk_strategy_batches[-2] + 1, risk_strategy_batches[-1]):
                    risk_strategy = base_repr(last_risk_strategy_base10, n_approaches, n_risks_sources)
                    cost, curr_risk = calc_cost_and_risk(risk_strategy)
                    ind = binsearch(local_max_costs, cost)
                    if curr_risk < thread_optimal_risks[-1, ind]:
                        thread_optimal_risks[-1, ind] = curr_risk
                        thread_optimal_costs[-1, ind] = cost
                        thread_best_strategy[-1, ind] = risk_strategy

            # combine the result of all of the workers into one optimal and best solution for all budget plans
            for cost_index in range(local_max_costs.shape[0]):
                for thread in range(threads + 1):
                    if thread_optimal_risks[thread, cost_index] < global_optimal_risks[cost_index]:
                        global_optimal_risks[cost_index] = thread_optimal_risks[thread, cost_index]
                        global_optimal_costs[cost_index] = thread_optimal_costs[thread, cost_index]
                        global_best_strategy[cost_index] = thread_best_strategy[thread, cost_index]

            # fill all the missing untouched risk management plans (some budgets can be still not visited)
            fill_index = 0
            best_risk = inf
            best_cost = inf
            best_strat = np.empty(n_risks_sources, dtype=int)
            while fill_index < max_costs.shape[0]:
                if global_optimal_risks[fill_index] < best_risk:
                    best_risk = global_optimal_risks[fill_index]
                    best_cost = global_optimal_costs[fill_index]
                    best_strat = global_best_strategy[fill_index]
                else:
                    global_best_strategy[fill_index] = best_strat[:]
                    global_optimal_risks[fill_index] = best_risk
                    global_optimal_costs[fill_index] = best_cost
                fill_index += 1

            return global_optimal_risks, global_optimal_costs, global_best_strategy

        # you can choose parallel or a single_core optimized version of the algorithm
        if multiprocessing_mode:
            if cls.use_numba:
                optimal_risks, optimal_costs, optimal_strats = calc_opt_strategy_score_multiprocessing_optimized()
            else:
                # slow alternative (works without numba), not recommended
                optimal_risks, optimal_costs, optimal_strats = calc_opt_strategy_score_parallel_raw_python()
        else:
            optimal_risks, optimal_costs, optimal_strats = calc_opt_strategy_score_singlecore_optimized()
        # update the class data
        BaseClass.optimal_risks = optimal_risks
        BaseClass.optimal_costs = optimal_costs
        BaseClass.max_costs = max_costs
        BaseClass.optimal_strategies = optimal_strats
        BaseClass.is_relevant = True
        optimal_point = BaseClass.save_optimal_strategy_curve()
        return optimal_point, max_costs[optimal_point], optimal_risks[optimal_point]

    @classmethod
    def edit_risk_table_element(cls, new_value, x, y):
        cls.risks_table.update_value(new_value, x, y)

    @classmethod
    def edit_costs_table_element(cls, new_value, x, y):
        cls.costs_table.update_value(new_value, x, y)

    @classmethod
    def edit_reasons_table_element(cls, new_value, x, y):
        cls.reasons_table.update_value(new_value, x, y)

    @classmethod
    def add_approach(cls):
        """
        Adds a new approach of managing the risks (to all of the tables)
        :return:
        """
        cls.risks_table.add_row()
        cls.costs_table.add_row()
        cls.reasons_table.add_row()
        cls.risks_table.n_approaches += 1
        cls.costs_table.n_approaches += 1
        cls.reasons_table.n_approaches += 1
        cls.is_relevant = False

    @classmethod
    def add_risk_source(cls):
        """
        Adds a new risk into consideration (to all of the tables)
        :return:
        """
        cls.risks_table.add_column()
        cls.costs_table.add_column()
        cls.reasons_table.add_column()
        cls.risks_table.n_risks_sources += 1
        cls.costs_table.n_risks_sources += 1
        cls.reasons_table.n_risks_sources += 1
        cls.risks_table.risks_names += ['']
        cls.costs_table.risks_names += ['']
        cls.reasons_table.risks_names += ['']
        cls.is_relevant = False

    @classmethod
    def remove_approach(cls, i=-1):
        """
        Removes an approach at index i from all of the tables, if i=-1 then removes the last one
        :param i:
        :return:
        """
        cls.risks_table.delete_row(i)
        cls.costs_table.delete_row(i)
        cls.reasons_table.delete_row(i)
        if cls.risks_table.n_approaches > 0:
            cls.risks_table.n_approaches -= 1
            cls.costs_table.n_approaches -= 1
            cls.reasons_table.n_approaches -= 1
        cls.is_relevant = False

    @classmethod
    def remove_risk_source(cls, i=-1):
        """
        Removes a risk source at index i from all of the tables, if i=1 then removes the last one
        :param i:
        :return:
        """
        cls.risks_table.delete_column(i)
        cls.costs_table.delete_column(i)
        cls.reasons_table.delete_column(i)
        if cls.risks_table.n_risks_sources > 0:
            cls.risks_table.n_risks_sources -= 1
            cls.costs_table.n_risks_sources -= 1
            cls.reasons_table.n_risks_sources -= 1
        cls.is_relevant = False

    @classmethod
    def change_number_of_steps_for_calc(cls, new_number_of_steps):
        cls.steps_for_calc = new_number_of_steps

    @classmethod
    def get_optimal_strategies(cls):
        return cls.optimal_strategies

    @classmethod
    def get_optimal_strategy_with_risk_and_cost_given_budget(cls, budget_plan):
        strat_index = binsearch(cls.max_costs, budget_plan)
        to_dump = cls.optimal_strategies[strat_index].astype(object)
        header_column = cls.costs_table.risks_names.astype(object)
        header_row = np.array(['Названия рисков', 'Оптимальный уровень проработки']).astype(object)
        return_json_table = np.append(header_column[None, :], to_dump[None, :], axis=0)
        return_json_table = np.append(header_row[:, None], return_json_table, axis=1).T
        return json.dumps(return_json_table.astype(str), cls=NumpyEncoder), \
               cls.optimal_risks[strat_index], \
               cls.optimal_costs[strat_index]


@njit
def binsearch(budgets, current_price):
    """
    This and base_repr, search_for_batch_best_solution functions there because parallel execution algorithm uses them
    with the multiprocessing pool, which can't execute functions from class in parallel due to a pickle problem.

    Right binary search, assuming a is sorted in ascending order.
    Used to get the closest available budget for the solution with price current_price from the budgets list budgets.

    :param budgets: list of budgets available for choice
    :param current_price: current price
    :return: The lowest budget in which solution with current_price can fit
    """
    lo = 0
    hi = len(budgets)
    while lo < hi:
        mid = (lo + hi) // 2
        if budgets[mid] < current_price:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit
def base_repr(x, base, digits_n):
    """
    Converts x from base 10 to base "base", this method is used to encode all states of the brute force.
    For several reasons it is more convenient then np.ndindex in this case. Also np.ndindex is not supported for
    parallel execution in numba.

    Actually the result is reversed but it doesn't matter because still each combination is presented by a unique number.
    You can change it if you want :)

    :param x: number to convert to base "base"
    :param base: the target base
    :param digits_n: fixed number of digits, pad with 0s if necessary
    :return: np.array of digits of length digits_n
    """

    max_deg = 0
    next_digit = 1
    while x // next_digit > 0:
        max_deg += 1
        next_digit *= base

    if x == 0:
        return np.zeros(digits_n, dtype='i4')

    digits = np.zeros(digits_n, dtype='i4')
    m = next_digit // base
    for digit in range(max_deg):
        d = x // m
        digits[max_deg - digit - 1] = d
        x = x - d * m
        m /= base
    return digits


def search_for_batch_best_solution(max_costs_copy, batch, local_max_costs_shape,
                                   increment_local,
                                   n_risks_sources_local, n_approaches_local, costs_l, risks_l):
    """
    Method for brute force solution for the data portion for one thread.
    Used for multiprocessing pool method in the solution without numba, which much is slower than solution with numba.
    The method is located in a global scope to avoid pickle problem (class methods cannot be pickled).

    The code will be more clear if you read the numba njitted version at the BaseClass
    :param max_costs_copy: copy of the array with budget plans
    :param batch: number up to which to use bruteforce (it encodes all the states)
    :param local_max_costs_shape: number of elements in max_costs_copy
    :param increment_local: number of elements for one core to operate with
    :param n_risks_sources_local: number of risks
    :param n_approaches_local: number of approaches
    :param costs_l: copy of cost table (for multiprocessing)
    :param risks_l: copy of risk table (for multiprocessing)
    :return: for each considered budget plan (from [batch - increment, ..., batch]):
                                                    optimal risk amount,
                                                    optimal cost of the solution,
                                                    the best strategy
    """
    inf_local = 2147483647.
    thread_optimal_risks_local = np.full(local_max_costs_shape, fill_value=inf_local)
    thread_optimal_costs_local = np.full(local_max_costs_shape, fill_value=inf_local)
    thread_best_strategy_local = np.full(shape=(local_max_costs_shape, n_risks_sources_local),
                                         fill_value=-1)
    for risk_strategy_base10 in range(batch - increment_local, batch):
        risk_strategy_local = base_repr(risk_strategy_base10, n_approaches_local, n_risks_sources_local)

        cost_local = 0.
        curr_risk_local = 0.
        for i in range(risk_strategy_local.shape[0]):
            cost_local += costs_l[risk_strategy_local[i], i]
            curr_risk_local += risks_l[risk_strategy_local[i], i, 0] * risks_l[risk_strategy_local[i], i, 1]

        ind_local = binsearch(max_costs_copy, cost_local)
        if curr_risk_local < thread_optimal_risks_local[ind_local]:
            thread_optimal_risks_local[ind_local] = curr_risk_local
            thread_optimal_costs_local[ind_local] = cost_local
            thread_best_strategy_local[ind_local] = risk_strategy_local

    fill_index = 0
    best_strat = np.empty(n_risks_sources_local)
    best_risk = inf_local
    best_cost = inf_local
    while fill_index < local_max_costs_shape:
        if thread_optimal_risks_local[fill_index] < best_risk:
            best_risk = thread_optimal_risks_local[fill_index]
            best_cost = thread_optimal_costs_local[fill_index]
            best_strat = thread_best_strategy_local[fill_index]
        else:
            thread_optimal_risks_local[fill_index] = best_risk
            thread_optimal_costs_local[fill_index] = best_cost
            thread_best_strategy_local[fill_index] = best_strat
        fill_index += 1

    return thread_optimal_risks_local, thread_optimal_costs_local, thread_best_strategy_local


if __name__ == "__main__":
    # 12 risks
    # 5 levels (by default)
    # you can add more of them using BaseClass methods
    BaseClass.optimize_for_all_costs(multiprocessing_mode=True)
    BaseClass.save_optimal_strategy_curve()

    # print('best strategies: \n', BaseClass.optimal_strategies, '\n\n\n')
    print('best risks with those: \n', BaseClass.optimal_risks, '\n\n\n')  # to big array to print (3655 elements) by default
    print('and the costs of strats: \n', BaseClass.optimal_costs, '\n\n\n')
    print(f'optimal for cost 50: \n{BaseClass.get_optimal_strategy_with_risk_and_cost_given_budget(35)}\n\n\n')
    print('finished')
