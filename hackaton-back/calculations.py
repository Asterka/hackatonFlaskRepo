# import warnings
import json
import numpy as np
from matplotlib import pyplot as plt
from numba import prange, njit


# it is better to install numba with conda (for llvm support)

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
        row_with_approach_names = np.arange(self.n_approaches + 1).astype(np.object)
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
        to_dump = np.append(column_wth_risks_names, np.transpose(to_dump, axes=(1, 0)), axis=1)

        to_dump = np.append(row_with_approach_names[None, :], to_dump, axis=0)
        return json.dumps(to_dump.astype(str), cls=NumpyEncoder)

    def read_from_json(self, json_data, shape='2d', dtype='float'):
        """
        reconstruct a table from json format to numpy array
        :param json_data: data
        :param shape: 2d or 3d (3d for risks table)
        :param dtype: float always except reasoning table
        :return: actually returning value can be ignored.
                 Returns np array of reconstructed data.
        """
        #
        json_load = json.loads(json_data)
        to_data = np.asarray(json_load)[1:, 1:]
        if shape != '2d':
            new_to_data = np.empty((to_data.shape[0], to_data.shape[1], 2), dtype=float)
            for i in range(to_data.shape[0]):
                for j in range(to_data.shape[1]):

                    pair = to_data[i, j].split(', ')

                    new_to_data[i, j, 0] = float(dmg_lvls_inversed[pair[0]])
                    new_to_data[i, j, 1] = float(probability_inversed[pair[1]])
            to_data = new_to_data.transpose((1, 0, 2))
        else:
            to_data = to_data.transpose((1, 0))
        if dtype == 'float':
            self.data = to_data.astype(float)  # comment this line for testing this function
            return to_data.astype(float)
        else:
            self.data = to_data  # comment this line for testing this function
            return to_data


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
            self.data = np.empty((self.n_approaches, self.n_risks_sources), dtype=str)

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

    max_costs = None
    optimal_risks = None
    optimal_costs = None
    is_relevant = None
    risk_cost = 2.

    @classmethod
    def save_optimal_strategy_curve(cls):
        """
            Represents minimal risk for each available budget and optimal point for a given risk_cost field of the class
        :return:  optimal budget that minimizes risk cost and risk management cost
        """
        plt.plot(cls.max_costs, cls.optimal_risks)

        optimal_point = np.argmin(
            cls.risk_cost * cls.optimal_risks + cls.optimal_costs)  # consider cost of data as twice cost of the solution
        plt.scatter(cls.max_costs[optimal_point], cls.optimal_risks[optimal_point], color='g')
        plt.gca().set_ylim(top=100, bottom=0)
        # plt.show()
        plt.xlabel('Стоимость баз. ед.')
        plt.ylabel('Риск в ед. риска')
        # plt.savefig('optimal_strategy_curve.png')
        return optimal_point

    @classmethod
    def optimize_for_all_costs(cls, costs_list=None, n_steps=None):
        """
        Calculates minimal risk for each available budget and optimal point for a given risk_cost field of the class
        Solution is simply brute force (that can be optimized by going through each combination just once, not for each
        cost: that was done for simple parallelization with numba, numba can be replaced with another lib with
        an opportunity to explicitly change number of workers to optimize this way)

        As soon as it is MVP we didn't consider smart update of the calculations after updating values: all calculations
        repeats to update the relevant solution

        :param costs_list: list of costs (by default: all possible costs)
        :param n_steps: number of steps for optimization (by default as many steps as there are costs_list elements)
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

            n_steps = int((max_theoretical_cost - min_theoretical_cost) / min_step) + 1
            max_costs = np.linspace(min_theoretical_cost, max_theoretical_cost, n_steps)

        else:
            max_costs = sorted(list(costs_list)[:])

        costs = BaseClass.costs_table.data[:]
        risks = BaseClass.risks_table.data[:]
        n_risks_sources = BaseClass.costs_table.n_risks_sources
        n_approaches = BaseClass.costs_table.n_approaches

        @njit
        def calc_cost_and_risk(risk_strategy, max_curr_cost=2147483647., opt_strategy_score=2147483647.):
            # calculates cost and risk of the single strategy
            """

            :param risk_strategy: 12 different risk sources (each has value from 0 to 4 according to level of security)
            :param max_curr_cost: max cost that this strategy can have (if it is more expensive strategy will not be considered)
            :param opt_strategy_score: optimal found strategy risk, more risky strategy will not be considered
            :return: nothing, just creates an image with plot of the solution
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
        def base_repr(x, base, digits_n):
            # converts x from base 10 to base "base"
            # returns np.array of digits of length digits_n
            # used to encode all states of the brute force
            # unfortunately np.ndindex does not work with parallel prange in numba, so here is the alternative :)
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
                digits[-digit-1] = d
                x = x - d * m
            return digits

        @njit
        def calc_opt_strategy_score_fixed_cost(max_curr_cost, options):
            # calculates best strategy and it's cost by a given max_curr_cost budget limitation
            inf = 2147483647
            opt_strategy_score = inf
            opt_strategy_cost = inf
            # brute force: going through all possible strategies as simple as possible
            # (can be optimized, but for our purposes time limits are satisfied)
            # ndindex and other similar solutions for permutations doesn't with numba with prange for some reason, soo....
            for risk_strategy_base10 in range(options):
                risk_strategy = base_repr(risk_strategy_base10, n_approaches, n_risks_sources)
                cost, curr_risk = calc_cost_and_risk(risk_strategy,
                                                     max_curr_cost,
                                                     opt_strategy_score)
                if curr_risk < opt_strategy_score and cost <= max_curr_cost:
                    opt_strategy_score = curr_risk
                    opt_strategy_cost = cost
            return opt_strategy_score, opt_strategy_cost

        @njit(parallel=True)
        def calc_opt_strategy_score(n_risk_sources=n_risks_sources, n_approaches=n_approaches):
            # calculates the best risk management strategy for all ranges of budget possible according to the data
            # returns two arrays with equal sizes - the smallest risk possible with the given budget[i]
            # solution is implemented in a brute force fashion
            optimal_risks_ = np.empty_like(max_costs)
            optimal_costs_ = np.empty_like(max_costs)
            options = n_approaches**n_risk_sources  # all possible strategies
            for index in prange(max_costs.shape[0]):
                # this loop can be eliminated: all the optimal strategies should be found in a one look
                # but then either no parallel computing or no numba prange feature can be used sadly
                optimal_risks_[index], optimal_costs_[index] = calc_opt_strategy_score_fixed_cost(max_costs[index], options)
                # we can say it equals max_curr_cost, but we should have a fair calculation :)
            return optimal_risks_, optimal_costs_

        @njit
        def calc_opt_rs(max_curr_cost=45.0, n_risk_sources=n_risks_sources, n_approaches=n_approaches):
            # calculates optimal strategy for a given max_curr_cost
            # no parallel execution due to conflict with the shared variables best_rs, opt_strategy
            inf = 2147483647
            opt_strategy_score = inf
            best_rs = np.arange(n_risks_sources)
            options = n_approaches ** n_risk_sources  # all possible strategies
            for risk_strategy_base10 in range(options):
                risk_strategy = base_repr(risk_strategy_base10, n_approaches)

                cost, curr_risk = calc_cost_and_risk(risk_strategy,
                                                     max_curr_cost,
                                                     opt_strategy_score)
                if curr_risk < opt_strategy_score and cost <= max_curr_cost:
                    opt_strategy_score = curr_risk
                    best_rs[:] = risk_strategy[:]

            return best_rs

        def global_run():
            # finding optimal data, strategies and optimal point
            # but not optimal strategy, because it violates rule of shared memory for multiprocessing with numba
            optimal_risks, optimal_costs = calc_opt_strategy_score()
            BaseClass.optimal_risks = optimal_risks
            BaseClass.optimal_costs = optimal_costs
            BaseClass.max_costs = max_costs
            optimal_point = BaseClass.save_optimal_strategy_curve()
            return optimal_point, max_costs[optimal_point], optimal_risks[optimal_point]

        _, optimal_budget, optimal_risk = global_run()
        # print(f"the best risk management strategy with cost(risk)/cost(rubbles for risk management) ~ 2 costs \
        #             {optimal_cost__}, leads to average risk = {optimal_risk} and can be achieved by the strategy: "
        #       f"{calc_opt_rs(max_curr_cost=optimal_cost__)}")
        BaseClass.is_relevant = True
        # return best_strategy = calc_opt_rs(max_curr_cost=optimal_budget)

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


if __name__ == "__main__":
    # 12 risks
    # 5 levels

    BaseClass.optimize_for_all_costs()
    # BaseClass.remove_risk_source()
    # BaseClass.remove_risk_source(3)
    # BaseClass.remove_approach()
    # BaseClass.remove_approach(2)
    # BaseClass.add_risk_source()
    # BaseClass.add_approach()
    print('finished')
