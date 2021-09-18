import numpy as np
from matplotlib import pyplot as plt
from numba import prange, njit
import json


# it is better to install numba with conda (for llvm support)


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class Table:
    # deferred class
    def __init__(self, n_risks_sources=12, n_approaches=5):
        self.data = np.empty(0)
        self.n_risks_sources = n_risks_sources
        self.n_approaches = n_approaches

    def update_value(self, new_val, x, y):
        self.data[x, y] = new_val
        BaseClass.is_relevant = False  # BaseClass optimal_strategy_curve is not relevant anymore

    def add_row(self):
        self.data = np.append(self.data, np.zeros(self.data.shape[0], dtype=self.data.dtype), axis=0)

    def add_column(self):
        self.data = np.append(self.data, np.zeros(self.data.shape[1], dtype=self.data.dtype), axis=1)

    def convert_numpy_to_json(self):
        return json.dumps(self.data, cls=NumpyEncoder)

    def read_from_json(self, json_data):
        json_load = json.loads(json_data)
        self.data = np.asarray(json_load)
        return self.data


class RiskTable(Table):
    def __init__(self, default=True, init_risks=None):
        super(RiskTable, self).__init__()
        if default:
            # shape (5: for each security lvl approach including base approach, 12: for each risk_source, 2: damage & probability)

            # security lvl dimensions:
            # 0: 1 ур. проработки
            # 1: 2 ур. проработки
            # 2: 3 ур. проработки
            # 3: 4 ур. проработки

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
            self.data = init_risks[:]


class CostsTable(Table):
    def __init__(self, default=True, init_costs=None):
        super(CostsTable, self).__init__()
        if default:
            self.data = np.empty((self.n_approaches, self.n_risks_sources))
            self.data[0] = np.zeros(self.n_risks_sources)
            self.data[1] = [2.5, 3, 3.5, 2., 1.5, 1., 2., 4., 5., 1., 1.5, 2.]
            self.data[2] = [3.125, 3.75, 4.375, 2.5, 1.875, 1.25, 2.5, 5., 6.25, 1.25, 1.875, 2.5]
            self.data[3] = [4.375, 5.25, 6.125, 3.5, 2.625, 1.75, 3.5, 7., 8.75, 1.75, 2.625, 3.5]
            self.data[4] = [7.875, 9.45, 11.025, 6.3, 4.725, 3.15, 6.3, 12.6, 15.75, 3.15, 4.725, 6.3]
        else:
            self.data = init_costs[:]


class Reasoning(Table):
    def __init__(self, default=True, init_reasoning=None):
        super(Reasoning, self).__init__()
        if default and not init_reasoning:
            self.data = np.empty((self.n_approaches, self.n_risks_sources), dtype=str)

            self.data[0] = ["11", "12", "13", "14",
                            "15", "16", "17", "18",
                            "19", "1_10", "1_11", "1_12"]
            self.data[1] = ["21", "22", "23", "24",
                            "25", "26", "27", "28",
                            "29", "2_10", "2_11", "2_12"]
            self.data[2] = ["31", "32", "33", "34",
                            "35", "36", "37", "38",
                            "39", "3_10", "3_11", "3_12"]
            self.data[3] = ["41", "42", "43", "44",
                            "45", "146", "47", "48",
                            "49", "4_10", "4_11", "4_12"]
            self.data[4] = ["51", "52", "53", "54",
                            "55", "56", "57", "58",
                            "59", "5_10", "5_11", "5_12"]

        elif not init_reasoning:
            self.data = np.empty((self.n_approaches, self.n_risks_sources), dtype=str)
        else:
            self.data = np.array(init_reasoning, dtype=str)


class BaseClass(metaclass=Singleton):
    risks_table = RiskTable()
    costs_table = CostsTable()
    reasons_table = Reasoning()
    max_costs = None
    optimal_risks = None
    optimal_costs = None
    is_relevant = None

    @classmethod
    def save_optimal_strategy_curve(cls, external=False):
        plt.plot(cls.max_costs, cls.optimal_risks)

        optimal_point = np.argmin(
            2 * cls.optimal_risks + cls.optimal_costs)  # consider cost of data as twice cost of the solution
        plt.scatter(cls.max_costs[optimal_point], cls.optimal_risks[optimal_point], color='g')
        plt.gca().set_ylim(top=100, bottom=0)
        # plt.show()
        plt.xlabel('Стоимость баз. ед.')
        plt.ylabel('Риск в ед. риска')
        plt.savefig('optimal_strategy_curve.png')
        return optimal_point

    @classmethod
    def optimize_for_all_costs(cls, costs_list=None, n_steps=None):

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
        # print(f'Number of steps: {n_steps}')

        @njit
        def calc_cost_and_risk(risk_strategy, max_curr_cost=2147483647., opt_strategy_score=2147483647.):
            # calculates cost and risk of the single strategy
            """

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
        def base_repr(x, base, digits_n):
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
            # this loop can be eliminated: all the optimal strategies should be found in a one look
            options = n_approaches**n_risk_sources  # all possible strategies
            for index in prange(max_costs.shape[0]):
                optimal_risks_[index], optimal_costs_[index] = calc_opt_strategy_score_fixed_cost(max_costs[index], options)
                # we can say it equals max_curr_cost, but we should have a fair calculation :)
            return optimal_risks_, optimal_costs_

        @njit
        def calc_opt_rs(max_curr_cost=45.0, n_risk_sources=n_risks_sources, n_approaches=n_approaches):
            inf = 2147483647
            opt_strategy_score = inf
            best_rs = np.arange(n_risks_sources)
            # as soon as no parallel execution is possible due to conflict with the shared variables best_rs, opt_strategy
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

        _ = global_run()
        # print(f"the best risk management strategy with cost(risk)/cost(rubbles for risk management) ~ 2 costs \
        #             {optimal_cost__}, leads to average risk = {optimal_risk} and can be achieved by the strategy: "
        #       f"{calc_opt_rs(max_curr_cost=optimal_cost__)}")
        BaseClass.is_relevant = True

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
    def add_row_to_risk(cls):
        cls.risks_table.add_row()

    @classmethod
    def add_row_to_costs(cls):
        cls.costs_table.add_row()

    @classmethod
    def add_row_to_reasons(cls):
        cls.reasons_table.add_row()

    @classmethod
    def add_column_to_risk(cls):
        cls.risks_table.add_column()

    @classmethod
    def add_column_to_costs(cls):
        cls.costs_table.add_column()

    @classmethod
    def add_column_to_reasons(cls):
        cls.reasons_table.add_column()


if __name__ == "__main__":
    """json_f = BaseClass.risks_table.convert_numpy_to_json()
    X = BaseClass.risks_table.data[:]
    BaseClass.risks_table.read_from_json(json_f)
    print(BaseClass.risks_table.data == X)"""
    BaseClass.optimize_for_all_costs()
