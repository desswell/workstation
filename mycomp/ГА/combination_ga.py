import random
import copy
import json
import numpy as np
from typing import Union
import time
import pandas as pd
from baumeva import GaData
from baumeva.ga.populations import OrderCatPopulation, BinaryPopulation
from baumeva.ga.fitness import HyperbolaFitness
from baumeva.ga.selections import TournamentSelection, BalancedSelection, RankedSelection
from baumeva.ga.crossovers import OrderCrossover, OnePointCrossover, TwoPointCrossover
from baumeva.ga.mutations import InversionMutation, SwapMutation, MovementMutation, ShiftMutation, BinStringMutation
from baumeva import NewGeneration
import plotly.graph_objects as go
import plotly.io as pio
import itertools
import shutil
import os


class LogisticsDataLoader:
    """
    Класс загрузки данных для логистической задачи
    :code_assign: service
    :code_type: Анализ данных
    :packages:
    import pandas as pd
    """

    def __init__(self,
                 sdm_path: str = None,
                 config_path: str = None,
                 utilization: int = 0.8,
                 max_capacity: int = 86):
        self.sdm_path = sdm_path
        self.config_path = config_path
        self.utilization = utilization
        self.max_capacity = max_capacity
        self.capacity = max_capacity * self.utilization

        self.mkr = {}
        self.ic_rates = {}
        self.lh_rates = {}
        self.lm_rates = {}
        self.unsupported_terminals = []
        self.required_terminals = []
        self.proposed_terminals = []
        self.preset_tariffs = {}
        self.extra_point_price = 0
        self.replace_cities = {}
        self.directions = {}

        self.process_data()

    def get_available_terminals(self, orders):
        """
        Получаем возможные терминалы
        :param orders: все заказы
        :return: возможные терминалы
        """
        terminals = []

        for order in orders:
            available_terminals = list(order.keys())

            # Избавляемся от городов в которых не может быть терминала
            available_terminals = list(set(available_terminals) - set(self.unsupported_terminals))

            # Добавляем обязательные терминалы
            # Реализовать

            # Оставляем только рекомендуемые терминалы
            if len(self.proposed_terminals) != 0:
                available_terminals = [term for term in available_terminals if term in self.proposed_terminals]
            terminals.append(available_terminals)

        return terminals

    def dispatches_division(self, orders_list, weekdays):
        """
        Разделение заказов на отправки
        :param weekdays: список дней отправок
        :return: список отправок
        """
        weekdays.sort()
        dispatches = dict()
        d = 1
        for day in range(0, 7):
            dispatches[day] = d
            if day in weekdays and day != weekdays[-1]:
                d += 1
            if day in weekdays and day == weekdays[-1]:
                dispatches[day] = d
                break

        if len(weekdays) != 0:
            for day in range(weekdays[-1] + 1, 7):
                dispatches[day] = 1

        dispatch_num = 1
        dispatch_column = []
        prev_dispatch_day = orders_list.iloc[0]['День недели']
        for day in orders_list['День недели'].iloc[1:]:
            dispatch_column.append(dispatch_num)
            cur_dispatch_day = day
            if cur_dispatch_day != prev_dispatch_day:
                if len(weekdays) == 1 and cur_dispatch_day == weekdays[0]:
                    dispatch_num += 1
                elif len(weekdays) > 1 and dispatches[prev_dispatch_day] != dispatches[cur_dispatch_day]:
                    dispatch_num += 1
            prev_dispatch_day = cur_dispatch_day
        dispatch_column.append(dispatch_num)
        orders_list['Отправка'] = dispatch_column

        dispatches = [group for _, group in orders_list.groupby('Отправка')]

        return dispatches

    def get_orders(self, orders):
        """
        Анализ заказов
        :param orders: путь к файлу с заказазами
        :return: список отправок, список дат отправок
        """
        orders_list = []
        
        terminals = self.get_available_terminals(orders_list)

        return orders_list, terminals

    def process_chosen_terminals(self, path):
        try:
            terminals_data = pd.read_excel(path, sheet_name='Терминалы - города')
        except Exception as e:
            print(f"Ошибка выбранные терминалы: {e}")
            raise

        return terminals_data['Терминалы'].tolist()

    def process_data(self):
        """
        Анализ файла конфигурации
        """

        if self.sdm_path is None:
            raise Exception("Не передан файл с матрицей кратчайших расстояний")
        if self.config_path is None:
            raise Exception("Не передан файл с конфигурацией маршрутов")

        self.process_sdm()
        self.proccess_rates(['Тариф лайнхолл', 'Тариф последняя миля', 'Тариф внутригород'])
        self.process_terminals_config('Ограничения на терминалы')
        self.process_preset_tariffs('Тарифы, 82 м³')
        self.process_extra_points_price('Доп точки')
        self.process_replace_cities(['Города - внутригород', 'Города - наименования'])
        self.process_directions('Направления')

    def process_directions(self, sheet_name):
        """
        Получаем направления
        """
        try:
            directions_data = pd.read_excel(self.config_path, sheet_name=sheet_name)
        except Exception as e:
            print(f"Ошибка направления: {e}")
            raise

        self.directions = {direction: directions_data[direction].dropna().tolist()
                           for direction in directions_data.columns}

    def process_replace_cities(self, sheet_names):
        """
        Получаем пригороды и правописания городов
        """
        try:
            suburbs_data = pd.read_excel(self.config_path, sheet_name=sheet_names[0])
        except Exception as e:
            print(f"Ошибка пригороды: {e}")
            raise

        suburbs = dict(zip(suburbs_data.iloc[:, 1], suburbs_data.iloc[:, 0]))

        self.replace_cities.update(suburbs)

        try:
            city_naming_data = pd.read_excel(self.config_path, sheet_name=sheet_names[1])
        except Exception as e:
            print(f"Ошибка наименования городов: {e}")
            raise

        city_naming = dict(zip(city_naming_data.iloc[:, 1], city_naming_data.iloc[:, 0]))

        self.replace_cities.update(city_naming)

    def process_extra_points_price(self, sheet_name):
        """
        Получаем стоимость доп точек
        """
        try:
            extra_point_data = pd.read_excel(self.config_path, sheet_name=sheet_name)
        except Exception as e:
            print(f"Ошибка доп точки: {e}")
            raise

        extra_point_prices = dict(zip(extra_point_data.iloc[:, 0], extra_point_data.iloc[:, 1]))
        self.extra_point_price = extra_point_prices['Лайнхолл']

    def process_preset_tariffs(self, sheet_name):
        """
        Получаем предустановленные тарифы
        """
        try:
            preset_tariffs_data = pd.read_excel(self.config_path, sheet_name=sheet_name)
        except Exception as e:
            print(f"Ошибка предустановленные тарифы: {e}")
            raise

        # Преобразование столбца 'Куда' в списки городов
        preset_tariffs_data['Куда'] = preset_tariffs_data['Куда'].apply(lambda route:
                                                                        tuple(['Великий Устюг'] + route.split(' - ')))

        self.preset_tariffs = preset_tariffs_data.set_index('Куда')['Стоимость'].to_dict()

    def process_sdm(self):
        """
        Задаем матрицу кратчайших расстояний
        """

        try:
            mkr_data = pd.read_excel(self.sdm_path)
        except Exception as e:
            print(f"Ошибка мкр: {e}")
            raise

        for idx, row in mkr_data.iterrows():
            city = row.iloc[0]
            distances = row.iloc[1:]
            self.mkr[city] = distances.tolist()

    def process_terminals_config(self, sheet_name):
        """
        Получаем конфигурации терминалов
        """

        try:
            terminals_data = pd.read_excel(self.config_path, sheet_name=sheet_name)
        except Exception as e:
            print(f"Ошибка конфигурация терминалов: {e}")
            raise

        self.unsupported_terminals = list(terminals_data['Города, на запрет установки терминала'])
        self.required_terminals = list(terminals_data['Города, обязательные на установку терминала (не реализовано)'])
        self.proposed_terminals = list(terminals_data['Города, где можно установить терминал'].dropna())

    def proccess_rates(self, sheet_names):
        """
        Получаем тарифы
        """

        self.get_lh_rates(sheet_names[0])
        self.get_lm_rates(sheet_names[1])
        self.get_ic_rates(sheet_names[2])

    def get_ic_rates(self, sheet_name):
        """
        Получаем тариф на внутригород
        """
        try:
            innercity_data = pd.read_excel(self.config_path, sheet_name=sheet_name)
        except Exception as e:
            print(f"Ошибка тарифы внутригород: {e}")
            raise

        ic_rates_dict = innercity_data.set_index(innercity_data.columns[0]).agg(list, axis=1).to_dict()

        for idx, value in enumerate(ic_rates_dict['Объем, дм³']):
            self.ic_rates[value] = [ic_rates_dict['Тариф, мандарин/саней'][idx],
                                    ic_rates_dict['Утилизация максимальная, %'][idx]]

    def get_lm_rates(self, sheet_name):
        """
        Получаем тариф на последнюю милю
        """
        try:
            lastmile_data = pd.read_excel(self.config_path, sheet_name=sheet_name)
        except Exception as e:
            print(f"Ошибка тарифы последняя миля: {e}")
            raise

        lm_rates_dict = lastmile_data.set_index(lastmile_data.columns[0]).agg(list, axis=1).to_dict()

        for idx, value in enumerate(lm_rates_dict['Объем, дм³']):
            self.lm_rates[value] = [lm_rates_dict['Тариф, мандарин/км'][idx],
                                    lm_rates_dict['Утилизация максимальная, %'][idx]]

    def get_lh_rates(self, sheet_name):
        """
        Получаем тариф на лайнхол
        """
        try:
            linehaul_data = pd.read_excel(self.config_path, sheet_name=sheet_name)
        except Exception as e:
            print(f"Ошибка тарифы лайнхол: {e}")
            raise

        min_rate, max_rate, min_distance, max_distance = linehaul_data.iloc[:, 1].values

        iterrator = 0
        distances = list(self.mkr.values())[list(self.mkr.keys()).index('Великий Устюг')]

        for city in list(self.mkr.keys()):
            if city != 'Великий Устюг':
                self.lh_rates[city] = max_rate - (max_rate - min_rate) * (
                        (distances[iterrator] - min_distance) / (max_distance - min_distance))

                if self.lh_rates[city] < min_rate:
                    self.lh_rates[city] = min_rate
                elif self.lh_rates[city] > max_rate:
                    self.lh_rates[city] = max_rate

            iterrator += 1


class LogisticsBudgetModule:
    """
    Класс подсчета бюджета для логистической задачи
    :code_assign: service
    :code_type: Анализ данных
    :packages:
    import pandas as pd
    """

    def __init__(self,
                 mkr: dict = {},
                 extra_point_price: int = 0,
                 innercity_rates: dict = {},
                 lastmile_rates: dict = {},
                 linehaul_rates: dict = {},
                 preset_tariffs: dict = {},
                 capacity: int = 0):
        self.mkr = mkr
        self.extra_point_price = extra_point_price
        self.innercity_rates = innercity_rates
        self.lastmile_rates = lastmile_rates
        self.linehaul_rates = linehaul_rates
        self.preset_tariffs = preset_tariffs
        self.capacity = capacity

    def recover_cars(self, dp, capacities):
        """
        Функция для поиска выбранных машин
        :param list dp: матрица выбора весов
        :return: выбранные машины
        """
        weight_to_deliver = len(dp[0]) - 1
        car_idx = len(capacities)
        chosen_cars = {}

        while weight_to_deliver > 0 and car_idx > 0:
            if dp[car_idx][weight_to_deliver] != dp[car_idx - 1][weight_to_deliver]:
                if capacities[car_idx - 1] in chosen_cars:
                    chosen_cars[capacities[car_idx - 1]] += 1
                else:
                    chosen_cars[capacities[car_idx - 1]] = 1
                weight_to_deliver -= capacities[car_idx - 1]
            else:
                car_idx -= 1

        return chosen_cars

    def get_cars_optimization(self, weight, tariff):
        """
        Поиск оптималього распределения машин
        :param weight: вес для развоза
        :param tariff: тариф для машин
        :return: словарь в виде {грузоподъемность: количество}
        """
        weight = int(round(weight))
        if weight == 0:
            weight = 1
        capacities = [int(int(key) * (int(tariff[key][1]) / 100)) for key in list(tariff.keys())]
        mapping = {int(int(key) * (int(tariff[key][1]) / 100)): key for key in list(tariff.keys())}
        costs = [value[0] for value in list(tariff.values())]
        num_cars = len(capacities)
        dp = [[float('inf')] * (weight + 1) for _ in range(num_cars + 1)]
        dp[0][0] = 0

        for car_idx in range(1, num_cars + 1):
            for current_weight in range(weight + 1):
                if capacities[car_idx - 1] <= current_weight:
                    dp[car_idx][current_weight] = min(dp[car_idx - 1][current_weight],
                                                      dp[car_idx][current_weight - capacities[car_idx - 1]] + costs[
                                                          car_idx - 1])
                else:
                    dp[car_idx][current_weight] = min(dp[car_idx - 1][current_weight], costs[car_idx - 1])
        cars = self.recover_cars(dp, capacities)
        return dp[num_cars][weight], {mapping[key]: cars[key] for key in cars}

    def get_distance(self, city1, city2) -> int:
        """
        Получаем расстояние между двумя городами

        :param str city1: Первый город
        :param str city2: Второй город
        :return: Дистанция между городами
        """
        idx = list(self.mkr.keys()).index(city2)
        return self.mkr[city1][idx]

    def calculate_ftl_budget(self, terminals):
        """
        Считаем стоимость ftl
        :param terminals: терминалы
        :return:
        """
        terminals_info = []
        for t in list(terminals.keys()):
            terminals_info.append(
                {
                    'Город': t,
                    'Привязанные города': terminals[t][0],
                    'Отправлено оленей': terminals[t][1] // self.capacity,
                    'Стоимость': int(terminals[t][1] // self.capacity * self.calculate_route_budget(['Великий Устюг', t]))
                }
            )
            terminals[t] = terminals[t][1] % self.capacity

        return terminals_info

    def calculate_innercity_budget(self, terminals):
        """
        Получаем стоимость внутригорода

        :return: информация о внутригороде в виде словаря и бюджет
        """
        ic_budgets = list()

        for term in terminals:
            price, distribution = self.get_cars_optimization(terminals[term], self.innercity_rates)
            ic_budgets.append({
                'Город': term,
                'Объем доставки, дм³': terminals[term],
                'Отправлено оленей': distribution,
                'Стоимость': price
            })

        return ic_budgets

    def calculate_last_mile_budget(self, terminal, city, weight, distance):
        """
        Рассчет стоимости последней мили

        :return: стоимость последней мили
        """
        price, distribution = self.get_cars_optimization(weight, self.lastmile_rates)

        lh_budget = {
            'Город': city,
            'Объем доставки, дм³': weight,
            'Терминал': terminal,
            'Расстояние до терминала': distance,
            'Отправлено оленей': distribution,
            'Стоимость': price * distance
        }

        return lh_budget

    def calculate_route_budget(self, route):
        """
        Считаем бюджет лайнхолла по указанному пути
        :param route: путь доставки
        :return: бюджет
        """

        if tuple(route) in self.preset_tariffs.keys():
            return self.preset_tariffs[tuple(route)]

        destination_city = route[-1]
        rate = self.linehaul_rates[destination_city]
        distance = 0
        extra_budget = 0
        for i in range(1, len(route)):
            extra_budget += self.extra_point_price
            distance += self.get_distance(route[i - 1], route[i])

        return int(distance * rate) + extra_budget

    def get_budget_from_info(self, structures_list):
        """
        Подсчитываем бюджет из структуры
        :return:
        """
        budget = 0
        for structure in structures_list:
            budget += structure['Стоимость']

        return budget


class LogisticsResultCollector:
    """
    Формирование результатов отработки
    :code_assign: service
    :code_type: Анализ данных
    :imports: Window, Canvas, GeoDataPlot, init_gui_dict
    :packages:
    import plotly.graph_objects as go
    import plotly.io as pio
    import pandas as pd
    import json
    import copy
    import numpy as np
    import random
    """

    def __init__(self, top_five):
        self.top_five = top_five
        self.first_dispatch = False
        self.info_to_table = list()
        self.for_table = []
        self.fig_to_vis = go.Figure()
        self.fig = go.Figure()
        self.total = str()
        self.map = go.Figure()
        self.coordinates = pd.DataFrame()
        self.gui_dict = init_gui_dict()
        self.cities = []

    def create_map(self):
        """
        Создание базовой карты России с помощью Plotly
        """
        with open("/usr/share/file-storage/useful_files/GA/russia_regions.geojson", "r",
                  encoding="utf-8") as geojson_file:
            geojson_data = json.load(geojson_file)
        # Создание базовой карты России с помощью Plotly
        self.map = go.Figure(go.Choroplethmapbox(
            geojson=geojson_data,
            locations=[],
            z=[],
            colorscale="Viridis",
            zmin=0,
        ))

    def get_coordinates(self, route: list) -> dict:
        """
        Функция, которая возвращает координаты городов, которые были использованы
        :param list route: список маршрутов
        :return: словарь координат
        """
        self.coordinates = pd.read_csv('/usr/share/file-storage/useful_files/GA/coordinates_russia.csv')
        routes = set()
        for unit in route:
            for cities in unit['cities']:
                routes.add(cities)
        cities = {"Великий Устюг": {"lat": 60.7603913, "lon": 46.3054414}}
        for index, row in self.coordinates.iterrows():
            city_name = row['Город']
            if city_name in routes:
                lat = row['Широта']
                lon = row['Долгота']
                cities[city_name] = {'lat': lat, 'lon': lon}
        return cities

    def plotly_visualize(self, routes: list, terminals: dict):
        """
        Функция для отрисовки карты маршрута
        :param list routes: список маршрутов
        :param dict terminals: список терминалов
        :return: plotly карта
        """
        self.fig = copy.deepcopy(self.map)
        scale_factor = 4
        terminals_list = ['Великий Устюг'] + list(terminals.keys())
        terminals_weight = [0] + list(terminals.values())
        # Загрузка геоданных регионов России из файла russia_regions.geojson
        self.cities = self.get_coordinates(routes)
        for city, city_data, terminal, weight in zip(list(self.cities.keys()), list(self.cities.values()),
                                                     terminals_list, terminals_weight):
            lat = city_data["lat"]
            lon = city_data["lon"]
            if city != 'Великий Устюг':
                color = 'red'
                log_weight = np.log(float(f"{weight:.1f}") + 1)
                size = int(scale_factor * log_weight)
            else:
                color = 'blue'
                size = 10

            self.fig.add_trace(go.Scattermapbox(
                mode="markers",
                lon=[lon],
                lat=[lat],
                marker=go.scattermapbox.Marker(
                    size=size,
                    color=color,
                ),
                text=[city],
                hoverinfo="text",
            ))

        available_colors = ['#FF5733',  # Красный
                            '#3366FF',  # Синий
                            '#990099',  # Фиолетовый
                            '#000000',  # Черный
                            '#FF9900',  # Оранжевый
                            '#663300',  # Коричневый
                            '#FF66FF',  # Розовый
                            '#00CC66',  # Темно-зеленый
                            '#FFFF00',  # Желтый
                            '#00FF00',  # Зеленый
                            '#800080',  # Пурпурный
                            '#00FFFF',  # Голубой
                            '#FFCC00',  # Желтый-оранжевый
                            '#6633CC',  # Фиолетовый
                            '#339966',  # Темно-зеленый
                            '#CC6600',  # Коричневый
                            '#669999',  # Серо-голубой
                            '#993333',  # Темно-красный
                            '#333333'  # Темно-серый
                            ]

        for route in routes:
            cities_list = route["cities"]
            route_price = route["route_price"]
            disposal = route['disposal']

            line_color = random.choice(available_colors)
            available_colors.remove(line_color)

            arc_lon = []
            arc_lat = []

            for city_index in range(len(cities_list) - 1):
                start_city = cities_list[city_index]
                end_city = cities_list[city_index + 1]

                start_lat = cities[start_city]["lat"]
                start_lon = cities[start_city]["lon"]
                end_lat = cities[end_city]["lat"]
                end_lon = cities[end_city]["lon"]

                num_points = 100
                for i in range(num_points + 1):
                    frac = i / num_points
                    arc_lon.append(start_lon + frac * (end_lon - start_lon))
                    arc_lat.append(start_lat + frac * (end_lat - start_lat))



            self.fig.add_trace(go.Scattermapbox(
                mode="lines",
                lon=arc_lon,
                lat=arc_lat,
                line=go.scattermapbox.Line(width=2, color=line_color),
                hoverinfo="text",
                text=f"Маршрут: {' - '.join(cities_list)}<br>Цена перевоза: {route_price}<br>Коэффициент утилизации: "
                     f"{disposal}",
                showlegend=False,
                opacity=0.5,
            ))

        # Настройка стилей при наведении на маршрут
        self.fig.update_traces(
            hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial"),
            # Белый фон и стиль шрифта для всплывающей подсказки
        )

        self.fig.update_layout(
            mapbox=dict(
                style="open-street-map",
                center=dict(lat=60, lon=100),
                zoom=2
            ),
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            showlegend=False,
            hovermode='closest',  # Изменение режима наведения
        )

    def accumulate_table(self, key, df_orders, df_linehaul_first_stage, df_terminals_last,
                         df_linehaul_second_stage, df_lastmile, df_innercity, orders):
        """
        Функция для записи данных для короткой сводки
        :param key
        :param df_orders: датасет с заказами
        :param df_linehaul_first_stage: датасет с лайнхоллами первой стадии
        :param df_terminals_last: датасет с оставшимися поездками
        :param df_linehaul_second_stage: датасет с лайнхоллами первой стадии
        :param df_lastmile: последняя миля
        :param df_innercity: внутригород
        :param orders: заказы
        """
        if self.first_dispatch:
            return
        new_str = {'Город': 'Total', 'Объем доставки, дм³': f'{sum(orders.values()):.2f}'}
        new_df = pd.DataFrame([new_str])
        df_orders = pd.concat([df_orders, new_df], ignore_index=True)
        self.fig_to_vis = self.fig
        self.info_to_table.append(df_orders)
        self.info_to_table.append(df_linehaul_first_stage)
        self.info_to_table.append(df_terminals_last)
        self.info_to_table.append(df_linehaul_second_stage)
        self.info_to_table.append(df_lastmile)
        self.info_to_table.append(df_innercity)
        self.total = f'Total: {sum(orders.values()):.2f}'
        terminals_before = self.top_five[key]['terminals_before']
        for name in terminals_before:
            terminals_before[name] = f'{terminals_before[name][1]:.1f} дм³'
        self.for_table = [key, terminals_before]

        self.first_dispatch = True

    def get_results_into_exe(self, fn, path_output, orders, cap_max):
        with pd.ExcelWriter(fn) as writer:
            all_results = {

            }
            if len(self.top_five) == 0:
                all_results[f'Решения нет'] = 0
                result_dfs = []
                result_dfs.extend([pd.DataFrame([f'Решения нет. Нет доступных терминалов'], columns=[' '])])
                df_orders = pd.DataFrame(list(orders.items()), columns=['Город', 'Объем доставки, дм³'])

                result_dfs.extend([pd.DataFrame(['Исходные данные:'], columns=[' ']),
                                   df_orders,
                                   pd.DataFrame({' ': [''], ' ': [f'Total: {sum(orders.values()):.2f}']})])

                sr = 0
                for el in result_dfs:
                    el.to_excel(writer, sheet_name=f'Исходные данные', startrow=sr, index=False)
                    sr += len(el) + 1

            for i, key in enumerate(list(self.top_five.keys())):
                if key == 0:
                    break
                data = self.top_five[key]
                budget = key
                lh_info, terminals, routes, weights, lh_budgets, lm_info, ic_info = data['linehaul_info'], \
                                                                                    data['terminals'], \
                                                                                    data['routes'], \
                                                                                    data['weights'], \
                                                                                    data['lh_budgets'], \
                                                                                    data['last_mile_info'], \
                                                                                    data['ic_budgets']
                all_results[f'Решение {i + 1}'] = budget
                result_dfs = []
                result_dfs.extend([pd.DataFrame([f'Решение {i + 1}. Бюджет: {budget} рублей.'], columns=[' '])])
                df_orders = pd.DataFrame(list(orders.items()), columns=['Город', 'Объем доставки, дм³'])

                result_dfs.extend([pd.DataFrame(['Исходные данные:'], columns=[' ']),
                                   df_orders,
                                   pd.DataFrame({' ': [''], ' ': [f'Total: {sum(orders.values()):.2f}']})])

                df_linehaul_first_stage = pd.DataFrame(lh_info)
                df_linehaul_first_stage.columns = ['Терминал', 'Привязанные города', 'Отправлено оленей',
                                                   'Стоимость отправки']
                result_dfs.extend(
                    [pd.DataFrame(['Зонирование пунктов назначения + FTL (full-trauck load):'], columns=[' ']),
                     df_linehaul_first_stage])

                df_terminals_last = pd.DataFrame(list(terminals.items()), columns=['Терминал', 'Осталось развести, дм³'])
                result_dfs.extend([pd.DataFrame(['Остатки груза после этапа FTL:'], columns=[' ']),
                                   df_terminals_last])

                routes_str = [' - '.join(route) for route in routes]
                util_rate = [f'{(weight / cap_max) * 100:.2f}%' for weight in weights]

                if key == min(self.top_five.keys()) and path_output is not None:
                    # Трансформация для визуализации
                    dict_to_vis = []
                    for route, price, disposal in zip(routes, lh_budgets, util_rate):
                        dict_to_vis.append({'cities': route, 'route_price': price, 'disposal': disposal})
                    # Вузализация
                    self.plotly_visualize(dict_to_vis, terminals)

                    pio.write_html(self.fig, f'{path_output}/solution_{i + 1}.html')

                    self.gui_dict['plot'].append(
                        Window(
                            window_title='Карта',
                            canvases=[Canvas(title=f'Карта',
                                             showlegend=False,
                                             plots=[GeoTracePlot(cities=self.cities,
                                                                routes=dict_to_vis,
                                                                terminals=terminals
                                                                )]
                                             )
                                      ]
                        ).to_dict()
                    )

                    # Конец визуализации

                df_linehaul_second_stage = pd.DataFrame(
                    {'Путь': routes_str, 'Стоимость': lh_budgets, 'Загружено в сани, дм³': weights,
                     'Утилизация, %': util_rate})
                result_dfs.extend([pd.DataFrame(['Лайнхолл:'], columns=[' ']),
                                   df_linehaul_second_stage])
                if lm_info:
                    df_lastmile = pd.DataFrame(lm_info)
                    df_lastmile.columns = ['Город доставки', 'Объем доставки, дм³', 'Отправление из терминала',
                                           'Расстояние от терминала', 'Отправлены олени', ' Стоимость']
                    result_dfs.extend([pd.DataFrame(['Последняя миля:'], columns=[' ']),
                                       df_lastmile])

                df_innercity = pd.DataFrame(ic_info)
                df_innercity.columns = ['Город доставки', 'Объем доставки, дм³', 'Отправлено оленей', ' Стоимость']
                result_dfs.extend([pd.DataFrame(['Внутригород по терминалам:'], columns=[' ']),
                                   df_innercity])

                if key == min(self.top_five.keys()):
                    self.accumulate_table(key, df_orders, df_linehaul_first_stage, df_terminals_last,
                                          df_linehaul_second_stage, df_lastmile, df_innercity, orders)
                sr = 0
                for el in result_dfs:
                    el.to_excel(writer, sheet_name=f'Решение {i + 1}', startrow=sr, index=False)
                    sr += len(el) + 1

            all_results = pd.DataFrame(sorted(all_results.items(), key=lambda item: item[1]),
                                       columns=['Номер решения', 'Бюджет'])
            all_results.to_excel(writer, sheet_name=f'Сводка', startrow=2, index=False)


class Noytech:
    """
    Класс решения задачи Нойтека
    :code_assign: service
    :code_type: Анализ данных
    :imports: LogisticsDataLoader, LogisticsBudgetModule, LogisticsResultCollector

    :packages:
    import pandas as pd
    import itertools
    from baumeva import GaData, BinaryGA
    from baumeva.ga.populations import OrderCatPopulation, BinaryPopulation
    from baumeva.ga.fitness import HyperbolaFitness
    from baumeva.ga.selections import TournamentSelection, BalancedSelection, RankedSelection
    from baumeva.ga.crossovers import OrderCrossover, OnePointCrossover, TwoPointCrossover
    from baumeva.ga.mutations import InversionMutation, SwapMutation, MovementMutation, BinStringMutation, ShiftMutation
    from baumeva import NewGeneration
    """

    def __init__(self,
                 data_loader: LogisticsDataLoader,
                 budget_module: LogisticsBudgetModule,
                 dispatch: dict = {},
                 terminals: list = [],
                 np_first_level: int = 10,
                 ng_first_level: int = 10,
                 early_stop_first_level: int = 10,
                 selection_first_level: str = 'tournament',
                 crossover_first_level: str = 'single_point',
                 mutation_first_level: str = 'normal',

                 np_second_level: int = 10,
                 ng_second_level: int = 10,
                 early_stop_second_level: int = 10,
                 selection_second_level: str = 'tournament',
                 crossover_second_level: str = 'order',
                 mutation_second_level: str = 'inversion'
                 ):
        """
        Инициализация класса

        :param data_loader: Объект, предоставляющий доступ к данным для решения задачи логистики.
        :param budget_module: Объект, управляющий бюджетом для логистических операций.
        :param dispatch: Словарь с информацией о заказах.
        :param terminals: Список терминалов.
        :param np_first_level: Размер популяции для первого уровня генетического алгоритма.
        :param ng_first_level: Количество поколений (генераций) для первого уровня генетического алгоритма.
        :param early_stop_first_level: Критерий преждевременной остановки для первого уровня генетического алгоритма.
        :param selection_first_level: Метод селекции для первого уровня генетического алгоритма
                                    ('tournament', 'balanced' или 'ranked').
        :param crossover_first_level: Метод скрещивания для первого уровня генетического алгоритма
                                    ('single_point' или 'double_point').
        :param mutation_first_level: Метод мутации для первого уровня генетического алгоритма
                                   ('normal', 'weak' или 'strong').
        :param np_second_level: Размер популяции для второго уровня генетического алгоритма.
        :param ng_second_level: Количество поколений (генераций) для второго уровня генетического алгоритма.
        :param early_stop_second_level: Критерий преждевременной остановки для второго уровня генетического алгоритма.
        :param selection_second_level: Метод выбора особей для второго уровня генетического алгоритма ('tournament' или другие).
        :param crossover_second_level: Метод скрещивания для второго уровня генетического алгоритма ('order').
        :param mutation_second_level: Метод мутации для второго уровня генетического алгоритма
                                    ('inversion', 'swap', 'movement', 'shift').
        """
        self.dispatch = dispatch
        self.terminals = terminals
        self.data = data_loader
        self.budget_module = budget_module
        self.top_five = dict()

        # Задаем параметры для генетического алгоритма первого уровня
        self.selections = {
            'tournament': TournamentSelection,
            'balanced': BalancedSelection,
            'ranked': RankedSelection
        }

        self.crossovers_first_level = {
            'single_point': OnePointCrossover,
            'double_point': TwoPointCrossover
        }

        self.num_population_first_level = np_first_level
        self.num_generations_first_level = ng_first_level
        self.early_stop_first_level = early_stop_first_level

        self.selection_first_level = self.selections.get(selection_first_level, TournamentSelection)
        self.crossover_first_level = self.crossovers_first_level.get(crossover_first_level, TournamentSelection)
        self.mutation_first_level = mutation_first_level

        self.crossovers_second_level = {
            'order': OrderCrossover
        }

        self.mutations_second_level = {
            'inversion': InversionMutation,
            'swap': SwapMutation,
            'movement': MovementMutation,
            'shift': ShiftMutation
        }

        self.num_population_second_level = np_second_level
        self.num_generations_second_level = ng_second_level
        self.early_stop_second_level = early_stop_second_level

        self.selection_second_level = self.selections.get(selection_second_level, TournamentSelection)
        self.crossover_second_level = self.crossovers_second_level.get(crossover_second_level, OrderCrossover)
        self.mutation_second_level = self.mutations_second_level.get(mutation_second_level, InversionMutation)

    def second_level_params_set(self, selection, crossover, mutation):
        """
        Задаем параметры второго уровня
        :param selection: отбор
        :param crossover: скрещивание
        :param mutation: мутация
        :return:
        """
        self.selection_second_level = self.selections.get(selection, TournamentSelection)
        self.crossover_second_level = self.crossovers_second_level.get(crossover, OrderCrossover)
        self.mutation_second_level = self.mutations_second_level.get(mutation, InversionMutation)

    def find_terminal_to_city(self, terminals: list, city: str, weight: float):
        """
        Поиск терминала для города
        :param terminals: Доступные терминалы
        :param city: Город доставки
        :param weight: Объем доставки
        :return: Информация о терминале
        """
        # Ищем 3 ближайших терминала
        nearest_terms = {}
        for t in terminals:
            nearest_terms[t] = self.budget_module.get_distance(t, city)
        nearest_terms = dict(sorted(nearest_terms.items(), key=lambda item: item[1]))
        nearest_terms = dict(list(nearest_terms.items())[:3])

        # Из трех терминалов отбираем лучший путем сравнения их удельных стоимостей (стоимость 1 кубометра на лх + лм)
        terms_to_choose = {}
        for term in nearest_terms:
            lm_info = self.budget_module.calculate_last_mile_budget(term, city, weight, nearest_terms[term])

            lm_unit_budget = lm_info['Стоимость'] / lm_info['Объем доставки, дм³']
            lh_unit_budget = self.data.lh_rates[term]
            lh_unit_budget = lm_unit_budget + lh_unit_budget
            terms_to_choose[term] = [lm_info, lh_unit_budget]

        final_term = min(terms_to_choose, key=lambda term: terms_to_choose[term][1])

        return terms_to_choose[final_term][0]

    def bind_cities_to_terminal(self, terminals: dict, cities: dict):
        """
        Привязываем города доставки к терминалам
        :param terminals: Терминалы
        :param cities: Города
        :return: Информация о привязанных городах
        """
        binded_cities = {t: [[], terminals[t]] for t in terminals}
        cities_info = []
        for city in list(cities.keys()):
            lm_info = self.find_terminal_to_city(list(terminals.keys()), city, cities[city])
            cities_info.append(lm_info)
            binded_cities[lm_info['Терминал']][0].append(lm_info['Город'])
            binded_cities[lm_info['Терминал']][1] += lm_info['Объем доставки, дм³']
        return binded_cities, cities_info

    def cut_path(self, route: list, terminals: dict):
        """
        Нарезаем наш путь на маленькие пути для каждого автомобиля
        :param list route: Путь для нарезки, представляющий из себя длинный путь
        :param list terminals: Индивид ГА, представляющий из себя длинный путь
        :return: Индексы, разделяющие путь по машинам
        """
        routes = [[]]
        weights = []
        cur_weight = 0
        idxs = []
        for i, city_idx in enumerate(route):
            city = list(terminals.keys())[city_idx]
            if cur_weight + terminals[city] <= self.data.capacity and len(routes[-1]) < 4:
                cur_weight += terminals[city]
                routes[-1].append(city)
            else:
                idxs.append(i)
                weights.append(cur_weight)
                cur_weight = terminals[city]
                routes.append([city])
        weights.append(cur_weight)

        return idxs, routes, weights

    def route_target_func(self, terminals: dict, gens: list):
        """
        Целевая функция ГА второго уровня
        :param dict terminals: Словарь терминалов
        :param list gens: Список генов
        :return: Общий бюджет доставки
        """
        idxs, routes, weights = self.cut_path(gens, terminals)
        budget = 0
        for route in routes:
            route.insert(0, 'Великий Устюг')
            if route != ['Великий Устюг']:
                budget += self.budget_module.calculate_route_budget(route)
        return budget

    def update_top_five(self, new_data: dict, score: int):
        """
        Обновляет словарь top_five, добавляя new_data, если его score удовлетворяет условиям.

        :param new_data: Новый словарь для добавления.
        :param score: Ключ, по которому производится сравнение score.
        """
        if len(self.top_five) < 5:
            self.top_five[score] = new_data
        else:
            worst_score = max(self.top_five.keys())

            if score < worst_score:
                del self.top_five[worst_score]
                self.top_five[score] = new_data

        while len(self.top_five) > 5:
            worst_score = max(self.top_five.keys())
            del self.top_five[worst_score]

    def generate_binary_combinations(self, length):
        """
        Генерируем все возможные комбинации терминалов
        :param length: Количество городов в заказе
        :return: Комбинации терминалов
        """
        return list(itertools.product([0, 1], repeat=length))[1:]

    def ga_second_level(self, ind: list):
        """
        Функция, реализующая ГА второго уровня (задача VRP)
        :param ind: Конфигурация терминалов
        :return: Фитнес функция/бюджет для первого уровня
        """
        terminals = {term: self.dispatch[term] for term in [self.terminals[i] for i in range(len(ind)) if ind[i] == 1]}
        cities = {city: self.dispatch[city] for city in list(set(self.dispatch.keys()) - set(terminals.keys()))}

        if len(terminals) == 0:
            return 10e20

        inncercity_info = self.budget_module.calculate_innercity_budget(terminals)
        terminals, last_mile_info = self.bind_cities_to_terminal(terminals, cities)
        terminals_before_ftl = terminals.copy()
        ftl_info = self.budget_module.calculate_ftl_budget(terminals)

        if len(terminals) < 3:
            real_gens = [i for i in range(len(terminals))]
            population = {ind: self.route_target_func(terminals, list(ind)) for ind in
                          list(itertools.permutations(real_gens))}
            best_ind = min(population, key=population.get)
            budget = population[best_ind]
            idxs, routes, weights = self.cut_path(best_ind, terminals)

        else:
            ga_data = GaData(num_generations=self.num_generations_second_level,
                             early_stop=self.early_stop_second_level)
            population = OrderCatPopulation()
            population.set_params(num_individ=self.num_population_second_level,
                                  gens=(0, len(terminals) - 1, len(terminals)))
            fitness_func = HyperbolaFitness(obj_function=self.route_target_func,
                                            obj_value=0,
                                            input_data=terminals)
            selection = self.selection_second_level()
            crossover = self.crossover_second_level()
            mutation = self.mutation_second_level()
            new_generation = NewGeneration('best')

            population.fill()
            ga_data.population = population
            fitness_func.execute(ga_data=ga_data)
            ga_data.update()

            for i in range(ga_data.num_generations):
                selection.execute(ga_data)
                crossover.execute(ga_data)
                mutation.execute(ga_data)
                new_generation.execute(ga_data)
                fitness_func.execute(ga_data)
                ga_data.update()

                if ga_data.num_generation_no_improve >= ga_data.early_stop:
                    break

            budget = ga_data.best_solution['obj_score']
            idxs, routes, weights = self.cut_path(ga_data.best_solution['genotype'], terminals)

        infos = [inncercity_info, last_mile_info, ftl_info]
        for info in infos:
            budget += self.budget_module.get_budget_from_info(info)
        lh_budgets = []

        for route in routes:
            route.insert(0, 'Великий Устюг')
            lh_budgets.append(self.budget_module.calculate_route_budget(route))

        self.update_top_five({'terminals_before': terminals_before_ftl,
                              'terminals': terminals,
                              'linehaul_info': ftl_info,
                              'last_mile_info': last_mile_info,
                              'routes': routes,
                              'weights': weights,
                              'lh_budgets': lh_budgets,
                              'ic_budgets': inncercity_info}, budget)

        return budget / 1000000

    def less_than_three_first_level(self):
        """
        Вручную перебираем все возможные комбинации терминалов
        """
        terminals_variations = self.generate_binary_combinations(len(self.terminals))

        for terminals in terminals_variations:
            self.ga_second_level(list(terminals))

    def ga_first_level(self):
        """
        Функция, реализующая ГА первого уровня (отбор терминалов)
        :return: лучшее расположение терминалов и прилагающуюся информацию (
                маршрут,
                бюджет,
                средняя утилизация,
                статистика по обучению)
        """

        if len(self.terminals) == 0:
            return
        elif len(self.terminals) < 4:
            self.less_than_three_first_level()
            return

        gen_params = [(0, 1, 1) for _ in range(len(self.terminals))]

        ga_data = GaData(num_generations=self.num_generations_first_level, early_stop=self.early_stop_first_level)
        population = BinaryPopulation()
        population.set_params(num_individ=self.num_population_first_level,
                              gens=gen_params,
                              input_population=None)
        fitness_func = HyperbolaFitness(obj_function=self.ga_second_level,
                                        obj_value=0)
        selection = self.selection_first_level()
        crossover = self.crossover_first_level()
        mutation = BinStringMutation(self.mutation_first_level)
        new_generation = NewGeneration()

        population.fill()
        ga_data.population = population
        fitness_func.execute(ga_data=ga_data)
        ga_data.update()

        for i in range(ga_data.num_generations):
            selection.execute(ga_data)
            crossover.execute(ga_data)
            mutation.execute(ga_data)
            new_generation.execute(ga_data)
            fitness_func.execute(ga_data)
            ga_data.update()

            if ga_data.num_generation_no_improve >= ga_data.early_stop:
                break

        return ga_data


def combination_genetic_algorithm(
        dataset: pd.DataFrame,
        opt_function,
        opt_function_value: Union[float],
        dropdown_block: dict,
        num_generations: int,
        num_individuals: int,
        early_stop: int = 25,
):
    """
    Генетический алгоритм

    :code_assign: users
    :code_type: Оптимизация
    :imports: init_gui_dict, Window, Canvas, LinePlot, Noytech, LogisticsDataLoader, LogisticsBudgetModule, LogisticsResultCollector
    :packages:
    import numpy
    import os
    import shutil
    :param_block pd.DataFrame dataset: датасет
    :param str opt_function: целевая функция
    :param float opt_function_value: экстремум целевой функции
    :param dict dropdown_block: dropdown_block
    :param int num_generations: количество поколений 1 уровня
    :param int num_individuals: количество индивидов 1 уровня
    :param int early_stop: критерий остановки

    :returns: gui_dict, error

    :rtype: dict, str
    :semrtype: ,
    """
    error = ""

    dropdown_block_values = list(dropdown_block.values())
    path_input, path_output = dropdown_block_values[:2]

    if not path_input or not path_output:
        raise Exception(f'{path_input} {path_output}')
    if os.path.exists(path_output):
        shutil.rmtree(path_output)
    else:
        raise Exception(f'{path_output} doesn`t exist')

    os.mkdir(f'{path_output}')
    # update_progress(1)

    LDR = LogisticsDataLoader(sdm_path=f'{path_input}/МКР.xlsx',
                              config_path=f'{path_input}/rates.xlsx')

    order_list, terminals_list = LDR.get_orders(
        orders=dataset)

    LBM = LogisticsBudgetModule(mkr=LDR.mkr,
                                extra_point_price=LDR.extra_point_price,
                                innercity_rates=LDR.ic_rates,
                                lastmile_rates=LDR.lm_rates,
                                linehaul_rates=LDR.lh_rates,
                                preset_tariffs=LDR.preset_tariffs,
                                capacity=LDR.capacity)

    LRC = LogisticsResultCollector(top_five=None)

    N = Noytech(data_loader=LDR,
                budget_module=LBM,
                dispatch={},
                terminals=[],
                np_first_level=num_individuals,
                ng_first_level=num_generations,
                early_stop_first_level=early_stop,)

    numuration = 0
    summery_of_orders = len(order_list)

    for dispatch, terminals in zip(order_list, terminals_list):
        N.dispatch = dispatch
        if opt_function == 'terminals_definition':
            selection_second_level, crossover_second_level, mutation_second_level = 'tournament', 'order', 'inversion'
            N.num_generations_second_level, N.num_population_second_level, N.early_stop_second_level = dropdown_block_values[2:]
            N.second_level_params_set(selection_second_level, crossover_second_level, mutation_second_level)
            N.terminals = terminals
            N.ga_first_level()
        elif opt_function == 'lh_definition':
            path_to_terminals = dropdown_block_values[2]
            chosen_terminals = LDR.process_chosen_terminals(path_to_terminals)
            N.terminals = chosen_terminals
            N.ga_second_level(ind=[1.0 for _ in range(len(chosen_terminals))])
        LRC.top_five = N.top_five
        new_dir = f'{path_output}'
        os.mkdir(new_dir)
        LRC.get_results_into_exe(f'{new_dir}/dispatch.xlsx', new_dir, dispatch, LDR.max_capacity)

        numuration += 1
        # update_progress(numuration / summery_of_orders * 100)
        N.top_five = {}

    titles = ['Исходные данные', 'Зонирование пунктов назначения + FTL (full-truck load)',
              'Остатки груза после этапа FTL', 'Лайнхолл', 'Последняя миля', 'Внутригород по терминалам']
    gui_dict = LRC.gui_dict
    gui_dict['text'].append({'title': f'Лучший бюджет: {LRC.for_table[0]:.0f} руб.', 'value': LRC.for_table[1]})
    for table, title in zip(LRC.info_to_table, titles):
        gui_dict['table'].append(
            {
                'title': title,
                'value': table.to_dict('list')
            }
        )

    return gui_dict, error


if __name__ == '__main__':

    LDR = LogisticsDataLoader(
        sdm_path='/Users/danielageev/Work/Приоритет_2030/Noytech/NoytechGA/september_test/МКР.xlsx',
        config_path='/Users/danielageev/Work/Приоритет_2030/Noytech/NoytechGA/september_test/rates.xlsx')

    order_list, terminals_list, dates = LDR.get_orders(
        orders='/Users/danielageev/Work/Приоритет_2030/Noytech/NoytechGA/september_test/data.xlsx',
        directions=['Север', 'Восток', 'Юг'], weekdays=[])

    LBM = LogisticsBudgetModule(mkr=LDR.mkr,
                                extra_point_price=LDR.extra_point_price,
                                innercity_rates=LDR.ic_rates,
                                lastmile_rates=LDR.lm_rates,
                                linehaul_rates=LDR.lh_rates,
                                preset_tariffs=LDR.preset_tariffs,
                                capacity=LDR.capacity)

    LRC = LogisticsResultCollector(top_five=None)

    N = Noytech(data_loader=LDR,
                budget_module=LBM,
                dispatch={},
                terminals=[],
                np_first_level=100,
                ng_first_level=100,
                early_stop_first_level=25,
                np_second_level=100,
                ng_second_level=100,
                early_stop_second_level=25,
                )

    t = time.time()
    for dispatch, terminals, date in zip(order_list, terminals_list, dates):
        N.dispatch = dispatch
        N.terminals = terminals
        N.ga_first_level()
        LRC.top_five = N.top_five
        LRC.get_results_into_exe(
            f'/Users/danielageev/Work/Приоритет_2030/Noytech/NoytechGA/results_new_code/res_simple_tariffs_more_params{date}.xlsx',
            None, date, dispatch, LDR.max_capacity)
        N.top_five = {}
    print(f"Время выполнения: {(time.time() - t) / 60} минут")
