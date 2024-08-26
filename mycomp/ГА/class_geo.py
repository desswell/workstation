class GeoTracePlot(SimplePlot):
    """
    Карта для визуализации маршрута

    :code_assign: service
    :code_type: Пользовательские функции
    :imports: SimplePlot

    :packages:
    import plotly.graph_objects as go
    import pandas as pd
    import numpy as np
    import random

    Параметры:
    cities: путь до файла с координатами
    routes: путь маршрута
    terminals: терминалы (города заезда)
    from_city: название города откуда везти
    row: строка в subplots
    col: столбец в subplots
    """

    def __init__(
            self,
            cities,
            routes,
            terminals,
            from_city='Москва',

            row: Union[int, None] = None,
            col: Union[int, None] = None
    ):
        super().__init__(
            row=row,
            col=col)

        self.cities = cities
        self.routes = routes
        self.terminals = terminals
        self.from_city = from_city

    def get_coordinates(self, path, route: list, from_city) -> dict:
        """
        Функция, которая возвращает координаты городов, которые были использованы
        :param list route: список маршрутов
        :return: словарь координат
        """
        coordinates = pd.read_csv(path)
        routes = set()
        for unit in route:
            for cities in unit['cities']:
                routes.add(cities)
        cities = {"Москва": {"lat": 55.751244, "lon": 37.618423}}
        for index, row in coordinates.iterrows():
            city_name = row['Город']
            if city_name == from_city:
                lat = row['Широта']
                lon = row['Долгота']
                cities[city_name] = {'lat': lat, 'lon': lon}
            if city_name in routes:
                lat = row['Широта']
                lon = row['Долгота']
                cities[city_name] = {'lat': lat, 'lon': lon}
        return cities

    def draw(self, fig: go.Figure):
        scale_factor = 4
        terminals_list = [self.from_city] + list(self.terminals.keys())
        terminals_weight = [0] + list(self.terminals.values())
        # Загрузка геоданных регионов России из файла russia_regions.geojson
        for city, city_data, terminal, weight in zip(list(self.cities.keys()), list(self.cities.values()),
                                                     terminals_list, terminals_weight):
            lat = city_data["lat"]
            lon = city_data["lon"]
            if city != self.from_city:
                color = 'red'
                log_weight = np.log(float(f"{weight:.1f}") + 1)
                size = int(scale_factor * log_weight)
            else:
                color = 'blue'
                size = 10

            fig.add_trace(go.Scattermapbox(
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

        for route in self.routes:
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

            fig.add_trace(go.Scattermapbox(
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
        fig.update_traces(
            hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial"),
            # Белый фон и стиль шрифта для всплывающей подсказки
        )

        fig.update_layout(
            mapbox=dict(
                style="open-street-map",
                center=dict(lat=60, lon=100),
                zoom=2
            ),
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            showlegend=False,
            hovermode='closest',  # Изменение режима наведения
        )

        return fig

    @property
    def amount_plots(self):
        return 1
