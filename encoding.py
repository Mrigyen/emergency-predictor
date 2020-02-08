import numpy as np
from scipy.spatial import distance


# Source: https://codereview.stackexchange.com/questions/28207/finding-the-closest-point-to-a-list-of-points
def closest_node(node, nodes):
    closest_index = distance.cdist([node], nodes).argmin()
    return nodes[closest_index]


def location_danger(node, nodes) -> float:
    closest_n = closest_node(node, nodes)
    dist = distance.cdist([node], [closest_n])
    reciprocal = 1/dist[0][0]
    return reciprocal


minutes_in_day = 24*60
days_in_year = 365  # TODO: Modify to add support for new years


def sin_time(military_time_in_minutes: int) -> float:
    return np.sin(2 * np.pi * military_time_in_minutes/minutes_in_day)


def cos_time(military_time_in_minutes: int) -> float:
    return np.cos(2 * np.pi * military_time_in_minutes/minutes_in_day)


def sin_day(day_of_year: int) -> float:
    return np.sin(2 * np.pi * day_of_year/days_in_year)


def cos_day(day_of_year: int) -> float:
    return np.cos(2 * np.pi * day_of_year/days_in_year)


def military_time_in_minutes_fn(military_time: str) -> int:
    # TODO: add validation
    military_time = military_time.split(sep=":")
    return 60 * int(military_time[0]) + int(military_time[1])
