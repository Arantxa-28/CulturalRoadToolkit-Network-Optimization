#This code replicates the Ntework Reading data structure

import os
import csv
from typing import List, Dict
from collections import defaultdict
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math
import bisect
import copy
import cplex
import itertools
import optuna
from pathlib import Path



# Project folders
PROJECT_ROOT = Path(__file__).resolve().parent.parent
input_folder = PROJECT_ROOT / "data"

# Input files
node_file = input_folder / "Node.csv"
tripfile = input_folder / "Trip.csv"
lineStop_file = input_folder / "LineStop.csv"
lines_file = input_folder / "Lines.csv"
linetime_file = input_folder / "LineTime.csv"
SchDep_file = input_folder / "SchDep.csv"
NodeG_file = input_folder / "LinkCon.csv"
PathPara_file = input_folder / "PathPara.csv"
DesignPara_file = input_folder / "DesignPara.csv"

# Create node containers
nodes = []
nodes_dict = {}

# Create line containers
lines = []
lines_dict = {}


#%% INPUT DATA LOADER

def load_input_data(input_folder):

    files = {
        "nodes": input_folder / "Node.csv",
        "trips": input_folder / "Trip.csv",
        "line_stops": input_folder / "LineStop.csv",
        "lines": input_folder / "Lines.csv",
        "line_times": input_folder / "LineTime.csv",
        "scheduled_departures": input_folder / "SchDep.csv",
        "link_connections": input_folder / "LinkCon.csv",
        "path_parameters": input_folder / "PathPara.csv",
        "design_parameters": input_folder / "DesignPara.csv",
    }

    # Check that all files exist
    missing_files = [
        str(path)
        for path in files.values()
        if not path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Missing input files:\n" + "\n".join(missing_files)
        )

    data = {
        name: pd.read_csv(path)
        for name, path in files.items()
    }

    print("All input files have been read")

    return data


data = load_input_data(input_folder)

# Keep the current variable names temporarily
nodeCSV = data["nodes"]
tripCSV = data["trips"]
LineStopCSV = data["line_stops"]
LinesCSV = data["lines"]
LineTimeCSV = data["line_times"]
SchDepCSV = data["scheduled_departures"]
GNCostCSV = data["link_connections"]
PathParaCSV = data["path_parameters"]
DesignParaCSV = data["design_parameters"]


print("All input files have been read")

# Basic structures derived from input data
Linestotal = int(LineStopCSV.shape[0])
Stops_dict = {}


#%% INPUT VALIDATION

def validate_input_data(data):

    required_columns = {
        "nodes": {
            "id",
            "Name",
            "Type"
        },
        "trips": {
            "id",
            "Origin",
            "Dest",
            "TargetDepTime",
            "Demand",
            "DemandGroup2"
        },
        "line_stops": {
            "link",
            "A",
            "B"
        },
        "lines": {
            "id",
            "Name",
            "SerType",
            "HeadWay",
            "StartTime",
            "EndTime",
            "VehType",
            "Cap",
            "FullCap",
            "Doors"
        },
        "line_times": {
            "lineID",
            "Time"
        },
        "path_parameters": {
            "Cost",
            "Value"
        }
    }

    # Check required columns
    for dataset_name, columns in required_columns.items():

        df = data[dataset_name]

        missing = columns - set(df.columns)

        if missing:
            raise ValueError(
                f"{dataset_name}: missing columns {sorted(missing)}"
            )

    # ------------------------------------------------------------------
    # Node validation
    # ------------------------------------------------------------------

    node_ids = set(data["nodes"]["id"])

    invalid_origins = (
        set(data["trips"]["Origin"])
        - node_ids
    )

    invalid_destinations = (
        set(data["trips"]["Dest"])
        - node_ids
    )

    if invalid_origins:
        raise ValueError(
            f"Trip.csv contains invalid origins: "
            f"{sorted(invalid_origins)}"
        )

    if invalid_destinations:
        raise ValueError(
            f"Trip.csv contains invalid destinations: "
            f"{sorted(invalid_destinations)}"
        )

    # ------------------------------------------------------------------
    # Line validation
    # ------------------------------------------------------------------

    line_ids = set(data["lines"]["id"])
    line_time_ids = set(data["line_times"]["lineID"])
    line_stop_ids = set(data["line_stops"]["link"])

    missing_time = line_ids - line_time_ids

    if missing_time:
        raise ValueError(
            f"Missing travel times for lines: "
            f"{sorted(missing_time)}"
        )

    missing_links = line_ids - line_stop_ids

    if missing_links:
        raise ValueError(
            f"Missing LineStop entries for lines: "
            f"{sorted(missing_links)}"
        )

    # ------------------------------------------------------------------
    # Service type validation
    # ------------------------------------------------------------------

    valid_service_types = {
        "Schedule",
        "Frequency"
    }

    invalid_types = (
        set(data["lines"]["SerType"])
        - valid_service_types
    )

    if invalid_types:
        raise ValueError(
            f"Unknown service types: "
            f"{sorted(invalid_types)}"
        )

    # ------------------------------------------------------------------
    # Capacity validation
    # ------------------------------------------------------------------

    if (data["lines"]["Cap"] <= 0).any():
        raise ValueError(
            "All lines must have Cap > 0."
        )

    if (
        data["lines"]["FullCap"]
        <
        data["lines"]["Cap"]
    ).any():
        raise ValueError(
            "FullCap must be >= Cap for every line."
        )

    print("Input validation completed successfully.")


validate_input_data(data)


#%% PATH PARAMETERS

path_params = (
    PathParaCSV
    .set_index("Cost")["Value"]
    .astype(float)
    .to_dict()
)

# Define constants from PathPara

WalkW = path_params["WalkW"]
WaitW = path_params["WaitW"]
ConW = path_params["ConW"]
ZoneWaitW = path_params["ZoneWaitW"]
TransferP = path_params["TransferP"]
InVTrainW = path_params["InVTrainW"]
InVSTrainW = path_params["InVSTrainW"]
InVBusW = path_params["InVBusW"]
InVMetroW = path_params["InVMetroW"]
MinWaitTime = path_params["MinWaitTime"]
TransferTime = path_params["TransferTime"]
WaitHeadway = path_params["WaitHeadway"]
BoundNonDomEventLower = path_params["BoundNonDomEventLower"]
BoundNonDomEventUpper = path_params["BoundNonDomEventUpper"]


#%% LINE GROUPS

line_ids = LinesCSV["id"].astype(int).tolist()

schedule_line_ids = (
    LinesCSV.loc[
        LinesCSV["SerType"] == "Schedule",
        "id"
    ]
    .astype(int)
    .tolist()
)

frequency_line_ids = (
    LinesCSV.loc[
        LinesCSV["SerType"] == "Frequency",
        "id"
    ]
    .astype(int)
    .tolist()
)

num_lines = len(line_ids)

print(f"Number of lines: {num_lines}")
print(f"Schedule-based lines: {schedule_line_ids}")
print(f"Frequency-based lines: {frequency_line_ids}")



#%% Define Lines Perception Factors

#in-veh perception factors
perception_factors = {
    'Bus': InVBusW,
    'Train': InVTrainW,
    'STrain': InVSTrainW,
    'Metro': InVMetroW
}

lines_type = {}
for i in range(len(LineTimeCSV)):
    line_id = int(LineTimeCSV.loc[i, 'lineID'])
    time = float(LineTimeCSV.loc[i, 'Time'])
    veh_type = LinesCSV.loc[LinesCSV['id'] == line_id, 'VehType'].values[0]
    factor = perception_factors[veh_type]  # Match con perception_factors

    # Calcular GenCostTravel
    gen_cost_travel = time * factor

    # Agregar todo al diccionario lines_type
    lines_type[line_id] = {
        'Time': time,
        'VehType': veh_type,
        'FactorInVeh': factor,
        'GenCostInVeh': gen_cost_travel
    }

#%%
for row in range (0,Linestotal,1):
    lineid = int(LineStopCSV.iloc[row,0])
    stop_list = []
    for col in LineStopCSV.columns[1:]:
        stop = LineStopCSV.iloc[row][col]
        if pd.notna(stop):  # Ignorar valores NaN
            stop_list.append(int(stop))
            Stops_dict[f"StopLine{lineid}"] = stop_list       
            exec(f"Stops_list{lineid}= stop_list")

#%% CREATE NODES

class NodeClass:
    def __init__(self, node_id, name, node_type):
        self.ID = node_id
        self.Name = name
        self.Type = node_type


nodes = []
nodes_dict = {}

for _, row in nodeCSV.iterrows():

    node_id = int(row["id"])

    node = NodeClass(
        node_id=node_id,
        name=row["Name"],
        node_type=row["Type"]
    )

    nodes_dict[node_id] = node
    nodes.append(node)

print(f"{len(nodes)} nodes created")


#%% CREATE TRIPS

class TripClass:
    def __init__(self, trip_id, origin, destination, demand):
        self.ID = trip_id
        self.Origin = origin
        self.Destination = destination
        self.Demand = demand


trips = []
trips_dict = {}

for _, row in tripCSV.iterrows():

    trip_id = int(row["id"])

    trip = TripClass(
        trip_id=trip_id,
        origin=int(row["Origin"]),
        destination=int(row["Dest"]),
        demand=float(row["Demand"])
    )

    trips_dict[trip_id] = trip
    trips.append(trip)

print(f"{len(trips)} trips created")


#%% CREATE TRANSIT LINES

class TransitLineClass:
    def __init__(
        self,
        set_id,
        name,
        start_time,
        end_time,
        headway,
        service_type,
        vehicle_type
    ):
        self.ID = set_id
        self.Name = name
        self.StartTime = start_time
        self.EndTime = end_time
        self.Headway = headway
        self.ServiceType = service_type
        self.VehicleType = vehicle_type

        self.Stops = []
        self.SegTravelTimes = []
        self.TravelLength = 0.0
        self.NumOfSegs = 0
        self.NumOfTrains = 0
        self.TrainCap = []
        self.IniDepTime = []
        self.NumOfDoors = 0
        self.FreCap = None


lines = []
lines_dict = {}

for _, row in LinesCSV.iterrows():

    line_id = int(row["id"])

    line = TransitLineClass(
        set_id=line_id,
        name=str(row["Name"]),
        start_time=int(row["StartTime"]),
        end_time=int(row["EndTime"]),
        headway=float(row["HeadWay"]),
        service_type=str(row["SerType"]),
        vehicle_type=str(row["VehType"])
    )

    lines_dict[line_id] = line
    lines.append(line)

print(f"{len(lines)} transit lines created")

#%% GENERATE INITIAL TIMETABLES

end_simulation = 120

def generate_initial_timetables(lines, end_simulation):

    timetables = {}
    wait_tables = {}

    for line in lines:

        line_id = int(line.ID)
        headway = int(line.Headway)
        start_time = int(line.StartTime)

        if headway <= 0:
            raise ValueError(
                f"Line {line_id} has invalid headway: {headway}"
            )

        timetable = []
        wait_table = []

        current_time = start_time

        while current_time < end_simulation:

            timetable.append(current_time)
            wait_table.append(headway / 2)

            current_time += headway

        timetables[line_id] = timetable
        wait_tables[line_id] = wait_table

    return timetables, wait_tables


timetables, wait_tables = generate_initial_timetables(
    lines,
    end_simulation=end_simulation
)

print("Timetables created")

for line_id in line_ids:
    print(
        f"Line {line_id}: {timetables[line_id]}"
    )

# Temporary compatibility with the current code
timetables_list = [
    timetables[line_id]
    for line_id in line_ids
]

wait_tables_list = [
    wait_tables[line_id]
    for line_id in line_ids
]


#%% BUILD NETWORK LINKS

# Merge link topology with travel times
links_df = LineStopCSV.merge(
    LineTimeCSV,
    left_on="link",
    right_on="lineID",
    how="left",
    validate="one_to_one"
)

# Check that every link has a travel time
if links_df["Time"].isna().any():
    missing_links = links_df.loc[
        links_df["Time"].isna(),
        "link"
    ].tolist()

    raise ValueError(
        f"Missing travel time for links: {missing_links}"
    )

# Create link dictionary
link_dict = {}

for _, row in links_df.iterrows():

    link_id = int(row["link"])

    link_dict[link_id] = {
        "Link": link_id,
        "LineID": link_id,
        "Time": int(row["Time"]),
        "A": int(row["A"]),
        "B": int(row["B"])
    }

num_links = len(link_dict)

print(f"{num_links} network links created")


#%% BUILD NETWORK GRAPH

grafo = nx.MultiDiGraph()

for link_id, link_data in link_dict.items():

    grafo.add_edge(
        link_data["A"],
        link_data["B"],
        key=link_id,
        line_id=link_id,
        cost=link_data["Time"]
    )

print(
    f"Network graph created: "
    f"{grafo.number_of_nodes()} nodes, "
    f"{grafo.number_of_edges()} links"
)


#%% DETECT OD PAIR

od_pairs = (
    tripCSV[["Origin", "Dest"]]
    .drop_duplicates()
    .reset_index(drop=True)
)

if len(od_pairs) != 1:
    raise ValueError(
        "Current route-generation version expects one OD pair. "
        f"Found {len(od_pairs)} OD pairs."
    )

source_node = int(od_pairs.loc[0, "Origin"])
target_node = int(od_pairs.loc[0, "Dest"])

print(
    f"Route generation OD: "
    f"{source_node} -> {target_node}"
)


#%% GENERATE ROUTES AUTOMATICALLY

def generate_routes(graph, source, target):

    # Simple graph to obtain node sequences
    simple_graph = nx.DiGraph()

    for u, v in graph.edges():
        simple_graph.add_edge(u, v)

    node_paths = list(
        nx.all_simple_paths(
            simple_graph,
            source=source,
            target=target
        )
    )

    routes = []

    for node_path in node_paths:

        edge_options = []

        for u, v in zip(
            node_path[:-1],
            node_path[1:]
        ):

            edges = graph.get_edge_data(u, v)

            if edges is None:
                raise ValueError(
                    f"No links found between nodes {u} and {v}"
                )

            available_links = sorted(
                int(link_id)
                for link_id in edges.keys()
            )

            edge_options.append(available_links)

        for combination in itertools.product(*edge_options):
            routes.append(list(combination))

    return routes


g_paths_merge = generate_routes(
    grafo,
    source=source_node,
    target=target_node
)


route_summary_lines = {
    f"route_{i + 1}": route
    for i, route in enumerate(g_paths_merge)
}


print("\nRoutes generated automatically:")

for route_name, route_lines in route_summary_lines.items():
    print(
        f"{route_name}: {route_lines}"
    )

print(
    f"Total routes generated: "
    f"{len(route_summary_lines)}"
)

#%% G PATHS MERGE

#generar las rutas por hora

#definie cicle to generate list as g_paths_merge
count_elements = len(g_paths_merge)
g_paths_merge_time=[]
for e in range(count_elements):
    lista_temp_a = g_paths_merge[e]
    num_links_list_a = len(lista_temp_a)
    lista_temp_b =[]
    for f in range(num_links_list_a):
        id_line_n = lista_temp_a[f]
        time_value = next(item['Time'] for item in link_dict.values() if item['LineID'] == id_line_n)
        lista_temp_b.append(time_value)
    g_paths_merge_time.append(lista_temp_b)

#%% create a disctionary of routes contained in g_paths_merge conatining in-veh-travel time and gne-in-veh-travel time

in_veh_perc = {}

for idx, path in enumerate(g_paths_merge):
    route_key = f"route_{idx + 1}"
    gen_costs = [lines_type[line]['GenCostInVeh'] for line in path]
    total_time = sum(lines_type[line]['Time'] for line in path)
    in_veh_perc[route_key] = {'GenCostInVeh': gen_costs, 'TotalTime': total_time}
    
    
#%% ROUTES DICT    
routes_dicts = {}  # Este diccionario almacenará todos los diccionarios de rutas

for i, (route, travel_times) in enumerate(zip(g_paths_merge, g_paths_merge_time), start=1):
    route_dict = {}  # Este diccionario almacenará las lineas de esta ruta
    for line_id, travel_time in zip(route, travel_times):
        timetable = timetables_list[line_id]  # Obtenemos el timetable correspondiente
        route_dict[line_id] = {
            'lineid': line_id,
            'timetable': timetable,
            'traveltime': travel_time
        }
    # Guardar el diccionario de la ruta con el nombre correspondiente
    routes_dicts[f"route_{i}"] = route_dict

#Add the In-veh cost
for route_key, route_dict in routes_dicts.items():  # Iterar sobre cada ruta en routes_dicts
    for line_id, line_info in route_dict.items():  # Iterar sobre cada línea en la ruta
        # Obtener el GenCostTravel de lines_type usando line_id
        gen_cost_travel = lines_type.get(line_id, {}).get('GenCostInVeh', 0)  # Default 0 si no se encuentra
        
        # Agregar GenCostTravel al diccionario de la línea
        route_dict[line_id]['GenCostInVeh'] = gen_cost_travel

    # Save dictionary using routes_dicts
    routes_dicts[route_key] = route_dict

#%% Next Step
#function to compute 
def calculate_route_times(routes_dict, route_sequence):
    result = {}

    # We iterate over each departure time in the first line of the sequence.
    for start_time in routes_dict[route_sequence[0]]['timetable']:
        current_time = start_time       # Initialize with the departure time from the first line.
        route_times = []                #  Store the route segments here.
        route_possible = True           # Flag to determine whether the route is feasible.

        # Iterate over each line in the given order.
        for line_id in route_sequence:
            timetable = routes_dict[line_id]['timetable']
            travel_time = routes_dict[line_id]['traveltime']
            
            # Check if the current time is later than the line's last departure time.
            # It is not possible to catch any transport on this line.
            if current_time > timetable[-1]:
                route_possible = False
                break
            
            # Find the index of the first departure time that is >= current_time.
            idx = bisect.bisect_left(timetable, current_time)
            # idx will always be less than len(timetable) due to the previous check.
            next_start_time = timetable[idx]
            arrival_time = next_start_time + travel_time + int(TransferTime)

            # Save
            route_times.append({
                'line': line_id,
                'departure_time': next_start_time,
                'arrival_time': arrival_time
            })

            # Update current_time for the next line.
            current_time = arrival_time

        # If the route is feasible for this departure time, add it to the result.
        if route_possible:
            result[start_time] = route_times

    return result


#%% PATHS PER HOUR INITIATE
# Generate paths_per_hour by calling the function to calculate the route times.
paths_per_hour = {}
numb_choices = len(g_paths_merge)

for i in range(numb_choices):
    route_key = f"route_{i+1}"
    dict_of_route = routes_dicts[route_key]
    sequ_of_route = g_paths_merge[i]
    route_times_result = calculate_route_times(dict_of_route, sequ_of_route)
    paths_per_hour[route_key] = route_times_result

# Eliminate elements form paths_per_hour that end after 120!
    
for route, departures in paths_per_hour.items():
    keys_to_delete = []
    for key, steps in departures.items():
        if any(step['arrival_time'] > 120 for step in steps):
            keys_to_delete.append(key)
    for key in keys_to_delete:
        del paths_per_hour[route][key]

print(" ")
print("paths_per_hour created: No arrival time after ", end_simulation)
print(" ")
print(paths_per_hour)

#%% Working on available routes by time interval

durations = {}

for route, times in paths_per_hour.items():
    durations[route] = {}
    for start_time, lines in times.items():
        departure_time_initial = lines[0]['departure_time']
        arrival_time_final = lines[-1]['arrival_time']
        total_duration = arrival_time_final - departure_time_initial

        durations[route][start_time] = {
            "start_time": start_time,  # Store the start_time.
            "total_duration": total_duration
        }

# Definimos los intervalos de tiempo (cada 3 unidades hasta 60)
intervals = list(range(0, 60, 3))

# Inicializamos interval_data con 0 rutas disponibles y demanda en 0
interval_data = {i: {'available_routes': 0, 'demand': 0} for i in intervals}

# Contamos cuántas rutas están disponibles en cada intervalo
for route, times in durations.items():
    for start_time, duration_info in times.items():
        # Encontramos el intervalo más cercano a start_time
        closest_interval = max([i for i in intervals if i <= start_time], default=None)
        
        if closest_interval is not None:
            interval_data[closest_interval]['available_routes'] += 1


# Demand by interval (Group 1 and Group 2)---------------------------------------------------------------------------

demand_dict = dict(
    zip(
        tripCSV["TargetDepTime"],
        tripCSV["Demand"]
    )
)

demand_dict_2 = dict(
    zip(
        tripCSV["TargetDepTime"],
        tripCSV["DemandGroup2"]
    )
)

# PASSENGER DEMAND CHECK

total_demand_g1 = sum(demand_dict.values())
total_demand_g2 = sum(demand_dict_2.values())

total_demand = total_demand_g1 + total_demand_g2

if total_demand_g1 <= 0:
    raise ValueError(
        "Passenger Group 1 must have positive demand."
    )

group_2_active = total_demand_g2 > 0

print(f"Total demand Group 1: {total_demand_g1:.2f}")
print(f"Total demand Group 2: {total_demand_g2:.2f}")
print(f"Total passenger demand: {total_demand:.2f}")

if total_demand > 0:
    print(
        f"Group 1 share: "
        f"{total_demand_g1 / total_demand:.2%}"
    )

    print(
        f"Group 2 share: "
        f"{total_demand_g2 / total_demand:.2%}"
    )

if group_2_active:
    print("Two passenger groups active.")
else:
    print(
        "Group 2 demand is zero. "
        "Running single-group scenario."
    )

#%% COMPUTE DEMAND BY AVAILABLE DEPARTURE INTERVAL

unique_start_times = sorted({
    time
    for route in durations.values()
    for time in route.keys()
})


def assign_demand_to_intervals(
    demand_input,
    unique_start_times
):

    if not unique_start_times:
        raise ValueError(
            "No available route departure times were generated."
        )

    accumulated_demand = {
        time: 0.0
        for time in unique_start_times
    }

    for time, demand in demand_input.items():

        previous_departures = [
            t
            for t in unique_start_times
            if t <= time
        ]

        if previous_departures:
            closest_time = max(previous_departures)
        else:
            closest_time = unique_start_times[0]

        accumulated_demand[closest_time] += float(demand)

    return accumulated_demand


demand_by_interval = assign_demand_to_intervals(
    demand_dict,
    unique_start_times
)

demand_by_interval_2 = assign_demand_to_intervals(
    demand_dict_2,
    unique_start_times
)

print(
    "Demand check Group 1:",
    sum(demand_dict.values()),
    "->",
    sum(demand_by_interval.values())
)

print(
    "Demand check Group 2:",
    sum(demand_dict_2.values()),
    "->",
    sum(demand_by_interval_2.values())
)

#%% COMBINED_DICT
start_route_list = unique_start_times
combined_dict = {
    key: {
        "durations": [data[key]['total_duration'] for route, data in durations.items() if key in data],
        "routes": [route for route, data in durations.items() if key in data]
    }
    for key in start_route_list
}

#COMPUTE CUMULATED WAITING TIME
demand_data_frame = pd.DataFrame(list(demand_by_interval.items()), columns=["Time", "Demand_cumulated"])

# Creamos una columna para los pasajeros liberados en cada intervalo
demand_data_frame["Demand_release"] = demand_data_frame["Time"].apply(lambda x: demand_dict.get(x, 0))
demand_data_frame["Waiting_time"] = 0

waiting_passengers = 0

last_demand_time = None
for index, row in demand_data_frame.iterrows():
    if row["Demand_cumulated"] > 0:
        # Si la demanda acumulada es mayor a 0, guardamos el tiempo
        last_demand_time = row["Time"]
        demand_data_frame.at[index, "Waiting_time"] = 0  # El tiempo de espera es 0 mientras haya demanda acumulada
    else:
        # Si no hay demanda acumulada, calculamos el tiempo de espera
        if last_demand_time is None:
            # Si nunca hubo una demanda acumulada, tomamos el tiempo actual como waiting time
            demand_data_frame.at[index, "Waiting_time"] = row["Time"]
        else:
            # Si hay demanda acumulada en el pasado, calculamos la diferencia
            demand_data_frame.at[index, "Waiting_time"] = row["Time"] - last_demand_time

demand_data_frame["Partial_waiting_time"] = demand_data_frame["Demand_release"] * demand_data_frame["Waiting_time"] * ZoneWaitW
partial_waiting_time = float(demand_data_frame["Partial_waiting_time"].sum())

print(" ")
print ("Partial Waiting Time Cumulated:",partial_waiting_time)
print(" ")

#%% PATH COST (real time of route) - to be used for compute probailities initially

#Before compute path cost

## No. 1 First Stop Wiaiting Time
# computed before in demand_data_frame

## No. 2 Number of transfers using g_paths_merge
route_transfer_penalty = {}

for i, ruta in enumerate(g_paths_merge, start=1):
    transfers = len(ruta) - 1
    penalty = transfers * TransferP
    route_transfer_penalty[f'route_{i}'] = penalty

print(" ")
print("Tranfer Penalty by route: ",route_transfer_penalty)
print(" ")

## No. 3 In-veh percived time
route_inveh_perc = {}

for i, ruta in enumerate(g_paths_merge, start=1):
    costo_total = sum(lines_type[line_id]['GenCostInVeh'] for line_id in ruta)
    route_inveh_perc[f'route_{i}'] = costo_total
 
print(" ")
print("In-veh (perception) by route: ", route_inveh_perc)
print(" ")

## No. 4 Waiting time in between stops

#Compute Path cost for probailities
path_cost={}
n_paths_merge=len(g_paths_merge)
for route in route_inveh_perc:
    path_cost[route] = route_inveh_perc[route] + route_transfer_penalty.get(route, 0)


travel_in_veh = {}

# Iterar sobre cada ruta en routes_dict
for route, lines in routes_dicts.items():
    # Sumar todos los valores de 'traveltime' de las líneas de cada ruta
    total_travel_time = sum(data['traveltime'] for data in lines.values())
    travel_in_veh[route] = total_travel_time
    
for key, value in combined_dict.items():
    in_veh_time_list = [sum(routes_dicts[route][line]['traveltime'] for line in routes_dicts[route]) for route in value['routes']]
    combined_dict[key]['in_veh_time'] = in_veh_time_list  # Agregar la lista

#   
for key, value in combined_dict.items():
    # Restar elemento a elemento durations - in_veh_time
    waiting_time_list = [d - i for d, i in zip(value['durations'], value['in_veh_time'])]
    combined_dict[key]['waiting_time'] = waiting_time_list  # Agregar la lista

#Modify combined_dict adding gen_travel_cost to combined dict
for time_slot, data in combined_dict.items():
    gen_travel_cost = []
    for i, route in enumerate(data['routes']):
        in_veh = route_inveh_perc.get(route, 0)
        transfer_penalty = route_transfer_penalty.get(route, 0)
        waiting = data['waiting_time'][i]
        total_cost = in_veh + transfer_penalty + (waiting * WaitW)
        gen_travel_cost.append(round(total_cost, 2))  # redondear si lo deseas
    data['gen_travel_cost'] = gen_travel_cost

for time_slot, data in combined_dict.items():
    gen_travel_cost_2 = []
    for i, route in enumerate(data['routes']):
        in_veh = route_inveh_perc.get(route, 0)
        transfer_penalty = route_transfer_penalty.get(route, 0)
        waiting = data['waiting_time'][i]
        total_cost_2 = in_veh + transfer_penalty + (waiting * WaitW) ## for demand group 2
        gen_travel_cost_2.append(round(total_cost_2, 2))  # redondear si lo deseas
    data['gen_travel_cost_2'] = gen_travel_cost_2
    
# No. 5 Crowding Cost Iteration no. 0
crowding_in_veh = {}

for key, value in combined_dict.items():
    in_veh_times = value.get('in_veh_time', [])
    adjusted_values = [0.50 * ConW for t in in_veh_times]
    crowding_in_veh[key] = adjusted_values

#Update combined_dict including crowding_in_veh
for key, value in combined_dict.items():
    in_veh_times = value.get('in_veh_time', [])
    crowding_values = [0.50 * ConW for t in in_veh_times]
    combined_dict[key]['crowding_in_veh'] = crowding_values

print(" ")
print("Crowding_in_veh: ")
print(crowding_in_veh)
print(" ")

#%% MODIFY COMBINED DICT

#Compute probailities of available routes USING DURATION OF ROUTE

# for demand
probabilities = {}

for key, value in combined_dict.items():
    gen_travel_costs = np.array(value['gen_travel_cost'])
    crowding_costs = np.array(value['crowding_in_veh'])

    # Nuevo costo combinado
    total_costs = gen_travel_costs + crowding_costs

    # Calcular exponencial negativa de los costos
    exp_neg_costs = np.exp(-total_costs)

    # Normalizar para obtener probabilidades (modelo logit)
    total_exp = np.sum(exp_neg_costs)
    if total_exp == 0:
        total_exp = 1e-10  # Evitar división por cero

    probs = exp_neg_costs / total_exp

    # Asociar cada ruta con su probabilidad
    routes = value['routes']
    probabilities[key] = {route: float(probs[i]) for i, route in enumerate(routes)}
 
# for demand_2
probabilities_2 = {}

for key, value in combined_dict.items():
    gen_travel_costs_2 = np.array(value['gen_travel_cost_2'])
    #crowding_costs = np.array(value['crowding_in_veh'])

    # Nuevo costo combinado
    total_costs = gen_travel_costs_2

    # Calcular exponencial negativa de los costos
    exp_neg_costs = np.exp(-total_costs)

    # Normalizar para obtener probabilidades (modelo logit)
    total_exp = np.sum(exp_neg_costs)
    if total_exp == 0:
        total_exp = 1e-10  # Evitar división por cero

    probs = exp_neg_costs / total_exp

    # Asociar cada ruta con su probabilidad
    routes = value['routes']
    probabilities_2[key] = {route: float(probs[i]) for i, route in enumerate(routes)}

print(" ")
print("Probabilities: ")
print(probabilities)
print(" ")
print("Probabilities 2: ")
print(probabilities_2)
print(" ")
    
#combinar combided_dict y prbabilities in dictionario new
passenger_route = {}
passenger_route_2 = {}

for time, data in combined_dict.items():
    routes = data["routes"]
    durations = data["durations"]
    route_probs = [probabilities[time].get(route, 0) for route in routes]
    
    # Guardar la información combinada
    passenger_route[time] = {
        "durations": durations,
        "routes": routes,
        "probabilities": route_probs
    }

for time in passenger_route:
    passenger_route[time]["demand"] = demand_by_interval.get(time, 0)

#for passenger_route_2
for time, data in combined_dict.items():
    routes = data["routes"]
    durations = data["durations"]
    route_probs = [probabilities_2[time].get(route, 0) for route in routes]
    
    # Guardar la información combinada
    passenger_route_2[time] = {
        "durations": durations,
        "routes": routes,
        "probabilities": route_probs
    }

for time in passenger_route_2:
    passenger_route_2[time]["demand"] = demand_by_interval_2.get(time, 0)

#%% COMPUTE PASSENGERS BY ROUTE DURING SIMULATION
## demand_probability_product

probability_demand_product = {}
probability_demand_product_2 = {}

# Group 1
for key, demand in demand_by_interval.items():
    if key in probabilities:
        prob_demand_list = [
            demand * prob
            for route, prob in probabilities[key].items()
        ]
        probability_demand_product[key] = prob_demand_list

# Group 2
for key, demand in demand_by_interval_2.items():
    if key in probabilities_2:
        prob_demand_list_2 = [
            demand * prob
            for route, prob in probabilities_2[key].items()
        ]
        probability_demand_product_2[key] = prob_demand_list_2
        
# for demand
for key, value in combined_dict.items():
    # Verificar si existe una lista de productos de demanda * probabilidad para este intervalo en probability_demand_product
    if key in probability_demand_product:
        # Agregar la lista de productos a combined_dict bajo la clave 'demand_prob_product'
        value['demand_prob_product'] = probability_demand_product[key]

# for demand_2
for key, value in combined_dict.items():
    # Verificar si existe una lista de productos de demanda * probabilidad para este intervalo en probability_demand_product
    if key in probability_demand_product_2:
        # Agregar la lista de productos a combined_dict bajo la clave 'demand_prob_product'
        value['demand_prob_product_2'] = probability_demand_product_2[key]

#MODIFY COMBINED_DICT
for key, value in combined_dict.items():
    if key in probability_demand_product:
        value['demand_prob_product'] = probability_demand_product[key]

for key, value in combined_dict.items():
    if key in probability_demand_product_2:
        value['demand_prob_product_2'] = probability_demand_product_2[key]


#%%
## objective function parte passengres
total_travelcost_routes = 0.0  # <- variable acumuladora

# Recorremos cada intervalo de tiempo en combined_dict
for key, value in combined_dict.items():
    # Obtener las listas de 'demand_prob_product' y 'gen_travel_cost' para este intervalo
    demand_prob_product = value.get('demand_prob_product', [])
    gen_travel_cost = value.get('gen_travel_cost', [])
    
    # Si ambas listas existen y tienen el mismo tamaño
    if len(demand_prob_product) == len(gen_travel_cost):
        # Calcular la suma de los productos para la función objetivo en este intervalo
        partial_sum = sum(d * g for d, g in zip(demand_prob_product, gen_travel_cost))
        
        # Sumar el resultado al total global
        total_travelcost_routes += partial_sum

print(" ")
print ("General travel time by route (demand x route Times):",total_travelcost_routes)
print(" ")

#%% compute waiting time - BODEFORE BOARDING
#COMPUTE CUMULATED WAITING TIME
demand_data_frame = pd.DataFrame(list(demand_by_interval.items()), columns=["Time", "Demand_cumulated"])

# Creamos una columna para los pasajeros liberados en cada intervalo
demand_data_frame["Demand_release"] = demand_data_frame["Time"].apply(lambda x: demand_dict.get(x, 0))
demand_data_frame["Waiting_time"] = 0

waiting_passengers = 0

last_demand_time = None
for index, row in demand_data_frame.iterrows():
    if row["Demand_cumulated"] > 0:
        # Si la demanda acumulada es mayor a 0, guardamos el tiempo
        last_demand_time = row["Time"]
        demand_data_frame.at[index, "Waiting_time"] = 0  # El tiempo de espera es 0 mientras haya demanda acumulada
    else:
        # Si no hay demanda acumulada, calculamos el tiempo de espera
        if last_demand_time is None:
            # Si nunca hubo una demanda acumulada, tomamos el tiempo actual como waiting time
            demand_data_frame.at[index, "Waiting_time"] = row["Time"]
        else:
            # Si hay demanda acumulada en el pasado, calculamos la diferencia
            demand_data_frame.at[index, "Waiting_time"] = row["Time"] - last_demand_time

demand_data_frame["Partial_waiting_time"] = demand_data_frame["Demand_release"] * demand_data_frame["Waiting_time"] * ZoneWaitW
partial_waiting_time = float(demand_data_frame["Partial_waiting_time"].sum())

print(" ")
print ("Partial Waiting Time Cumulated:",partial_waiting_time)
print(" ")

#%% fucntions to accumulate demand

def compute_accumulated_demand(demand_dict, unique_start_times):
    unique_start_times = sorted(unique_start_times)
    accumulated_demand = {}

    previous_time = 0
    for current_time in unique_start_times:
        total = 0
        # Acumular solo entre previous_time < t <= current_time
        for t in range(previous_time + 1, current_time + 1):
            total += demand_dict.get(t, 0)
        accumulated_demand[current_time] = total
        previous_time = current_time

    return accumulated_demand


# %%
#Compute CROWDING ON BOARD THE LINES - DATAFRAME PANDAS
 ####------------------------HERE 121!
time_intervals = list(range(121))
lines_data = []

for _, row in LinesCSV.iterrows():
    line_id = row['id']
    line_name = row['Name']
    line_data = {'LineID': line_id, 'Name': line_name}
    
    # Añadir columnas de tiempo (0-60) con ceros
    for time in time_intervals:
        line_data[time] = 0.0
    
    lines_data.append(line_data)

# Crear un nuevo DataFrame con la información actualizada
crowding_simulation = pd.DataFrame(lines_data)
crowding_simulation_empty = pd.DataFrame(lines_data)

print(" ")
print("Crowding Simulation created ")
print(" ")

#%% Update Crowding simulation using combined_dict, paths_per_hour, and the dataframestructure crowding_simulation

seating_cap = {}

for _, row in LinesCSV.iterrows():
    line_id = row['id']
    seating_cap[line_id] = {
        'Name': row['Name'],
        'Cap': row['Cap'],
        'FullCap': row['FullCap']
    }

def distribuir_demanda(crowding_simulation, combined_dict, paths_per_hour):
    for time, route_data in combined_dict.items():
        for route, demand_prob_1, demand_prob_2 in zip(
            route_data['routes'],
            route_data['demand_prob_product'],
            route_data['demand_prob_product_2']
        ):
            total_demand = demand_prob_1 + demand_prob_2

            if route in paths_per_hour and time in paths_per_hour[route]:
                path_info = paths_per_hour[route][time]
                
                for trip in path_info:
                    line_index = trip['line']
                    departure_time = trip['departure_time']
                    arrival_time = trip['arrival_time']

                    for t in range(departure_time, arrival_time):
                        if 0 <= t <= 120:
                            crowding_simulation[t] = crowding_simulation[t].astype(float)
                            crowding_simulation.loc[
                                crowding_simulation['LineID'] == line_index, t
                            ] += float(total_demand)
    return crowding_simulation

# Llamamos a la función para distribuir la demanda
crowding_lines = distribuir_demanda(crowding_simulation, combined_dict, paths_per_hour)


# %% RE-COMPUTE PASSENGERS COST CONSIDERING CORWDING-SEATING

seating_cap = {}

for _, row in LinesCSV.iterrows():
    line_id = row['id']
    seating_cap[line_id] = {
        'Name': row['Name'],
        'Cap': row['Cap'],
        'FullCap': row['FullCap']
    }

crowding_numeric = crowding_lines.copy()
crowding_numeric.iloc[:, 2:] = crowding_numeric.iloc[:, 2:].apply(pd.to_numeric, errors='coerce')

# PLATFORM_BALANCE

starting_lines = list(set(route[0] for route in route_summary_lines.values()))



# %%    

# Creamos el nuevo DataFrame de diferencia
non_seating_diff_df = crowding_numeric.copy()

# Restamos capacidad a cada fila (solo columnas horarias)
for idx, row in non_seating_diff_df.iterrows():
    line_id = row['LineID']  # Accedemos al LineID directamente
    cap = seating_cap[line_id]['Cap']  # Usamos line_id para acceder a la capacidad
    non_seating_diff_df.iloc[idx, 2:] = crowding_numeric.iloc[idx, 2:] - cap

# Opcional: reemplazar negativos por 0 (si solo interesan los pasajeros de pie)
non_seating_diff_df.iloc[:, 2:] = non_seating_diff_df.iloc[:, 2:].clip(lower=0)

# Update crowding_in_veh using values of crowding_proportion

#this is to compute standing passengers
crowding_seating = crowding_numeric.copy()

for idx in range(2, crowding_numeric.shape[1]):  # Empezamos desde la columna 2 (horarios)
    crowding_seating.iloc[:, idx] = crowding_numeric.iloc[:, idx] - non_seating_diff_df.iloc[:, idx]

#%% FUNCION DE COMPUTE non_seating_diff_df

def compute_non_seating_diff_df(seating_cap, crowding_numeric):
    # Asegurar que los datos numéricos estén bien
    crowding_numeric = crowding_numeric.copy()
    crowding_numeric.iloc[:, 2:] = crowding_numeric.iloc[:, 2:].apply(pd.to_numeric, errors='coerce')

    # Creamos el nuevo DataFrame de diferencia
    non_seating_diff_df = crowding_numeric.copy()

    # Restamos la capacidad de asientos a cada fila (solo columnas horarias)
    for idx, row in non_seating_diff_df.iterrows():
        line_id = row['LineID']
        cap = seating_cap.get(line_id, {}).get('Cap', 0)  # Evitar errores si falta LineID
        non_seating_diff_df.iloc[idx, 2:] = crowding_numeric.iloc[idx, 2:] - cap

    # Reemplazar valores negativos por 0 (solo pasajeros de pie)
    non_seating_diff_df.iloc[:, 2:] = non_seating_diff_df.iloc[:, 2:].clip(lower=0)

    return non_seating_diff_df


#%% CROWDING PROPORTION

# Update paths_per_hour_it 
# for cosndeirng the crowding_proportion based on dataframes non_seating_diff_df / crowding_numeric

# Asegura que los datos sean float64 para evitar problemas de tipo
crowding_numeric.iloc[:, 2:] = crowding_numeric.iloc[:, 2:].astype('float64')
non_seating_diff_df.iloc[:, 2:] = non_seating_diff_df.iloc[:, 2:].astype('float64')

# Crea una copia base del DataFrame
crowding_proportion = crowding_numeric.copy()

# Recorre fila por fila y columna por columna desde la tercera
for row in crowding_proportion.index:
    for col in crowding_proportion.columns[2:]:
        numerator = non_seating_diff_df.at[row, col]
        denominator = crowding_numeric.at[row, col]
        
        if denominator == 0:
            crowding_proportion.at[row, col] = 0.0
        else:
            crowding_proportion.at[row, col] = numerator / denominator


def compute_crowding_proportion(crowding_numeric, non_seating_diff_df, crowding_simulation_empty):
    # Copia para no modificar los originales
    crowding_proportion = crowding_simulation_empty.copy()

    # Asegurar que los datos sean numéricos
    crowding_numeric.iloc[:, 2:] = crowding_numeric.iloc[:, 2:].apply(pd.to_numeric, errors='coerce')
    non_seating_diff_df.iloc[:, 2:] = non_seating_diff_df.iloc[:, 2:].apply(pd.to_numeric, errors='coerce')

    # Recorre fila por fila y columna por columna desde la tercera
    for row in crowding_proportion.index:
        for col in crowding_proportion.columns[2:]:
            numerator = non_seating_diff_df.at[row, col]
            denominator = crowding_numeric.at[row, col]

            # Si el denominador es 0, asignamos 0.0
            if denominator == 0:
                crowding_proportion.at[row, col] = 0.0
            else:
                # Realizamos la división
                crowding_proportion.at[row, col] = numerator / denominator

    return crowding_proportion

# # At this point we have: --------------------------------------------------------------------
# # Total passengers on board  saved on      crowding_numeric
# # Paseengers seating         saved on      crowding_seating
# # Passengers standing        saved on      non_seating_diff_df
# # Porportion of passengers standing on     crowding_proportion
# # At this point we have: --------------------------------------------------------------------

# #%% Add crowding_value for each line in paths_per_hour_it

## Compute a new attribute to add to paths_per_hour_it
paths_per_hour_it = copy.deepcopy(paths_per_hour)

for route, times in paths_per_hour_it.items():
    for departure_time, trips in times.items():
        # Recorremos cada uno de los viajes (trips) en esa ruta y hora de salida
        for trip in trips:
            line_id = trip['line']  # Obtenemos el ID de la línea
            # Accedemos al valor correspondiente de crowding_proportion usando line_id y departure_time
            crowding_value = crowding_proportion.loc[crowding_proportion['LineID'] == line_id, departure_time].values
            if crowding_value.size > 0:
                # Asignamos el valor de crowding_value al diccionario en paths_per_hour
                trip['crowding_value'] = float(crowding_value[0])
            else:
                # Si no se encuentra un valor, asignamos 0.0
                trip['crowding_value'] = float(0.0)

def update_paths_with_crowding(paths_per_hour, crowding_proportion):

    paths_per_hour_it = copy.deepcopy(paths_per_hour)

    for route, times in paths_per_hour_it.items():
        for departure_time, trips in times.items():
            for trip in trips:
                line_id = trip['line']
                crowding_value = crowding_proportion.loc[crowding_proportion['LineID'] == line_id, departure_time].values
                trip['crowding_value'] = float(crowding_value[0]) if crowding_value.size > 0 else 0.0

    return paths_per_hour_it

print(" ")        
print("paths_per_hour_it created")
print(" ")

#%% COMPUTE A COMBINED_DICT_new NEXT ITERATION

# Compute crowding_in_veh with real crowding values

combined_dict_new = {}

#COMPUTE THE NEW crowding_in_veh

## crowding_in_veh
for key, value in combined_dict.items():
    new_value = value.copy()
    if 'crowding_in_veh' in new_value:
        new_value['crowding_in_veh'] = [0] * len(new_value['crowding_in_veh']) 
    combined_dict_new[key] = new_value

for hour, values in combined_dict_new.items():
    crowding_in_veh = []
    
    # Para cada ruta dentro de hour
    for route in values['routes']:
        total_crowding = 0
        for entry in paths_per_hour_it[route][hour]:
            line_id = entry['line']
            crowding_value = entry['crowding_value']     
            # Obtenemos el travel_time de la línea correspondiente desde LinesTimeCSV
            travel_time = LineTimeCSV.loc[LineTimeCSV['lineID'] == line_id, 'Time'].values[0]
            
            # Calculamos el nuevo valor de crowding_in_veh
            updated_crowding = float(crowding_value * ConW)
            total_crowding += updated_crowding
        
        # Añadimos el total de crowding calculado por ruta al diccionario
        crowding_in_veh.append(total_crowding)
    
    # Actualizamos el diccionario con el nuevo valor de crowding_in_veh
    combined_dict_new[hour] = {
        'durations': values['durations'],
        'routes': values['routes'],
        'in_veh_time': values['in_veh_time'],
        'waiting_time': values['waiting_time'],
        'gen_travel_cost': values['gen_travel_cost'],  # Permanece igual
        'gen_travel_cost_2': values['gen_travel_cost_2'],  # Permanece igual
        'crowding_in_veh': crowding_in_veh,  # Se actualiza
        'demand_prob_product': values['demand_prob_product'],
        'demand_prob_product_2': values['demand_prob_product_2']
    }

#%% Probabilities
## Recompute probabilities for combined_dict_it
probabilities_it = {}

# Calcular probabilidades de las rutas utilizando la duración de la ruta, el nuevo crowding_in_veh y gen_travel_cost
for key, value in combined_dict_new.items():
    # Obtener los costos de viaje y los nuevos valores de crowding_in_veh
    gen_travel_cost_it = np.array(value['gen_travel_cost'])
    crowding_costs_it = np.array(value['crowding_in_veh'])

    # Calcular el costo total para cada ruta (costos generados + crowding)
    total_cost_it = gen_travel_cost_it + crowding_costs_it

    # Calcular la exponencial negativa de los costos
    exp_neg_costs_it = np.exp(-total_cost_it)

    # Normalizar para obtener las probabilidades (modelo logit)
    total_exp_it = np.sum(exp_neg_costs_it)
    if total_exp_it == 0:
        total_exp_it = 1e-10  # Evitar división por cero

    probs_it = exp_neg_costs_it / total_exp_it

    # Asociar cada ruta con su probabilidad
    routes_it = value['routes']
    probabilities_it[key] = {route: float(probs_it[i]) for i, route in enumerate(routes_it)}


def calculate_route_probabilities(combined_dict_input):
    probabilities = {}

    for key, value in combined_dict_input.items():
        gen_travel_cost_it = np.array(value['gen_travel_cost'])
        crowding_costs_it = np.array(value['crowding_in_veh'])

        # Suma total de costos por ruta
        total_cost_it = gen_travel_cost_it #+ crowding_costs_it

        # Aplicar función logit (exponencial negativa de los costos)
        exp_neg_costs_it = np.exp(-total_cost_it)

        # Normalización
        total_exp_it = np.sum(exp_neg_costs_it)
        if total_exp_it == 0:
            total_exp_it = 1e-10  # Para evitar división por cero

        probs_it = exp_neg_costs_it / total_exp_it

        # Asociar a cada ruta su probabilidad
        routes_it = value['routes']
        probabilities[key] = {route: float(probs_it[i]) for i, route in enumerate(routes_it)}

    return probabilities
 
# function for recompute probailities for group demand 2

def calculate_route_probabilities_2(combined_dict_input):

    probabilities_2 = {}

    for key, value in combined_dict_input.items():

        gen_travel_cost_it = np.array(
            value['gen_travel_cost_2']
        )

        # Apply MNL
        exp_neg_costs = np.exp(-gen_travel_cost_it)

        total_exp = np.sum(exp_neg_costs)

        if total_exp == 0:
            total_exp = 1e-10

        probs = exp_neg_costs / total_exp

        routes = value['routes']

        probabilities_2[key] = {
            route: float(probs[i])
            for i, route in enumerate(routes)
        }

    return probabilities_2   
 
# recompute demand_prob_product para combined_dict_new

#%% Compute demand_prob_product

probability_demand_product_it = {}

for key, demand in demand_by_interval.items():
    # Verificamos si existe probabilidades para este intervalo en probabilities_it
    if key in probabilities_it:
        # Multiplicamos la demanda por la probabilidad de cada ruta
        prob_demand_list_it = [
            demand * prob  # Multiplicamos la demanda por la probabilidad
            for route, prob in probabilities_it[key].items()
        ]
        # Guardamos la lista en el diccionario
        probability_demand_product_it[key] = prob_demand_list_it

# Actualizar el combined_dict_it con el nuevo 'demand_prob_product'
for key, value in combined_dict_new.items():
    # Verificar si existe una lista de productos de demanda * probabilidad para este intervalo en probability_demand_product_it
    if key in probability_demand_product_it:
        # Agregar la lista de productos a combined_dict_it bajo la clave 'demand_prob_product'
        value['demand_prob_product'] = probability_demand_product_it[key]

print("DEMAND BY INTERVAL STRUCTURE", demand_by_interval)
print("demand by interval structure")

def update_demand_prob_product(
    demand_by_interval,
    probabilities_input
):

    probability_demand_product = {}

    # If total demand is zero, return zero flows
    if sum(demand_by_interval.values()) == 0:

        for key, route_probs in probabilities_input.items():

            probability_demand_product[key] = [
                0.0
                for _ in route_probs
            ]

        return probability_demand_product

    # Normal case
    for key, demand in demand_by_interval.items():

        if key in probabilities_input:

            probability_demand_product[key] = [
                float(demand) * float(prob)
                for prob in probabilities_input[key].values()
            ]

    # Demand conservation check
    total_demand = sum(demand_by_interval.values())

    total_assigned = sum(
        sum(values)
        for values in probability_demand_product.values()
    )

    error = total_demand - total_assigned

    if abs(error) > 1e-6:

        for key in reversed(
            sorted(probability_demand_product.keys())
        ):

            if probability_demand_product[key]:

                probability_demand_product[key][0] += error
                break

    return probability_demand_product



# Finalmente, si necesitas calcular el total travel cost considerando los productos demandados
total_travelcost_routes_it = 0.0  # Variable acumuladora para el total

# Recorremos cada intervalo de tiempo en combined_dict_it
for key, value in combined_dict_new.items():
    # Obtener las listas de 'demand_prob_product', 'gen_travel_cost' y 'routes' para este intervalo
    demand_prob_product_it = value.get('demand_prob_product', [])
    gen_travel_cost_it = value.get('gen_travel_cost', [])
    
    # Si existen las listas y tienen el mismo tamaño
    if len(demand_prob_product_it) == len(gen_travel_cost_it):
        # Calcular la suma de los productos para la función objetivo en este intervalo
        partial_sum_it = sum(d * g for d, g in zip(demand_prob_product_it, gen_travel_cost_it))
        
        # Sumar el resultado al total global
        total_travelcost_routes_it += partial_sum_it

#%%% NOW WE HAVE COMBINED_DICT AND COMBINED_DICT_NEW (afeter crowding)
#%% MSA AVERAGING FUNCTION

def average_passengers(
    combined_dict_previous,
    combined_dict_new,
    iteration
):

    combined_dict_iteration = {}

    # MSA step size
    alpha = 1.0 / iteration

    for interval in combined_dict_previous:

        combined_dict_iteration[interval] = copy.deepcopy(
            combined_dict_previous[interval]
        )

        # Previous iteration
        crowding_previous = combined_dict_previous[interval].get(
            'crowding_in_veh', []
        )

        demand_previous = combined_dict_previous[interval].get(
            'demand_prob_product', []
        )

        demand_previous_g2 = combined_dict_previous[interval].get(
            'demand_prob_product_2', []
        )

        # Newly calculated values
        crowding_new = combined_dict_new[interval].get(
            'crowding_in_veh', []
        )

        demand_new = combined_dict_new[interval].get(
            'demand_prob_product', []
        )

        demand_new_g2 = combined_dict_new[interval].get(
            'demand_prob_product_2', []
        )

        # MSA update
        avg_crowding = [
            old + alpha * (new - old)
            for old, new
            in zip(crowding_previous, crowding_new)
        ]

        avg_demand = [
            old + alpha * (new - old)
            for old, new
            in zip(demand_previous, demand_new)
        ]

        avg_demand_g2 = [
            old + alpha * (new - old)
            for old, new
            in zip(demand_previous_g2, demand_new_g2)
        ]

        combined_dict_iteration[interval][
            'crowding_in_veh'
        ] = avg_crowding

        combined_dict_iteration[interval][
            'demand_prob_product'
        ] = avg_demand

        combined_dict_iteration[interval][
            'demand_prob_product_2'
        ] = avg_demand_g2

    return combined_dict_iteration

   
#%% LOOP Mean Successive Average (MSA)  

iteration_counter = 0
max_iterations = 25  # Definimos el máximo de iteraciones
converged = False  # Variable de convergencia

# Paso 1: Promedio de pasajeros para la primera iteracion
combined_dict_msa =  combined_dict_new

while not converged and iteration_counter < max_iterations:
    iteration_counter += 1
    

    # Paso 2: Redistribuir la demanda en el dataframe
    crowding_table = pd.DataFrame(lines_data)
    crowding_distributed_msa = distribuir_demanda(crowding_table, combined_dict_msa, paths_per_hour)

    # Paso 3: Total de pasajeros a bordo
    crowding_total = crowding_distributed_msa.copy()
    non_seating_msa = compute_non_seating_diff_df(seating_cap, crowding_total)
    crowding_prop_msa = compute_crowding_proportion(crowding_total, non_seating_msa, crowding_simulation_empty)

    # Paso 4: Actualización de paths_per_hour con el peso de crowding
    paths_per_hour_msa = update_paths_with_crowding(paths_per_hour, crowding_prop_msa)

    # Paso 5: Inicializar el diccionario para la siguiente iteración
    combined_dict_msa_it = copy.deepcopy(combined_dict_new)

    # Paso 6: Reset values before new assignment
    for key, value in combined_dict_msa_it.items():
    
        if 'crowding_in_veh' in value:
            value['crowding_in_veh'] = (
                [0.0] * len(value['crowding_in_veh'])
            )
    
        if 'demand_prob_product' in value:
            value['demand_prob_product'] = (
                [0.0] * len(value['demand_prob_product'])
            )
    
        if 'demand_prob_product_2' in value:
            value['demand_prob_product_2'] = (
                [0.0] * len(value['demand_prob_product_2'])
            )

    # Paso 7: Calcular 'crowding_in_veh' para cada ruta
    for hour, values in combined_dict_msa_it.items():
        crowding_in_veh = []

        # Para cada ruta dentro de 'hour'
        for route in values['routes']:
            total_crowding = 0
            for entry in paths_per_hour_msa[route][hour]:
                line_id = entry['line']
                crowding_value = entry['crowding_value']     
                # Obtener el tiempo de viaje de la línea correspondiente
                travel_time = LineTimeCSV.loc[LineTimeCSV['lineID'] == line_id, 'Time'].values[0]
                
                # Actualizar el valor de 'crowding_in_veh'
                updated_crowding = float(crowding_value * ConW)
                total_crowding += updated_crowding
            
            # Añadir el total de 'crowding_in_veh' para esta ruta
            crowding_in_veh.append(total_crowding)
        
        # Actualizar el diccionario con el nuevo valor de 'crowding_in_veh'
        combined_dict_msa_it[hour] = {
            'durations': values['durations'],
            'routes': values['routes'],
            'in_veh_time': values['in_veh_time'],
            'waiting_time': values['waiting_time'],
            'gen_travel_cost': values['gen_travel_cost'],  # Permanece igual
            'gen_travel_cost_2': values['gen_travel_cost_2'],
            'crowding_in_veh': crowding_in_veh,  # Se actualiza
            'demand_prob_product': values['demand_prob_product'],
            'demand_prob_product_2': values['demand_prob_product_2']
        }

    # Paso 8: Recalcular probabilidades (only for demand_regular)
    probabilities_msa_it = calculate_route_probabilities (combined_dict_msa_it)
    probabilities_msa_it_2 = calculate_route_probabilities_2 (combined_dict_msa_it)

    # Paso 9: Recalcular 'demand_prob_product'
    demand_prob_product_msa_it = update_demand_prob_product(demand_by_interval, probabilities_msa_it)
    # Actualizar el combined_dict con el nuevo 'demand_prob_product'
    for key, value in combined_dict_msa_it.items():
        if key in demand_prob_product_msa_it:
            # Agregar la lista de productos a combined_dict_it bajo la clave 'demand_prob_product'
            value['demand_prob_product'] = demand_prob_product_msa_it[key]
    
    #recalculate demand_prob_product for demand_2
    demand_prob_product_msa_it_2 = update_demand_prob_product(demand_by_interval_2, probabilities_msa_it_2)
    for key, value in combined_dict_msa_it.items():
        if key in demand_prob_product_msa_it_2:
            value['demand_prob_product_2'] = demand_prob_product_msa_it_2[key]

    # Paso 10: Aplicar actualización MSA
    combined_new_iteration = average_passengers(
        combined_dict_msa,
        combined_dict_msa_it,
        iteration_counter
    )

    # Mostrar el 'demand_prob_product' de esta iteración
    print(f"Iteración {iteration_counter} - demand_prob_product:")
    for key in combined_dict_msa:
        print(f"  {key}: {combined_dict_msa[key]['demand_prob_product']}")
    
    # Mostrar el 'crowding_in_veh' de esta iteración
    print(f"Iteración {iteration_counter} - crowding_in_veh:")
    for key in combined_dict_msa:
        print(f"  {key}: {combined_dict_msa[key]['crowding_in_veh']}")
        
    # Mostrar el 'crowding_in_veh' de esta iteración
    print(f"Iteración {iteration_counter} - gen_travel_cost:")
    for key in combined_dict_msa:
        print(f"  {key}: {combined_dict_msa[key]['gen_travel_cost']}")

    # Paso 10: Aplicar actualización MSA
    combined_new_iteration = average_passengers(
        combined_dict_msa,
        combined_dict_msa_it,
        iteration_counter
    )
    
    
    # Paso 11: Verificación de convergencia
    converged = True
    
    for key in combined_dict_msa:
    
        if key in combined_new_iteration:
    
            prev_demand_prob = (
                combined_dict_msa[key]['demand_prob_product']
            )
            new_demand_prob = (
                combined_new_iteration[key]['demand_prob_product']
            )
    
            prev_demand_prob_2 = (
                combined_dict_msa[key]['demand_prob_product_2']
            )
            new_demand_prob_2 = (
                combined_new_iteration[key]['demand_prob_product_2']
            )
    
            for (
                prev_val,
                new_val,
                prev_val_2,
                new_val_2
            ) in zip(
                prev_demand_prob,
                new_demand_prob,
                prev_demand_prob_2,
                new_demand_prob_2
            ):
    
                if (
                    abs(prev_val - new_val) >= 0.001
                    or
                    abs(prev_val_2 - new_val_2) >= 0.001
                ):
                    converged = False
                    break
    
        if not converged:
            break
    
    
    # Paso 12: Actualizar solución MSA
    combined_dict_msa = combined_new_iteration
    
    
    if iteration_counter == max_iterations and not converged:
        print(
            f"🚨 Max number of iterations "
            f"({max_iterations}). "
            f"Convergence not reached."
        )
        break


#%% CAPACITY FUNCTION 

def max_capacity_combined_dict(combined_dict, starting_lines, unique_start_times, seating_cap, route_summary_lines):
    # Crear el DataFrame 'demand_initial_platform'
    demand_initial_platform = pd.DataFrame(
        0.0,
        index=range(len(starting_lines)),
        columns=unique_start_times
    )
    
    cost_for_overcrowding = 0.0

    # Agregar la columna LineId
    demand_initial_platform.insert(0, 'LineId', starting_lines)

    # Crear 'demand_for_platforms' a partir de 'combined_dict'
    demand_for_platforms = {}
    for key, value in combined_dict.items():
        demand_for_platforms[key] = {
            'routes': value['routes'],
            'demand_prob_product': value['demand_prob_product'],
            'demand_prob_product_2': value['demand_prob_product_2'],
        }

    # Llenar el DataFrame 'demand_initial_platform'
    for time, info in demand_for_platforms.items():
        routes = info['routes']
        demand1 = info['demand_prob_product']
        demand2 = info['demand_prob_product_2']

        for idx, route in enumerate(routes):
            total_demand = demand1[idx] + demand2[idx]
            line_id = route_summary_lines[route][0]

            row_idx = demand_initial_platform.index[demand_initial_platform['LineId'] == line_id]
            if not row_idx.empty:
                demand_initial_platform.loc[row_idx, time] += total_demand

    # Definir FullCap
    line_full_caps = {
        line_id: seating_cap[line_id]['FullCap']
        for line_id in starting_lines
    }

    # Copiar combined_dict para no modificar el original
    combined_dict_capacity = {}
    combined_dict_capacity = combined_dict.copy()

    for idx, row in demand_initial_platform.iterrows():
        line_id = row['LineId']
        full_cap = line_full_caps[line_id]

        for time_idx, time in enumerate(unique_start_times):
            value = row[time]

            if value > full_cap:
                excess = value - full_cap
                

                # 1. Encontrar rutas afectadas (que contengan line_id)
                routes_current = []
                indexes_current = []
                for route_idx, route in enumerate(combined_dict_capacity[time]['routes']):
                    if line_id in route_summary_lines[route]:
                        routes_current.append(route)
                        indexes_current.append(route_idx)

                # 2. Extraer los demand_prob_product actuales de esas rutas
                demands_current = [combined_dict_capacity[time]['demand_prob_product'][i] for i in indexes_current]
                total_demand_current = sum(demands_current)

                if total_demand_current == 0:
                    continue  # Evitar división por cero

                # 3. Reducir proporcionalmente los demand_prob_product
                for i, idx_route in enumerate(indexes_current):
                    proportion = demands_current[i] / total_demand_current
                    combined_dict_capacity[time]['demand_prob_product'][idx_route] -= float(excess * proportion)

                # 4. Agregar exceso proporcionalmente al siguiente start_time (si existe)
                if time_idx + 1 < len(unique_start_times):
                    next_time = unique_start_times[time_idx + 1]
                    routes_next = combined_dict_capacity[next_time]['routes']
                    demands_next = combined_dict_capacity[next_time]['demand_prob_product']
                    total_costs_next = []
                    for route_idx, route in enumerate(routes_next):
                        
                        gen_cost = combined_dict_capacity[next_time]['gen_travel_cost'][route_idx]
                        crowding_cost = combined_dict_capacity[next_time]['crowding_in_veh'][route_idx]
                        total_cost = gen_cost + crowding_cost
                        total_costs_next.append(total_cost)
                    exp_neg_costs_next = np.exp(-np.array(total_costs_next))
                    sum_exp_next = np.sum(exp_neg_costs_next)
                    if sum_exp_next == 0:
                        sum_exp_next = 1e-10
                    probs_next = exp_neg_costs_next / sum_exp_next
                    for j in range(len(demands_next)):
                        combined_dict_capacity[next_time]['demand_prob_product'][j] += float(excess * probs_next[j])
                        
                index_from = time_idx
                index_to = time_idx + 1
                start_time_real = unique_start_times[index_from]
                end_time_real = unique_start_times[index_to]
                index_difference = float(end_time_real - start_time_real)
                cost_for_overcrowding += float(excess*index_difference*0.575)
                
                
    return combined_dict_capacity, cost_for_overcrowding

new_test_ari, ari_cost_for_overcrowding = max_capacity_combined_dict(combined_dict, starting_lines, unique_start_times, seating_cap, route_summary_lines)




#%% FUNCTION 
print("demand_by_interval", demand_by_interval)

def MSA_Convergence(combined_dict_new, starting_lines, unique_start_times, route_summary_lines , lines_data, paths_per_hour, seating_cap, LineTimeCSV, ConW, demand_by_interval, demand_by_interval_2,  max_iterations=25, convergence_threshold=0.001):
    iteration_counter = 0
    converged = False
    combined_dict_msa = combined_dict_new
    
    combined_dict_msa_cap, sim_cost_for_overcrowding = max_capacity_combined_dict(combined_dict_msa, starting_lines, unique_start_times, seating_cap, route_summary_lines)
    
    combined_dict_msa = combined_dict_msa_cap

    while not converged and iteration_counter < max_iterations:
        iteration_counter += 1
        
        # Paso 2: Redistribuir la demanda en el dataframe
        crowding_table = pd.DataFrame(lines_data)
        crowding_distributed_msa = distribuir_demanda(crowding_table, combined_dict_msa, paths_per_hour)

        # Paso 3: Total de pasajeros a bordo
        crowding_total = crowding_distributed_msa.copy()
        non_seating_msa = compute_non_seating_diff_df(seating_cap, crowding_total)
        crowding_prop_msa = compute_crowding_proportion(crowding_total, non_seating_msa, crowding_simulation_empty)

        # Paso 4: Actualización de paths_per_hour con el peso de crowding
        paths_per_hour_msa = update_paths_with_crowding(paths_per_hour, crowding_prop_msa)

        # Paso 5: Inicializar el diccionario para la siguiente iteración
        combined_dict_msa_it = copy.deepcopy(combined_dict_new)

        # Paso 6: Reset values before new assignment
        for key, value in combined_dict_msa_it.items():
        
            if 'crowding_in_veh' in value:
                value['crowding_in_veh'] = (
                    [0.0] * len(value['crowding_in_veh'])
                )
        
            if 'demand_prob_product' in value:
                value['demand_prob_product'] = (
                    [0.0] * len(value['demand_prob_product'])
                )
        
            if 'demand_prob_product_2' in value:
                value['demand_prob_product_2'] = (
                    [0.0] * len(value['demand_prob_product_2'])
                )

        # Paso 7: Calcular 'crowding_in_veh' para cada ruta
        for hour, values in combined_dict_msa_it.items():
            crowding_in_veh = []

            # Para cada ruta dentro de 'hour'
            for route in values['routes']:
                total_crowding = 0
                for entry in paths_per_hour_msa[route][hour]:
                    line_id = entry['line']
                    crowding_value = entry['crowding_value']     
                    # Obtener el tiempo de viaje de la línea correspondiente
                    travel_time = LineTimeCSV.loc[LineTimeCSV['lineID'] == line_id, 'Time'].values[0]
                    
                    # Actualizar el valor de 'crowding_in_veh'
                    updated_crowding = float(crowding_value * ConW)
                    total_crowding += updated_crowding
                
                # Añadir el total de 'crowding_in_veh' para esta ruta
                crowding_in_veh.append(total_crowding)
            
            # Actualizar el diccionario con el nuevo valor de 'crowding_in_veh'
            combined_dict_msa_it[hour] = {
                'durations': values['durations'],
                'routes': values['routes'],
                'in_veh_time': values['in_veh_time'],
                'waiting_time': values['waiting_time'],
                'gen_travel_cost': values['gen_travel_cost'],  # Permanece igual
                'gen_travel_cost_2': values['gen_travel_cost_2'],  # Permanece igual
                'crowding_in_veh': crowding_in_veh,  # Se actualiza
                'demand_prob_product': values['demand_prob_product'],
                'demand_prob_product_2': values['demand_prob_product_2']
            }

        # Paso 8: Recalcular probabilidades
        probabilities_msa_it = calculate_route_probabilities(combined_dict_msa_it)
        probabilities_msa_it_2 = calculate_route_probabilities_2 (combined_dict_msa_it)

        # Paso 9: Recalcular 'demand_prob_product'
        demand_prob_product_msa_it = update_demand_prob_product(demand_by_interval, probabilities_msa_it)

        # Actualizar el combined_dict con el nuevo 'demand_prob_product'
        for key, value in combined_dict_msa_it.items():
            if key in demand_prob_product_msa_it:
                value['demand_prob_product'] = demand_prob_product_msa_it[key]
        
        demand_prob_product_msa_it_2 = update_demand_prob_product(demand_by_interval_2, probabilities_msa_it_2)
        for key, value in combined_dict_msa_it.items():
            if key in demand_prob_product_msa_it_2:
                value['demand_prob_product_2'] = demand_prob_product_msa_it_2[key]

        # Paso 10: Aplicar actualización MSA
        combined_new_iteration = average_passengers(
            combined_dict_msa,
            combined_dict_msa_it,
            iteration_counter
        )
        
        
        # Paso 11: Verificación de convergencia
        converged = True
        
        for key in combined_dict_msa:
        
            if key in combined_new_iteration:
        
                prev_demand_prob = (
                    combined_dict_msa[key]['demand_prob_product']
                )
                new_demand_prob = (
                    combined_new_iteration[key]['demand_prob_product']
                )
        
                prev_demand_prob_2 = (
                    combined_dict_msa[key]['demand_prob_product_2']
                )
                new_demand_prob_2 = (
                    combined_new_iteration[key]['demand_prob_product_2']
                )
        
                for (
                    prev_val,
                    new_val,
                    prev_val_2,
                    new_val_2
                ) in zip(
                    prev_demand_prob,
                    new_demand_prob,
                    prev_demand_prob_2,
                    new_demand_prob_2
                ):
        
                    if (
                        abs(prev_val - new_val) >= convergence_threshold
                        or
                        abs(prev_val_2 - new_val_2) >= convergence_threshold
                    ):
                        converged = False
                        break
        
            if not converged:
                break
        
        
        # Paso 12: Actualizar solución MSA
        combined_dict_msa = combined_new_iteration

        if iteration_counter == max_iterations and not converged:
            print(
                f"🚨 Max number of iterations "
                f"({max_iterations}). "
                f"Convergence not reached."
            )
            break

    if converged:
        print(
            f"✅ Convergence of MSA FUNCTION reached "
            f"in {iteration_counter} iterations!"
        )

    return combined_dict_msa, sim_cost_for_overcrowding

# Ejecutar MSA
combined_dict_msa, msa_cost_for_overcrowding = MSA_Convergence(
    combined_dict_new,
    starting_lines,
    unique_start_times,
    route_summary_lines,
    lines_data,
    paths_per_hour,
    seating_cap,
    LineTimeCSV,
    ConW,
    demand_by_interval,
    demand_by_interval_2,
    max_iterations=25,
    convergence_threshold=0.05
)


#%% FUNCTION TO COMPUTE GENERALIZED COST - PASSENGER COST
#input combined_dict_msa

def compute_passenger_cost(combined_dict_msa):
    sum_total = 0
    for key, data in combined_dict_msa.items():
        for cost, crowding, demand, cost_2, demand_2 in zip(data['gen_travel_cost'], data['crowding_in_veh'], data['demand_prob_product'], data['gen_travel_cost_2'], data['demand_prob_product_2']):
            sum_total += demand * (cost) + demand_2 * (cost_2) # mising corwding
    return sum_total

passenger_cost_simulation = compute_passenger_cost(combined_dict_msa)

print(" ")
print("Passenger Cost based on combined dict ")
print(passenger_cost_simulation)
print(" ")

# %% COMPUTE OPERATIONAL COST

line_dict_fre_sche = {}


for idx in range(len(LinesCSV)):
    line_dict_fre_sche[idx] = {
        'SerType': LinesCSV.loc[idx, 'SerType'],
        'HeadWay': int(LinesCSV.loc[idx, 'HeadWay']),
        'Name': LinesCSV.loc[idx, 'Name'],
        'timetable': timetables_list[idx]
    }

cost_total_freq = 0.0
cost_total_schedule = 0.0

headway_fr_cost = 100
schedule_cost = 150

for line in line_dict_fre_sche.values():
    if line['SerType'] == 'Frequency':
        cost_total_freq += 60 / line['HeadWay']*headway_fr_cost
for line in line_dict_fre_sche.values():
    if line['SerType'] == 'Schedule':
        cost_total_schedule += 60 / line['HeadWay']*schedule_cost

operational_cost_sche_fre = cost_total_freq + cost_total_schedule

# %% FUNCTION OPERATIONAL COST
# input line_dict_fre_sche

def calculate_line_costs_by_timetable_count(line_dict_fre_sche, headway_fr_cost=100, schedule_cost=150):
    cost_per_line = {}
    total_cost = 0.0

    for line_id, line_data in line_dict_fre_sche.items():
        timetable_len = len(line_data['timetable'])
        if timetable_len == 0:
            cost = 0  # evitar división por cero
        else:
            if line_data['SerType'] == 'Frequency':
                cost = (60 / timetable_len) * headway_fr_cost
            elif line_data['SerType'] == 'Schedule':
                cost = (60 / timetable_len) * schedule_cost
            else:
                cost = 0  # tipo desconocido

        cost_per_line[line_id] = cost
        total_cost += cost

    return total_cost

#%% FUNCTION TO COMPUTE WAITING COST
# Input demand_data_frame

def sum_partial_waiting_time(demand_data_frame):
    return demand_data_frame['Partial_waiting_time'].sum()

#%% function to generate timtables 

def generate_timetables(start_times, headway_list, max_time=120):
    timetables_list_opt = []
    
    for start, headway in zip(start_times, headway_list):
        timetable = list(range(start, max_time + 1, headway))
        timetables_list_opt.append(timetable)
    
    return timetables_list_opt


#%% function to obtain operational cost from timetable_list

line_sertype_dict = dict(zip(LinesCSV['id'], LinesCSV['SerType']))
headway_fr_cost = 100
schedule_cost = 150

def compute_operational_cost(headway_list, line_sertype_dict, schedule_cost=150, headway_fr_cost=100):
    total_cost = 0

    for line_id, headway in enumerate(headway_list):
        ser_type = line_sertype_dict.get(line_id, 'Schedule')  # Default 'Schedule'
        cost_per_trip = schedule_cost if ser_type == 'Schedule' else headway_fr_cost
        line_cost = (120 / headway) * cost_per_trip
        total_cost += line_cost

    return total_cost

#%% FUNCTION TO UPDATE THE TIMETABLE OF ROUTES_DICT

def update_routes_timetable(routes_dicts, timetables_list_opt):
    # Crear una copia profunda del diccionario original
    routes_dict_opt = copy.deepcopy(routes_dicts)
    
    # Obtener todos los lineids en orden para mapearlos con los nuevos horarios
    line_ids = []
    for route in routes_dicts.values():
        for line in route.values():
            line_ids.append(line["lineid"])
    
    # Asegurar que no haya más líneas que timetables disponibles
    unique_line_ids = sorted(set(line_ids))  # Obtener IDs únicos ordenados
    
    if len(unique_line_ids) != len(timetables_list_opt):
        raise ValueError("El número de lineas y timetables no coincide.")
    
    # Crear un diccionario para mapear lineid con su nuevo timetable
    lineid_to_timetable = {line_id: timetable for line_id, timetable in zip(unique_line_ids, timetables_list_opt)}
    
    # Actualizar el timetable en la copia de routes_dicts
    for route in routes_dict_opt.values():
        for line in route.values():
            line_id = line["lineid"]
            if line_id in lineid_to_timetable:
                line["timetable"] = lineid_to_timetable[line_id]

    return routes_dict_opt

#%%
def update_paths_with_crowding_new(paths_per_hour, crowding_proportion):
    paths_per_hour_it = copy.deepcopy(paths_per_hour)

    for route, times in paths_per_hour_it.items():
        for departure_time, trips in times.items():
            departure_time = str(departure_time)  # Asegurar que sea string
            for trip in trips:
                line_id = trip['line']
                
                if line_id not in crowding_proportion['LineID'].values:
                    print(f"Advertencia: line_id {line_id} no encontrado en crowding_proportion.")
                    trip['crowding_value'] = 0.0
                    continue
                
                filtered_rows = crowding_proportion[crowding_proportion['LineID'] == line_id]

                if departure_time in filtered_rows.columns:
                    crowding_value = filtered_rows[departure_time].values
                    trip['crowding_value'] = float(crowding_value[0]) if crowding_value.size > 0 else 0.0
                else:
                    print(f"Advertencia: departure_time {departure_time} no encontrado en crowding_proportion.")
                    trip['crowding_value'] = 0.0

    return paths_per_hour_it

#%% CAOMPUTE WAITING TIME OF SIMULATION USANDO LAS UNIQUE START TIMES Y LA DEMAND_DICT

def compute_waiting_time_cumulated(demand_dict, unique_start_times, WaitW):
    total_waiting_time = 0

    unique_start_times = sorted(unique_start_times)  # Asegurar orden

    for time, demand in demand_dict.items():
        # Buscar el primer tiempo en unique_start_times que sea >= time
        future_departures = [start for start in unique_start_times if start >= time]
        if not future_departures:
            continue  # No hay salida futura para este grupo, ignoramos o asumimos espera infinita

        next_departure = future_departures[0]
        wait_time = next_departure - time

        total_waiting_time += wait_time * demand * WaitW

    return total_waiting_time

# #%% OPTIMIZATON INITIATES HERE

# Parámetros iniciales
headway_fr_cost = 100
schedule_cost = 150

# Definir las listas de start_times y headway_list de acuerdo a los valores en timetables_list
start_times = [t[0] for t in timetables_list]  # Extraer los tiempos de inicio
headway_list = [t[1] - t[0] for t in timetables_list]  # Calcular los intervalos de tiempo (headways)


print("Start Optimization")
print("start_times initial", start_times)
print("headway_list initial", headway_list)

#%% FUNCTION TO MAKE DEMAND SUMMARY FOR ALL REOUTES SELECTION

def create_demand_summary(combined_dict):
    # Inicializamos el nuevo diccionario
    demand_summary_simulation = {}
    
    # Asumimos que las rutas son siempre las mismas
    rutas = list(route_summary_lines.keys())
    
    # Inicializamos cada ruta
    for ruta in rutas:
        demand_summary_simulation[ruta] = {'group_1': 0.0, 'group_2': 0.0}
    
    # Ahora recorremos el combined_dict
    for _, data in combined_dict.items():
        routes = data['routes']
        demand_prob_product = data['demand_prob_product']
        demand_prob_product_2 = data['demand_prob_product_2']
        
        for i, ruta in enumerate(routes):
            demand_summary_simulation[ruta]['group_1'] += demand_prob_product[i]
            demand_summary_simulation[ruta]['group_2'] += demand_prob_product_2[i]
    
    return demand_summary_simulation

demand_summary_dict = create_demand_summary(combined_dict)

#%%

##funtion to gest GENERALIZED COST given start_times and headway_list
def objective_function(start_times, headway_list):
    
    routes_dicts_00 = copy.deepcopy(routes_dicts)
    g_paths_merge_00 = copy.deepcopy(g_paths_merge)
    # GENERAR HORARIOS OPTIMIZADOS
    timetables_list_opt = generate_timetables(start_times, headway_list, max_time=120)
    print(" Timetables_optimization")
    print(timetables_list_opt)
    print(" --")
    # CÁLCULO DE COSTO OPERACIONAL
    operational_cost = compute_operational_cost(headway_list, line_sertype_dict, schedule_cost=150, headway_fr_cost=100)

    # ACTUALIZAR RUTAS CON LOS HORARIOS OPTIMIZADOS
    
    routes_dict_opt = update_routes_timetable(routes_dicts_00, timetables_list_opt)

    # GENERAR RUTAS POR HORA
    paths_per_hour_opt = {}
    numb_choices = len(g_paths_merge_00)
    for i in range(numb_choices):
        route_key = f"route_{i+1}"
        dict_of_route = routes_dict_opt[route_key]
        sequ_of_route_opt = g_paths_merge_00[i]
        route_times_result = calculate_route_times(dict_of_route, sequ_of_route_opt)
        paths_per_hour_opt[route_key] = route_times_result
    
    ## eliminate paths per hour with arrival time after 120!
    for route, departures in paths_per_hour_opt.items():
        keys_to_delete = []
        for key, steps in departures.items():
            if any(step['arrival_time'] > 120 for step in steps):
                keys_to_delete.append(key)
        for key in keys_to_delete:
            del paths_per_hour_opt[route][key]
    

    # CALCULAR DURACIONES DE CADA RUTA
    durations_opt = {}
    for route, times in paths_per_hour_opt.items():
        durations_opt[route] = {}
        for start_time, lines in times.items():
            departure_time_initial = lines[0]['departure_time']
            arrival_time_final = lines[-1]['arrival_time']
            total_duration = arrival_time_final - departure_time_initial
            durations_opt[route][start_time] = {
                "start_time": start_time,
                "total_duration": total_duration
            }

    # COMBINAR DATOS DE DURACIONES Y RUTAS
    start_times_by_route_opt = {route: list(data.keys()) for route, data in durations_opt.items()}
    combined_dict_opt = {}
    unique_start_times_opt = sorted(set(time for times in start_times_by_route_opt.values() for time in times))
    
    # print(" ")
    # print("unique_start_times")
    # print(unique_start_times_opt)
    # print(" ---------")
    
    for time in unique_start_times_opt:
        combined_dict_opt[time] = [route for route, times in start_times_by_route_opt.items() if time in times]

    # INCLUIR COSTOS, TIEMPOS DE VIAJE Y ESPERA
    for start_time, routes in combined_dict_opt.items():
        durations = [durations_opt[route][start_time]['total_duration'] for route in routes if start_time in durations_opt.get(route, {})]
        combined_dict_opt[start_time] = {'routes': routes, 'durations': durations}

    # INCLUIR TIEMPOS EN VEHÍCULO Y TIEMPOS DE ESPERA
    route_in_veh_time = {f'route_{i+1}': int(sum(LineTimeCSV.loc[LineTimeCSV['lineID'].isin(route), 'Time'])) for i, route in enumerate(g_paths_merge_00)}
    
    ## Compute in_veh_time (generalized cost)
    for key, value in combined_dict_opt.items():
        value["in_veh_time"] = [route_in_veh_time[route] for route in value["routes"] if route in route_in_veh_time]
    
    ## Compute waiting_time (Headway/2 for all lines excep the first one)
    for key, value in combined_dict_opt.items():
        total_waiting_time = []
        for route in value["routes"]:
            lines_in_route = route_summary_lines.get(route, [])
            # Excluir la primera línea
            subsequent_lines = lines_in_route[1:]

            # Calcular el tiempo de espera como headway / 2 para cada línea subsecuente
            waiting_time_per_route = sum((headway_list[line] / 2) for line in subsequent_lines)
            total_waiting_time.append(waiting_time_per_route)
        
        value["waiting_time"] = total_waiting_time
    
    
    ## storage transfer_penalty
    route_transfer_penalty_opt = route_transfer_penalty.copy()
    
    for start_time, value in combined_dict_opt.items():
        route_list = value['routes']
        penalty_route = [route_transfer_penalty_opt.get(route, 0.0) for route in route_list]
        combined_dict_opt[start_time]['penalty_route'] = penalty_route


    # CALCULAR COSTO DE VIAJE GENERADO
    for key, value in combined_dict_opt.items():
        value["gen_travel_cost"] = [
            waiting_time * WaitW + route_inveh_perc[route]  + penalty
            for waiting_time, route, penalty in zip(value["waiting_time"], value["routes"], value["penalty_route"])
        ]
    
    for key, value in combined_dict_opt.items():
        value["gen_travel_cost_2"] = [
            waiting_time * WaitW + route_inveh_perc[route]  + penalty*2.0
            for waiting_time, route, penalty in zip(value["waiting_time"], value["routes"], value["penalty_route"])
        ]

    # INCLUIR CARGA EN VEHÍCULO
    for time, data in combined_dict_opt.items():
        data['crowding_in_veh'] = [0.0] * len(data['routes'])
        
    
    # SIMULACIÓN DE DEMANDA ------ SPLIT X 2 CATEGORIES
    accumulated_demand_opt = compute_accumulated_demand(demand_dict, unique_start_times_opt)
    accumulated_demand_opt_2 = compute_accumulated_demand(demand_dict_2, unique_start_times_opt)
    
    demand_intervals = list(range(0, 80, 1))
    
    demand_by_interval_opt_1={}
    demand_by_interval_opt_2={}
    
    demand_by_interval_opt_1 = {time: accumulated_demand_opt.get(time, 0) for time in demand_intervals}
    demand_by_interval_opt_2 = {time: accumulated_demand_opt_2.get(time, 0) for time in demand_intervals}
    
    # print(" ")
    # print("demand_by_interval_opt_1")
    # print(demand_by_interval_opt_1)
    # print(" ---------")
    
    # CORRETC!-------------------------------------------------------------
    
    #INSERT FUNCTION TO VERIFY DEMAND_BY_INTERVAL_OPT_2
    
    # CALULAR WAITING TIME CUMULATED OF ENTIRE SIMULATION
    total_waiting_opts = compute_waiting_time_cumulated(demand_dict, unique_start_times_opt, WaitW)
    total_waiting_opts_2 = compute_waiting_time_cumulated(demand_dict_2, unique_start_times_opt, WaitW)

    # CALCULAR PROBABILIDADES DE RUTA
    probability_opt = calculate_route_probabilities(combined_dict_opt)
    probability_opt_2 = calculate_route_probabilities_2(combined_dict_opt)
    # CORRETC!-------------------------------------------------------------

    # INCLUIR DEMANDA Y PRODUCTO DE PROBABILIDAD
    demand_prob_product_preMSA = update_demand_prob_product(demand_by_interval_opt_1, probability_opt)
    demand_prob_product_preMSA_2 = update_demand_prob_product(demand_by_interval_opt_2, probability_opt_2)
    
    # print(" ")
    # print("demand_prob_product_preMSA")
    # print(demand_prob_product_preMSA)
    # print(" ---------")
    # CORRETC!-------------------------------------------------------------
    
    for time, data in combined_dict_opt.items():
        data['demand_prob_product'] = demand_prob_product_preMSA.get(time, [0.0] * len(data['routes']))
    
    for time, data in combined_dict_opt.items():
        data['demand_prob_product_2'] = demand_prob_product_preMSA_2.get(time, [0.0] * len(data['routes']))
        
    # print(" ")
    # print("combined_dict_opt")
    # print(combined_dict_opt)
    # print(" ---------")
    
    Passenger_cost_routes = compute_passenger_cost(combined_dict_opt)
    
    demand_summary_dict = create_demand_summary(combined_dict_opt)
    # print("--")
    # print("summary demand group 1 and group 2")
    # print(demand_summary_dict)
    # print("--")
    
    # CORRETC!-------------------------------------------------------------

    # SIMULACIÓN DE AGLOMERACIÓN
    crowding_simulation_opt = crowding_simulation_empty
    crowding_empty= crowding_simulation_empty.copy()
    
    ## MSA while
    
    combined_dict_for_cost =  copy.deepcopy(combined_dict_opt) 
    
    cost_for_overcrowding_opt = 0.0
    combined_dict_for_cost, cost_for_overcrowding_opt = MSA_Convergence(combined_dict_opt, starting_lines, unique_start_times_opt, route_summary_lines, lines_data, paths_per_hour_opt, seating_cap, LineTimeCSV, ConW, demand_by_interval_opt_1, demand_by_interval_opt_2, max_iterations=40, convergence_threshold=0.001)

    
    # print(" ")
    # print("combined_dict_for_cost")
    # print(combined_dict_for_cost)
    # print(" ---------")

    crowding_simulation_opt = distribuir_demanda(pd.DataFrame([{'LineID': row['id'], 'Name': row['Name'], **{t: 0 for t in range(121)}} for _, row in LinesCSV.iterrows()]), combined_dict_for_cost, paths_per_hour_opt)

    crowding_non_seating_opt = compute_non_seating_diff_df(seating_cap, crowding_simulation_opt)
    
    crowding_prop_msa_opt = compute_crowding_proportion(crowding_simulation_opt, crowding_non_seating_opt, crowding_simulation_empty)

    # ACTUALIZAR RUTAS CON NUEVAS AGLOMERACIONES
    paths_per_hour_msa_opt = update_paths_with_crowding(paths_per_hour_opt, crowding_prop_msa_opt)


    combined_dict_new_opt = copy.deepcopy(combined_dict_for_cost)
    for hour, values in combined_dict_new_opt.items():
        crowding_in_veh = []
        for route in values['routes']:
            total_crowding = 0
            for entry in paths_per_hour_msa_opt[route][hour]:
                line_id = entry['line']
                crowding_value = entry['crowding_value']
                travel_time = LineTimeCSV.loc[LineTimeCSV['lineID'] == line_id, 'Time'].values[0]
                updated_crowding = float(crowding_value * ConW)
                total_crowding += updated_crowding
            crowding_in_veh.append(total_crowding)
        values['crowding_in_veh'] = crowding_in_veh
    
    
        
    demand_check_g1g2 = 0.0
    for route_data in demand_summary_dict.values():
        demand_check_g1g2 += route_data['group_1'] + route_data['group_2'] 

    # CÁLCULO DE COSTO DE PASAJEROS
    
    print(" ")
    print("Passenger_cost_routes")
    print(Passenger_cost_routes)
    print("Operational_cost")
    print(operational_cost)
    print("Waiting_time_cost")
    print(total_waiting_opts)
    print("Waiting_time_cost_2")
    print(total_waiting_opts_2)
    print("cost_for_overvrowding")
    print(cost_for_overcrowding_opt)

    # DEFINIMOS EL COSTO FINAL QUE QUEREMOS MINIMIZAR
    
    global cplex_total_cost
    cplex_total_cost = float(Passenger_cost_routes) + operational_cost + float(total_waiting_opts) +  float(total_waiting_opts_2) + float(cost_for_overcrowding_opt)
    print(" total cost ", cplex_total_cost)
    print("end of run of objective function -------------------------------------------------------")

    return cplex_total_cost, demand_summary_dict, combined_dict_new_opt, demand_check_g1g2


#%% OPTUNA OPTIMIZATION

# Store results from all trials
all_trials_results = []

# ------------------------------------------------------------------
# OPTUNA PARAMETERS
# ------------------------------------------------------------------

START_TIME_MIN = 0
START_TIME_MAX = 20

HEADWAY_MIN = 5
HEADWAY_MAX = 20

N_TRIALS = 50


# ------------------------------------------------------------------
# SAFETY CHECK
# ------------------------------------------------------------------
# Current objective_function uses lists indexed by line_id,
# therefore line IDs must currently be consecutive: 0, 1, 2, ...

expected_line_ids = list(range(len(line_ids)))

if line_ids != expected_line_ids:
    raise ValueError(
        "Current optimization structure requires consecutive line IDs "
        f"starting at 0. Found: {line_ids}"
    )


# ------------------------------------------------------------------
# OPTUNA OBJECTIVE
# ------------------------------------------------------------------

def optuna_objective(trial):

    # --------------------------------------------------------------
    # Decision variable 1: start time of each line
    # --------------------------------------------------------------

    start_times = [
        trial.suggest_int(
            f"start_{line_id}",
            START_TIME_MIN,
            START_TIME_MAX
        )
        for line_id in line_ids
    ]

    print("\n----------------------------------------")
    print(f"Trial {trial.number}")
    print("Start times:", start_times)


    # --------------------------------------------------------------
    # Decision variable 2: headway of each line
    # --------------------------------------------------------------

    headway_list = [
        trial.suggest_int(
            f"headway_{line_id}",
            HEADWAY_MIN,
            HEADWAY_MAX
        )
        for line_id in line_ids
    ]

    print("Headways:", headway_list)
    print("----------------------------------------")


    # --------------------------------------------------------------
    # Evaluate solution
    # --------------------------------------------------------------

    try:

        (
            cost,
            demand_sum_dict_TRA,
            combined_dict_new_opt_TRA,
            demand_check_g1g2_TRA
        ) = objective_function(
            start_times,
            headway_list
        )

        cost = float(cost)

        print(f"Trial {trial.number} cost: {cost:.4f}")


    except Exception as e:

        print(
            f"[ERROR] Trial {trial.number} failed: {e}"
        )

        cost = float("inf")

        demand_sum_dict_TRA = None
        combined_dict_new_opt_TRA = None
        demand_check_g1g2_TRA = None


    # --------------------------------------------------------------
    # Save trial results
    # --------------------------------------------------------------

    trial_result = {
        "trial_number": trial.number,
        "cost": cost
    }

    # Add start times individually
    for line_id, start in zip(
        line_ids,
        start_times
    ):
        trial_result[
            f"start_{line_id}"
        ] = start

    # Add headways individually
    for line_id, headway in zip(
        line_ids,
        headway_list
    ):
        trial_result[
            f"headway_{line_id}"
        ] = headway

    # Demand conservation check
    trial_result[
        "assigned_demand"
    ] = demand_check_g1g2_TRA

    all_trials_results.append(
        trial_result
    )

    return cost



#%% CREATE OPTUNA STUDY

study = optuna.create_study(
    direction="minimize"
)


#%% RUN OPTIMIZATION

study.optimize(
    optuna_objective,
    n_trials=N_TRIALS
)


#%% OPTUNA RESULTS

print("\n========================================")
print("OPTUNA OPTIMIZATION COMPLETED")
print("========================================")

print(
    f"Best objective value: "
    f"{study.best_value:.4f}"
)

print("\nOptimal parameters:")

for key, value in study.best_params.items():
    print(
        f"{key}: {value}"
    )


#%% BEST SOLUTION

best_start_times = [
    study.best_params[
        f"start_{line_id}"
    ]
    for line_id in line_ids
]

best_headway_list = [
    study.best_params[
        f"headway_{line_id}"
    ]
    for line_id in line_ids
]


print("\nBest start times:")
print(best_start_times)

print("\nBest headways:")
print(best_headway_list)


#%% TRIAL RESULTS DATAFRAME

TRA_df_trials = pd.DataFrame(
    all_trials_results
)

print("\nNumber of completed trials:")
print(len(TRA_df_trials))


