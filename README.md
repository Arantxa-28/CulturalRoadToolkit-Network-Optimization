**ToolKit for Headway and Timetable Optimization based on heuristic search using Optuna.**

**Transit Timetable Optimization with Optuna**

This Python code optimizes the **start times** and **headways** of transit lines using **Optuna**. 
For each trial, Optuna proposes a combination of timetable parameters (headways and start times), the model evaluates its total cost (passenger cost and operator cost), and the study searches for the combination with the lowest cost.


**Important:**
In the current implementation, line IDs must be consecutive integers starting at 0 (for example, 0, 1, 2).

**How to run**
1. Place the required CSV files in the data folder.
2. Install the Python dependencies and configure CPLEX if needed.
3. Open the code and locate the **OPTUNA PARAMETERS** section near the end.
4. Set the number of trials and the search ranges:

START_TIME_MIN = 0
START_TIME_MAX = 20

HEADWAY_MIN = 5
HEADWAY_MAX = 20

N_TRIALS = 50

Each trial tests a start time and a headway for every line. Increasing N_TRIALS allows Optuna to evaluate more candidate combinations but generally increases runtime.

**Note on perception parameters:**
The PathPara.csv file contains indicative values for passenger perception factors (Denmark context) used to calculate generalized travel costs components.
These values are provided as a reference and can be adjusted to reflect the specific characteristics, assumptions, and local context of each case study.
**User Heterogeneity:** The current implementation accounts for heterogeneous passenger demand by considering two user groups: 90% regular passengers and 10% passengers with mobility disabilities. This distinction allows the model to represent different passenger needs and travel perceptions within the network optimization process.

**Results**
After the optimization, the script prints:
- The lowest objective-function value found
- The best start time for each line
- The best headway for each line
- The number of trials recorded
The main results are also available as Python variables:
- study.best_value: best total cost
- study.best_params: best parameters by line
- best_start_times: list of optimized start times
- best_headway_list: list of optimized headways
- TRA_df_trials: DataFrame containing trial costs, tested parameters, and the demand-assignment check
Note: The current script creates TRA_df_trials in memory, but it does not automatically export this DataFrame to a CSV file.
