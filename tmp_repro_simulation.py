from src.scenarios.scenario_simulator import ScenarioSimulator, ScenarioInput

sim = ScenarioSimulator()
scenario = ScenarioInput()
for i in range(3):
    print(f"RUN {i+1}")
    res = sim.run_scenario_simulation(scenario)
    print('ACTION:', res['recommendations']['action'])
    print('DAYS_TO_STOCKOUT:', res['state_details']['days_to_stockout'])
    print('INVENTORY:', res['state_details']['current_inventory'])
    print('-' * 40)
