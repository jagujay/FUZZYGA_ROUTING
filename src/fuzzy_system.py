import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt


def create_risk_assessment_system():
    """
    Creates and configures the Fuzzy Inference System for assessing road segment risk.

    Returns:
        risk_assessment_ctrl (ctrl.ControlSystemSimulation): The fuzzy control system simulation.
    """
    # Define the universe of discourse for each variable
    # Antecedents (Inputs)
    traffic = ctrl.Antecedent(np.arange(0, 101, 1), 'traffic')  # 0-100%
    weather = ctrl.Antecedent(np.arange(0, 11, 1), 'weather')  # Scale 1-10
    road_quality = ctrl.Antecedent(np.arange(0, 11, 1), 'road_quality')  # Scale 1-10

    # Consequent (Output)
    segment_risk = ctrl.Consequent(np.arange(0, 11, 1), 'segment_risk')  # Scale 0-10

    # Generate fuzzy membership functions
    # Traffic Membership Functions
    traffic['low'] = fuzz.trimf(traffic.universe, [0, 0, 50])
    traffic['medium'] = fuzz.trimf(traffic.universe, [20, 50, 80])
    traffic['high'] = fuzz.trimf(traffic.universe, [60, 100, 100])

    # Weather Membership Functions
    weather['bad'] = fuzz.trimf(weather.universe, [0, 0, 5])
    weather['okay'] = fuzz.trimf(weather.universe, [3, 5, 8])
    weather['good'] = fuzz.trimf(weather.universe, [6, 10, 10])

    # Road Quality Membership Functions
    road_quality['poor'] = fuzz.trimf(road_quality.universe, [0, 0, 5])
    road_quality['average'] = fuzz.trimf(road_quality.universe, [3, 5, 8])
    road_quality['good'] = fuzz.trimf(road_quality.universe, [6, 10, 10])

    # Segment Risk Membership Functions
    segment_risk['low'] = fuzz.trimf(segment_risk.universe, [0, 0, 4])
    segment_risk['medium'] = fuzz.trimf(segment_risk.universe, [2, 5, 8])
    segment_risk['high'] = fuzz.trimf(segment_risk.universe, [6, 10, 10])

    # Define the fuzzy rules (the "expert knowledge")
    rule1 = ctrl.Rule(traffic['high'] | weather['bad'], segment_risk['high'])
    rule2 = ctrl.Rule(traffic['medium'] & road_quality['poor'], segment_risk['high'])
    rule3 = ctrl.Rule(traffic['high'] & weather['okay'], segment_risk['medium'])
    rule4 = ctrl.Rule(traffic['low'] & road_quality['good'] & weather['good'], segment_risk['low'])
    rule5 = ctrl.Rule(traffic['medium'] | road_quality['average'], segment_risk['medium'])
    rule6 = ctrl.Rule(road_quality['poor'], segment_risk['high'])

    # Create the control system
    risk_assessment_system = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6])
    risk_assessment_ctrl = ctrl.ControlSystemSimulation(risk_assessment_system)

    print("Fuzzy Inference System created successfully.")
    return risk_assessment_ctrl


def visualize_membership_functions():
    """
    Generates and saves plots of the membership functions for visualization.
    """
    # This is a helper function to show what the fuzzy sets look like
    # It re-creates the variables to plot them.
    traffic = ctrl.Antecedent(np.arange(0, 101, 1), 'traffic')
    weather = ctrl.Antecedent(np.arange(0, 11, 1), 'weather')
    segment_risk = ctrl.Consequent(np.arange(0, 11, 1), 'segment_risk')

    traffic['low'] = fuzz.trimf(traffic.universe, [0, 0, 50])
    traffic['medium'] = fuzz.trimf(traffic.universe, [20, 50, 80])
    traffic['high'] = fuzz.trimf(traffic.universe, [60, 100, 100])

    weather['bad'] = fuzz.trimf(weather.universe, [0, 0, 5])
    weather['okay'] = fuzz.trimf(weather.universe, [3, 5, 8])
    weather['good'] = fuzz.trimf(weather.universe, [6, 10, 10])

    segment_risk['low'] = fuzz.trimf(segment_risk.universe, [0, 0, 4])
    segment_risk['medium'] = fuzz.trimf(segment_risk.universe, [2, 5, 8])
    segment_risk['high'] = fuzz.trimf(segment_risk.universe, [6, 10, 10])

    # Plotting
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(8, 9))

    ax0.plot(traffic.universe, traffic['low'].mf, 'b', linewidth=1.5, label='Low')
    ax0.plot(traffic.universe, traffic['medium'].mf, 'g', linewidth=1.5, label='Medium')
    ax0.plot(traffic.universe, traffic['high'].mf, 'r', linewidth=1.5, label='High')
    ax0.set_title('Traffic Density')
    ax0.legend()

    ax1.plot(weather.universe, weather['bad'].mf, 'b', linewidth=1.5, label='Bad')
    ax1.plot(weather.universe, weather['okay'].mf, 'g', linewidth=1.5, label='Okay')
    ax1.plot(weather.universe, weather['good'].mf, 'r', linewidth=1.5, label='Good')
    ax1.set_title('Weather Condition')
    ax1.legend()

    ax2.plot(segment_risk.universe, segment_risk['low'].mf, 'b', linewidth=1.5, label='Low')
    ax2.plot(segment_risk.universe, segment_risk['medium'].mf, 'g', linewidth=1.5, label='Medium')
    ax2.plot(segment_risk.universe, segment_risk['high'].mf, 'r', linewidth=1.5, label='High')
    ax2.set_title('Segment Risk')
    ax2.legend()

    # Turn off top/right axes
    for ax in (ax0, ax1, ax2):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

    plt.tight_layout()
    plt.savefig('../results/membership_functions.png')
    plt.show()


# This block allows you to run this script directly to test its functionality
if __name__ == "__main__":
    # 1. Create the FIS
    risk_system = create_risk_assessment_system()

    # 2. Test the system with sample inputs (simulating a "bad" scenario)
    risk_system.input['traffic'] = 85.0  # High traffic
    risk_system.input['weather'] = 3.5  # Bad weather
    risk_system.input['road_quality'] = 4.0  # Poor road quality

    # 3. Compute the result
    risk_system.compute()

    # 4. Print the final crisp output value
    print(f"\nCalculated Segment Risk: {risk_system.output['segment_risk']:.2f}")

    # 5. Visualize the membership functions
    print("\nDisplaying membership function plots...")
    visualize_membership_functions()