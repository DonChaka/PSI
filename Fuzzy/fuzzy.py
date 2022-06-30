import skfuzzy as fuzz
from skfuzzy import control as ctrl
import numpy as np


def v(d, a):
    return np.sqrt((d * 9.81) / np.sin(2 * a))


def main():
    x_distance = np.arange(0, 100, 0.1)
    x_angle = np.arange(0.01, np.pi / 2, 0.1)

    distance = ctrl.Antecedent(x_distance, 'distance')
    angle = ctrl.Antecedent(x_angle, 'angle')

    velocity = ctrl.Consequent(np.arange(0, 100, 1), 'velocity')

    distance.automf(3)
    angle.automf(3)
    velocity.automf(3)

    rules = [
        ctrl.Rule(distance['poor'] & angle['poor'], velocity['poor']),
        ctrl.Rule(distance['poor'] & angle['good'], velocity['poor']),
        ctrl.Rule(distance['average'] & angle['average'], velocity['poor']),
        ctrl.Rule(distance['good'] & angle['poor'], velocity['good']),
        ctrl.Rule(distance['good'] & angle['good'], velocity['good'])
    ]

    # rules = [
    #     ctrl.Rule(distance['poor'] & angle['poor'], velocity['poor']),
    #     ctrl.Rule(distance['poor'] & angle['average'], velocity['poor']),
    #     ctrl.Rule(distance['poor'] & angle['good'], velocity['poor']),
    #     ctrl.Rule(distance['average'] & angle['poor'], velocity['average']),
    #     ctrl.Rule(distance['average'] & angle['average'], velocity['poor']),
    #     ctrl.Rule(distance['average'] & angle['good'], velocity['average']),
    #     ctrl.Rule(distance['good'] & angle['poor'], velocity['good']),
    #     ctrl.Rule(distance['good'] & angle['average'], velocity['poor']),
    #     ctrl.Rule(distance['good'] & angle['good'], velocity['good']),
    # ]

    velocity_ctrl = ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules=rules))

    mse = 0
    i = 0
    for dst in x_distance:
        for ang in x_angle:
            i += 1
            true = v(dst, ang)
            velocity_ctrl.input['distance'] = dst
            velocity_ctrl.input['angle'] = ang
            velocity_ctrl.compute()

            mse += (true - velocity_ctrl.output['velocity']) ** 2

    mse /= i

    test_values = (
        (10, 10),
        (10, 85),
        (25, 10),
        (25, 85),
        (50, 50),
        (50, 10),
        (75, 90),
        (75, 85),
        (100, 10),
        (100, 85)
    )

    for dst, ang in test_values:
        true = v(dst, np.radians(ang))
        velocity_ctrl.input['distance'] = dst
        velocity_ctrl.input['angle'] = np.radians(ang)
        velocity_ctrl.compute()

        print(f"Dystans={dst}, Kat={ang}, Oczekiwana predkosc={true}, przewidziana predkosc={velocity_ctrl.output['velocity']:.2f}")

    print(f'MSE dla predyktora wynioslo {mse}')

    velocity.view(sim=velocity_ctrl)

if __name__ == '__main__':
    main()
