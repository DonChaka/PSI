import itertools

import matplotlib.pyplot as plt
import numpy as np
from skfuzzy import control as ctrl
from skfuzzy.control import Rule
from itertools import product


def v(d, a, k, m):
    return (d * k) / (m * np.sin(2 * np.radians(a)) * np.exp(-2 * np.sin(np.radians(a))))


def add_3d_plot(ax1, ax2, controller: ctrl.ControlSystemSimulation, x_distance, x_angle, air_resistance, mass):
    preds = []
    for angle, dist in product(x_angle, x_distance):
        controller.input['distance'] = dist
        controller.input['angle'] = angle
        controller.input['air_resistance'] = air_resistance
        controller.input['mass'] = mass
        controller.compute()
        preds.append(controller.output['velocity'])

    X, Y = np.meshgrid(x_distance, x_angle)

    Z = v(X, Y, air_resistance, mass)
    ax2.plot_surface(X, Y, Z, rstride=1, cstride=1,
                     cmap='viridis', edgecolor='none')
    ax2.set_title(f'TRUE k={air_resistance}, m={mass}')
    ax2.set_xlabel('dystans')
    ax2.set_ylabel('kat')
    ax2.set_zlabel('moc rzutu')

    preds = np.array(preds).reshape(Z.shape)
    ax1.plot_surface(X, Y, preds, rstride=1, cstride=1,
                     cmap='viridis', edgecolor='none')
    ax1.set_title(f'PRED k={air_resistance}, m={mass}')
    ax1.set_xlabel('dystans')
    ax1.set_ylabel('kat')
    ax1.set_zlabel('moc rzutu')


def main():
    x_distance = np.arange(1, 100, 5)
    x_angle = np.arange(1, 90, 5)
    x_air_resistance = np.arange(0.1, 1, 0.1)
    x_mass = np.arange(1, 10, 1)

    distance = ctrl.Antecedent(x_distance, 'distance')
    angle = ctrl.Antecedent(x_angle, 'angle')
    air_resistance = ctrl.Antecedent(x_air_resistance, 'air_resistance')
    mass = ctrl.Antecedent(x_mass, 'mass')

    velocity = ctrl.Consequent(np.arange(0, 1000, 50), 'velocity')

    distance.automf(3)
    angle.automf(3)
    air_resistance.automf(3)
    mass.automf(3)
    velocity.automf(7)
    # dismal
    # poor
    # mediocre
    # average
    # decent
    # good
    # excellent

    rules = [
        Rule(air_resistance['poor'] & mass['poor'] & (distance['poor'] | angle['average']), velocity['dismal']),
        Rule(air_resistance['poor'] & mass['poor'] & (angle['poor'] | angle['good']) & distance['average'],
             velocity['mediocre']),
        Rule(air_resistance['poor'] & mass['poor'] & (angle['poor'] | angle['good']) & distance['good'],
             velocity['average']),

        Rule(air_resistance['poor'] & mass['average'], velocity['dismal']),

        Rule(air_resistance['poor'] & mass['good'], velocity['dismal']),

        Rule(air_resistance['average'] & mass['poor'] & (distance['poor'] | angle['average']), velocity['mediocre']),
        Rule(air_resistance['average'] & mass['poor'] & (angle['poor'] | angle['good']) & distance['average'],
             velocity['good']),
        Rule(air_resistance['average'] & mass['poor'] & (angle['poor'] | angle['good']) & distance['good'],
             velocity['excellent']),

        Rule(air_resistance['average'] & mass['average'] & (distance['poor'] | angle['average']), velocity['dismal']),
        Rule(air_resistance['average'] & mass['average'] & (angle['poor'] | angle['good']) & distance['average'],
             velocity['poor']),
        Rule(air_resistance['average'] & mass['average'] & (angle['poor'] | angle['good']) & distance['good'],
             velocity['average']),

        Rule(air_resistance['average'] & mass['good'] & (distance['poor'] | angle['average']), velocity['dismal']),
        Rule(air_resistance['average'] & mass['good'] & (angle['poor'] | angle['good']) & distance['average'],
             velocity['dismal']),
        Rule(air_resistance['average'] & mass['good'] & (angle['poor'] | angle['good']) & distance['good'],
             velocity['poor']),

        Rule(air_resistance['good'] & mass['poor'] & (distance['poor'] | angle['average']), velocity['good']),
        Rule(air_resistance['good'] & mass['poor'] & (angle['poor'] | angle['good']) & distance['average'],
             velocity['excellent']),
        Rule(air_resistance['good'] & mass['poor'] & (angle['poor'] | angle['good']) & distance['good'],
             velocity['excellent']),

        Rule(air_resistance['good'] & mass['average'] & (distance['poor'] | angle['average']), velocity['dismal']),
        Rule(air_resistance['good'] & mass['average'] & (angle['poor'] | angle['good']) & distance['average'],
             velocity['average']),
        Rule(air_resistance['good'] & mass['average'] & (angle['poor'] | angle['good']) & distance['good'],
             velocity['good']),

        Rule(air_resistance['good'] & mass['good'] & (distance['poor'] | angle['average']), velocity['dismal']),
        Rule(air_resistance['good'] & mass['good'] & (angle['poor'] | angle['good']) & distance['average'],
             velocity['poor']),
        Rule(air_resistance['good'] & mass['good'] & (angle['poor'] | angle['good']) & distance['good'],
             velocity['mediocre']),
    ]

    # rules = [
    #     ctrl.Rule((angle['good'] | angle['poor']) & distance['poor'] & air_resistance['good'] & mass['good'], velocity['dismal']),
    #     ctrl.Rule(angle['poor'] & distance['poor'] & air_resistance['poor'] & mass['poor'], velocity['dismal']),
    #
    #     ctrl.Rule(((air_resistance['poor'] & mass['poor']) | ((air_resistance['average']) & mass['average']) | (air_resistance['good'] & mass['good'])) & angle['average'] & (distance['poor'] | distance['average']), velocity['dismal']),
    #     ctrl.Rule(((air_resistance['poor'] & mass['poor']) | ((air_resistance['average']) & mass['average']) | (air_resistance['good'] & mass['good'])) & (angle['good'] | angle['poor']) & distance['average'], velocity['poor']),
    #     ctrl.Rule(((air_resistance['poor'] & mass['poor']) | ((air_resistance['average']) & mass['average']) | (air_resistance['good'] & mass['good'])) & (angle['good'] | angle['poor']) & distance['good'], velocity['mediocre']),
    #
    #     ctrl.Rule((air_resistance['poor'] & (mass['average'] | mass['good'])) | (air_resistance['average'] & mass['good']), velocity['dismal']),
    #
    #     ctrl.Rule(air_resistance['average'] & mass['poor'] & angle['poor'] & distance['poor'], velocity['dismal']),
    #     ctrl.Rule(air_resistance['average'] & mass['poor'] & angle['poor'] & (distance['average'] | distance['good']), velocity['excellent']),
    #     ctrl.Rule((air_resistance['average'] & mass['poor']) & angle['average'] & distance['poor'], velocity['poor']),
    #     ctrl.Rule((air_resistance['average'] & mass['poor']) & (angle['poor'] & angle['good']) & distance['average'], velocity['average']),
    #     ctrl.Rule((air_resistance['average'] & mass['poor']) & (angle['poor'] & angle['good']) & distance['poor'], velocity['good']),
    #
    #     ctrl.Rule(air_resistance['good'] & mass['poor'] & angle['poor'] & distance['poor'], velocity['dismal']),
    #     ctrl.Rule(air_resistance['good'] & mass['poor'] & angle['poor'] & distance['average'], velocity['good']),
    #     ctrl.Rule((air_resistance['good'] & mass['poor']) & angle['average'] & distance['poor'], velocity['poor']),
    #     ctrl.Rule((air_resistance['good'] & mass['poor']) & (angle['poor'] & angle['good']) & distance['average'], velocity['decent']),
    #     ctrl.Rule((air_resistance['good'] & mass['poor']) & (angle['poor'] & angle['good']) & distance['poor'], velocity['excellent']),
    #
    #     ctrl.Rule((air_resistance['good'] & mass['average']) & angle['average'] & distance['poor'], velocity['dismal']),
    #     ctrl.Rule((air_resistance['good'] & mass['average']) & (angle['poor'] & angle['good']) & distance['average'], velocity['poor']),
    #     ctrl.Rule((air_resistance['good'] & mass['average']) & (angle['poor'] & angle['good']) & distance['poor'], velocity['mediocre']),
    #
    #     ctrl.Rule(((air_resistance['average'] | air_resistance['good']) & mass['poor']) & (angle['poor'] | angle['good']), velocity['good']),
    #     ctrl.Rule(((air_resistance['average'] | air_resistance['good']) & mass['poor']) & angle['average'], velocity['mediocre']),
    #
    #     ctrl.Rule((air_resistance['good'] & mass['average']) & (angle['poor'] | angle['good']), velocity['mediocre']),
    #     ctrl.Rule((air_resistance['good'] & mass['average']) & (angle['average']), velocity['poor']),
    #
    #
    #     ctrl.Rule((air_resistance['good'] & mass['poor']) & angle['good'] & distance['average'], velocity['mediocre']),
    #     ctrl.Rule((air_resistance['good'] & mass['average']) & angle['good'] & distance['good'], velocity['good']),
    # ]

    # rules = [
    #     ctrl.Rule(distance['poor'], velocity['poor']),
    #     ctrl.Rule(mass['good'] | mass['average'], velocity['poor']),
    #
    #     ctrl.Rule(distance['average'] & (angle['mediocre'] | angle['average'] | angle['decent']) & mass['poor'] & air_resistance['poor'], velocity['mediocre']),
    #     ctrl.Rule(distance['average'] & (angle['mediocre'] | angle['average'] | angle['decent']) & air_resistance['average'], velocity['average']),
    #     ctrl.Rule(distance['average'] & (angle['mediocre'] | angle['average'] | angle['decent']) & air_resistance['good'], velocity['decent']),
    #
    #     ctrl.Rule(distance['average'] & (angle['poor'] | angle['good']) & mass['poor'] & air_resistance['poor'], velocity['average']),
    #     ctrl.Rule(distance['average'] & (angle['poor'] | angle['good']) & air_resistance['average'], velocity['decent']),
    #     ctrl.Rule(distance['average'] & (angle['poor'] | angle['good']) & air_resistance['good'], velocity['good']),
    #
    #     ctrl.Rule(distance['good'] & (angle['mediocre'] | angle['average'] | angle['decent']) & air_resistance['poor'], velocity['mediocre']),
    #     ctrl.Rule(distance['good'] & (angle['mediocre'] | angle['average'] | angle['decent']) & air_resistance['average'], velocity['average']),
    #     ctrl.Rule(distance['good'] & (angle['mediocre'] | angle['average'] | angle['decent']) & air_resistance['good'], velocity['decent']),
    #
    #     ctrl.Rule(distance['good'] & (angle['poor'] | angle['good']), velocity['good']),
    # ]

    velocity_ctrl = ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules=rules))

    mse = 0
    i = 0
    preds = []
    for ang, dst, air, m in product(x_angle, x_distance, x_air_resistance, x_mass):
        i += 1
        true = v(dst, ang, air, m)

        velocity_ctrl.input['distance'] = dst
        velocity_ctrl.input['angle'] = ang
        velocity_ctrl.input['air_resistance'] = air
        velocity_ctrl.input['mass'] = m

        try:
            velocity_ctrl.compute()
        except ValueError:
            print(f'Error for a:{ang}, d:{dst}, k:{air}, m:{m} - expected {true}')
            continue
        preds.append(velocity_ctrl.output['velocity'])

        mse += (true - velocity_ctrl.output['velocity']) ** 2

    mse /= i
    print(f'MSE: {mse}')

    fig = plt.figure(figsize=(20, 20))

    axs = [fig.add_subplot(3, 6, n, projection='3d') for n in range(1, 19)]
    add_3d_plot(axs[0], axs[1], velocity_ctrl, x_distance, x_angle, 0.1, 2)
    add_3d_plot(axs[2], axs[3], velocity_ctrl, x_distance, x_angle, 0.1, 11)
    add_3d_plot(axs[4], axs[5], velocity_ctrl, x_distance, x_angle, 0.1, 20)
    add_3d_plot(axs[6], axs[7], velocity_ctrl, x_distance, x_angle, 0.45, 2)
    add_3d_plot(axs[8], axs[9], velocity_ctrl, x_distance, x_angle, 0.45, 11)
    add_3d_plot(axs[10], axs[11], velocity_ctrl, x_distance, x_angle, 0.45, 20)
    add_3d_plot(axs[12], axs[13], velocity_ctrl, x_distance, x_angle, 0.8, 2)
    add_3d_plot(axs[14], axs[15], velocity_ctrl, x_distance, x_angle, 0.8, 11)
    add_3d_plot(axs[16], axs[17], velocity_ctrl, x_distance, x_angle, 0.8, 20)

    # axs = [fig.add_subplot(3, 3, n, projection='3d') for n in range(1, 10)]
    # add_3d_plot(axs[1], axs[0], velocity_ctrl, x_distance, x_angle, 0.1, 1)
    # add_3d_plot(axs[1], axs[1], velocity_ctrl, x_distance, x_angle, 0.1, 5)
    # add_3d_plot(axs[1], axs[2], velocity_ctrl, x_distance, x_angle, 0.1, 10)
    # add_3d_plot(axs[1], axs[3], velocity_ctrl, x_distance, x_angle, 0.45, 1)
    # add_3d_plot(axs[1], axs[4], velocity_ctrl, x_distance, x_angle, 0.45, 5)
    # add_3d_plot(axs[1], axs[5], velocity_ctrl, x_distance, x_angle, 0.45, 10)
    # add_3d_plot(axs[1], axs[6], velocity_ctrl, x_distance, x_angle, 0.8, 1)
    # add_3d_plot(axs[1], axs[7], velocity_ctrl, x_distance, x_angle, 0.8, 5)
    # add_3d_plot(axs[1], axs[8], velocity_ctrl, x_distance, x_angle, 0.8, 10)

    plt.show()


if __name__ == '__main__':
    main()
