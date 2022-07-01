import skfuzzy as fuzz
from skfuzzy import control as ctrl
import numpy as np
import matplotlib.pyplot as plt


def v(d, a):
    return np.sqrt((d * 9.81) / np.sin(2 * np.radians(a)))


def main():
    x_distance = np.arange(1, 100, 5)
    x_angle = np.arange(1, 90, 1)

    distance = ctrl.Antecedent(x_distance, 'distance')
    angle = ctrl.Antecedent(x_angle, 'angle')

    velocity = ctrl.Consequent(np.arange(0, 100, 1), 'velocity')

    distance.automf(3)
    angle.automf(5)
    velocity.automf(5)
    # poor
    # mediocre
    # average
    # decent
    # good

    rules = [
        ctrl.Rule(distance['poor'], velocity['poor']),
        ctrl.Rule(distance['average'] & (angle['mediocre'] | angle['average'] | angle['decent']), velocity['mediocre']),
        ctrl.Rule(distance['average'] & (angle['poor'] | angle['good']), velocity['average']),
        ctrl.Rule(distance['good'] & (angle['mediocre'] | angle['average'] | angle['decent']), velocity['mediocre']),
        ctrl.Rule(distance['good'] & (angle['poor'] | angle['good']), velocity['good']),
    ]

    velocity_ctrl = ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules=rules))

    mse = 0
    i = 0
    preds = []
    for ang in x_angle:
        for dst in x_distance:
            i += 1
            true = v(dst, ang)
            velocity_ctrl.input['distance'] = dst
            velocity_ctrl.input['angle'] = ang
            velocity_ctrl.compute()
            preds.append(velocity_ctrl.output['velocity'])

            mse += (true - velocity_ctrl.output['velocity']) ** 2
    mse /= i
    print(f'MSE: {mse}')

    X, Y = np.meshgrid(x_distance, x_angle)
    Z = v(X, Y)

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')
    ax.set_title('Prawdziwa funkcja mocu rzutu')
    ax.set_xlabel('dystans')
    ax.set_ylabel('kat')
    ax.set_zlabel('moc rzutu')

    Z = np.array(preds).reshape(Z.shape)
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')
    ax.set_title('Predykcja funkcji mocu rzutu')
    ax.set_xlabel('dystans')
    ax.set_ylabel('kat')
    ax.set_zlabel('moc rzutu')

    plt.show()


if __name__ == '__main__':
    main()
