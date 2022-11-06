import set_up as s
import step as step
import computing as comp
import numpy as np
import matplotlib.pyplot as plt
import cmasher as cmr

def init():
    previous_velocities = np.zeros(s.vector_shape)
    time = 0.0

    for i in range(s.PICTURE_NUM):
        time += s.dt
        current_velocities = step.step(previous_velocities, time)
        previous_velocities = current_velocities

        # Plot
        curl = comp.curl_2d(current_velocities)
        plt.contourf(
            s.X,
            s.Y,
            curl,
            cmap=cmr.iceburn,
            levels=100,
        )
        plt.quiver(
            s.X,
            s.Y,
            current_velocities[..., 0],
            current_velocities[..., 1],
            color="red",
        )

        plt.draw()
        plt.pause(0.0001)
        number = str(i)
        if i in range(1, 10):
            number = "0" + str(i)
        photo_name = "fluid_step{}.png"
        plt.savefig(photo_name.format(number))
        plt.clf()
        print(f"Picture {i} of {s.PICTURE_NUM}")

if __name__ == "__main__":
    plt.style.use("dark_background")
    plt.figure(figsize=(4, 4), dpi=150)
    init()



